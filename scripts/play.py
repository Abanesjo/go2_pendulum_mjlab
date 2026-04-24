#!/usr/bin/env python3
"""Play/export a trained Go2 pendulum MJLab policy."""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import mjlab
import torch
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import go2_pendulum_mjlab  # noqa: F401
from go2_pendulum_mjlab.tasks.go2_pendulum import TASK_ID
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


@dataclass
class PlayConfig:
  checkpoint_file: str | None = None
  agent: str = "trained"
  num_envs: int = 1
  device: str | None = None
  viewer: str = "auto"
  export: bool = False


def main() -> None:
  args = tyro.cli(PlayConfig, config=mjlab.TYRO_FLAGS)
  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env_cfg = load_env_cfg(TASK_ID, play=True)
  env_cfg.scene.num_envs = args.num_envs
  agent_cfg = load_rl_cfg(TASK_ID)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  if args.agent == "zero":
    def policy(_obs):
      return torch.zeros(wrapped.unwrapped.action_space.shape, device=device)
  elif args.agent == "random":
    def policy(_obs):
      return 2.0 * torch.rand(wrapped.unwrapped.action_space.shape, device=device) - 1.0
  else:
    if args.checkpoint_file is None:
      raise ValueError("--checkpoint-file is required for trained play/export")
    checkpoint = Path(args.checkpoint_file).expanduser()
    runner = MjlabOnPolicyRunner(wrapped, asdict(agent_cfg), device=device)
    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)
    if args.export:
      export_dir = checkpoint.parent
      runner.export_policy_to_jit(str(export_dir), "policy.pt")
      runner.export_policy_to_onnx(str(export_dir), "policy.onnx")
      print(f"[INFO] Exported policy.pt and policy.onnx to {export_dir}")

  if args.export:
    env.close()
    return

  if args.viewer == "auto":
    viewer = "native" if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") else "viser"
  else:
    viewer = args.viewer
  if viewer == "native":
    NativeMujocoViewer(wrapped, policy).run()
  elif viewer == "viser":
    ViserPlayViewer(wrapped, policy).run()
  else:
    raise ValueError(f"Unknown viewer: {viewer}")
  env.close()


if __name__ == "__main__":
  main()
