#!/usr/bin/env python3
"""Train the Go2 pendulum MJLab task with RSL-RL."""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import mjlab
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import go2_pendulum_mjlab  # noqa: F401
from go2_pendulum_mjlab.tasks.go2_pendulum import TASK_ID
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.os import dump_yaml, get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder


@dataclass
class TrainConfig:
  env: ManagerBasedRlEnvCfg = field(default_factory=lambda: load_env_cfg(TASK_ID))
  agent: RslRlBaseRunnerCfg = field(default_factory=lambda: load_rl_cfg(TASK_ID))
  device: str | None = None
  resume_file: str | None = None
  video: bool = False
  video_length: int = 200
  video_interval: int = 2000
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])


def _select_device(cfg: TrainConfig) -> str:
  if cfg.device is not None:
    return cfg.device
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    return "cpu"
  return "cuda:0"


def main() -> None:
  cfg = tyro.cli(TrainConfig, config=mjlab.TYRO_FLAGS)
  configure_torch_backends()

  if cfg.gpu_ids is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  elif cfg.gpu_ids != "all":
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in cfg.gpu_ids)
  os.environ.setdefault("MUJOCO_GL", "egl")

  device = _select_device(cfg)
  cfg.env.seed = cfg.agent.seed

  log_root = Path("logs") / "rsl_rl" / cfg.agent.experiment_name
  log_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if cfg.agent.run_name:
    log_name += f"_{cfg.agent.run_name}"
  log_dir = log_root / log_name
  print(f"[INFO] Training {TASK_ID} on {device}; log_dir={log_dir}")

  env = ManagerBasedRlEnv(
    cfg=cfg.env,
    device=device,
    render_mode="rgb_array" if cfg.video else None,
  )
  if cfg.video:
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "train",
      step_trigger=lambda step: step % cfg.video_interval == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )
    print(f"[INFO] Recording train videos to {log_dir / 'videos' / 'train'}")
  env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)
  runner = MjlabOnPolicyRunner(env, asdict(cfg.agent), str(log_dir), device)

  resume_path = None
  if cfg.resume_file is not None:
    resume_path = Path(cfg.resume_file).expanduser()
  elif cfg.agent.resume:
    resume_path = get_checkpoint_path(log_root, cfg.agent.load_run, cfg.agent.load_checkpoint)
  if resume_path is not None:
    print(f"[INFO] Loading checkpoint: {resume_path}")
    runner.load(str(resume_path), map_location=device)

  dump_yaml(log_dir / "params" / "env.yaml", asdict(cfg.env))
  dump_yaml(log_dir / "params" / "agent.yaml", asdict(cfg.agent))

  runner.learn(num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True)
  env.close()


if __name__ == "__main__":
  main()
