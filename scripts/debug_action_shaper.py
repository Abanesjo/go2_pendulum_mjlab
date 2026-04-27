#!/usr/bin/env python3
"""Exercise the Go2 action shaper with synthetic command jumps."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import go2_pendulum_mjlab  # noqa: F401
from go2_pendulum_mjlab.tasks.go2_pendulum import TASK_ID
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends


LAST_ACTION_SLICE = slice(40, 52)


def _policy_obs(obs: Any) -> torch.Tensor:
  if isinstance(obs, tuple):
    return _policy_obs(obs[0])
  if isinstance(obs, dict):
    return obs["policy"]
  return obs


def _step_obs(step_result: Any) -> torch.Tensor:
  if isinstance(step_result, tuple):
    return _policy_obs(step_result[0])
  return _policy_obs(step_result)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--num-envs", type=int, default=4)
  parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
  parser.add_argument("--steps", type=int, default=12)
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args()

  configure_torch_backends()
  os.environ.setdefault("MUJOCO_GL", "egl")
  torch.manual_seed(args.seed)

  env_cfg = load_env_cfg(TASK_ID, play=False)
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.seed = args.seed
  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)

  try:
    obs = _policy_obs(env.reset())
    term = env.action_manager.get_term("joint_pos")
    action_shape = (env.num_envs, term.action_dim)
    actions = torch.zeros(action_shape, device=env.device)

    raw_max = 0.0
    applied_max = 0.0
    max_delta = 0.0
    applied_differs_from_raw = False
    any_nonfinite = not bool(torch.isfinite(obs).all())
    last_action_matches_applied = True
    last_action_matches_raw = True

    patterns = (
      torch.zeros(action_shape, device=env.device),
      torch.ones(action_shape, device=env.device),
      -torch.ones(action_shape, device=env.device),
    )
    for step in range(args.steps):
      if step < len(patterns):
        actions = patterns[step]
      else:
        actions = 2.0 * torch.rand(action_shape, device=env.device) - 1.0
      step_result = env.step(actions)
      obs = _step_obs(step_result)
      target_delta = torch.max(torch.abs(term.target_pos - term.prev_target_pos)).item()
      max_delta = max(max_delta, target_delta)
      raw_max = max(raw_max, torch.max(torch.abs(term.raw_action)).item())
      applied_max = max(applied_max, torch.max(torch.abs(term.applied_action)).item())
      applied_differs_from_raw = applied_differs_from_raw or not bool(
        torch.allclose(term.raw_action, term.applied_action, atol=1.0e-6, rtol=1.0e-6)
      )
      any_nonfinite = any_nonfinite or not bool(torch.isfinite(obs).all())
      any_nonfinite = any_nonfinite or not bool(torch.isfinite(term.target_pos).all())
      obs_last_action = obs[:, LAST_ACTION_SLICE]
      last_action_matches_applied = last_action_matches_applied and bool(
        torch.allclose(obs_last_action, term.applied_action, atol=1.0e-6, rtol=1.0e-6)
      )
      last_action_matches_raw = last_action_matches_raw and bool(
        torch.allclose(obs_last_action, term.raw_action, atol=1.0e-6, rtol=1.0e-6)
      )

    allowed_delta = term.cfg.max_target_rate * env.step_dt
    print(f"obs shape: {tuple(obs.shape)}")
    print(f"action shape: {action_shape}")
    print(f"raw action max: {raw_max:.6f}")
    print(f"applied action max: {applied_max:.6f}")
    print(f"applied action differs from raw action: {applied_differs_from_raw}")
    print(f"max target position delta per step: {max_delta:.6f}")
    print(f"configured max allowed target delta: {allowed_delta:.6f}")
    print(f"latency steps sampled: {term.latency_steps.detach().cpu().tolist()}")
    print(f"last_action equals applied_action: {last_action_matches_applied}")
    print(f"last_action equals raw_action: {last_action_matches_raw}")
    print(f"any NaN/Inf occurred: {any_nonfinite}")
  finally:
    env.close()


if __name__ == "__main__":
  main()
