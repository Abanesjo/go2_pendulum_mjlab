"""IsaacLab difficulty curriculum knobs for the Go2 pendulum task."""

from __future__ import annotations

import torch

from go2_pendulum_mjlab.tasks.go2_pendulum.constants import DIFFICULTY_PRESETS


def isaac_difficulty(
  env,
  env_ids: torch.Tensor,
  command_name: str,
  pendulum_reset_event_name: str,
  pendulum_limits_event_name: str,
  pendulum_termination_name: str,
  position_termination_name: str,
  push_event_name: str | None = None,
  total_steps: int = 25_000 * 32,
  override_level: int = -1,
) -> dict[str, torch.Tensor]:
  del env_ids
  if override_level >= 1:
    level = int(override_level)
  elif total_steps <= 0:
    level = 5
  else:
    progress = min(1.0, max(0.0, env.common_step_counter / float(total_steps)))
    level = min(5, int(progress * 5.0) + 1)
  preset = DIFFICULTY_PRESETS[level]

  cmd_cfg = env.command_manager.get_term_cfg(command_name)
  cmd_cfg.dist_range = preset["goal_dist"]
  cmd_cfg.bearing_range = preset["goal_bearing"]
  cmd_cfg.yaw_range = preset["goal_yaw"]

  reset_cfg = env.event_manager.get_term_cfg(pendulum_reset_event_name)
  reset_cfg.params["angle_range"] = preset["pendulum_reset"]

  limit_cfg = env.event_manager.get_term_cfg(pendulum_limits_event_name)
  limit_cfg.params["limit_range"] = preset["pendulum_limits"]

  pend_term_cfg = env.termination_manager.get_term_cfg(pendulum_termination_name)
  inner = dict(pend_term_cfg.params["inner"])
  inner_params = dict(inner.get("params", {}))
  inner_params["angle_rad"] = preset["pendulum_terminate_angle"]
  inner["params"] = inner_params
  pend_term_cfg.params["inner"] = inner
  pend_term_cfg.params["duration_s"] = preset["pendulum_terminate_duration"]

  pos_term_cfg = env.termination_manager.get_term_cfg(position_termination_name)
  inner = dict(pos_term_cfg.params["inner"])
  inner_params = dict(inner.get("params", {}))
  inner_params["max_dist"] = preset["position_tolerance"]
  inner["params"] = inner_params
  pos_term_cfg.params["inner"] = inner

  if push_event_name is not None and push_event_name in env.event_manager.active_terms:
    push_cfg = env.event_manager.get_term_cfg(push_event_name)
    push_cfg.params["force_range"] = preset["push_force_xy"]

  return {
    "level": torch.tensor(float(level)),
    "goal_dist_max": torch.tensor(float(preset["goal_dist"][1])),
    "position_tolerance": torch.tensor(float(preset["position_tolerance"])),
  }

