"""Position-goal command matching the IsaacLab/C++ body-frame goal error."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import wrap_to_pi


@dataclass(kw_only=True)
class PositionGoalCommandCfg(CommandTermCfg):
  entity_name: str
  dist_range: tuple[float, float] = (0.0, 0.0)
  bearing_range: tuple[float, float] = (-math.pi, math.pi)
  yaw_range: tuple[float, float] = (0.0, 0.0)

  def build(self, env) -> "PositionGoalCommand":
    return PositionGoalCommand(self, env)


class PositionGoalCommand(CommandTerm):
  cfg: PositionGoalCommandCfg

  def __init__(self, cfg: PositionGoalCommandCfg, env):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.target_pos_w = torch.zeros(env.num_envs, 2, device=env.device)
    self.target_yaw_w = torch.zeros(env.num_envs, device=env.device)
    self._state_error = torch.zeros(env.num_envs, 3, device=env.device)
    self.metrics["error_pos_xy"] = torch.zeros(env.num_envs, device=env.device)
    self.metrics["error_yaw"] = torch.zeros(env.num_envs, device=env.device)

  @property
  def command(self) -> torch.Tensor:
    return self._state_error

  def _update_metrics(self) -> None:
    dist = torch.linalg.vector_norm(self._state_error[:, :2], dim=-1)
    max_step = self.cfg.resampling_time_range[1] / self._env.step_dt
    self.metrics["error_pos_xy"] += dist / max_step
    self.metrics["error_yaw"] += self._state_error[:, 2].abs() / max_step

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    d = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.dist_range)
    bearing = torch.empty_like(d).uniform_(*self.cfg.bearing_range)
    yaw_offset = torch.empty_like(d).uniform_(*self.cfg.yaw_range)
    root_pos = self.robot.data.root_link_pos_w[env_ids, :2]
    root_yaw = self.robot.data.heading_w[env_ids]
    self.target_pos_w[env_ids, 0] = root_pos[:, 0] + d * torch.cos(bearing)
    self.target_pos_w[env_ids, 1] = root_pos[:, 1] + d * torch.sin(bearing)
    self.target_yaw_w[env_ids] = wrap_to_pi(root_yaw + yaw_offset)

  def _update_command(self) -> None:
    root_pos = self.robot.data.root_link_pos_w[:, :2]
    yaw = self.robot.data.heading_w
    delta = self.target_pos_w - root_pos
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    self._state_error[:, 0] = c * delta[:, 0] + s * delta[:, 1]
    self._state_error[:, 1] = -s * delta[:, 0] + c * delta[:, 1]
    self._state_error[:, 2] = wrap_to_pi(self.target_yaw_w - yaw)

