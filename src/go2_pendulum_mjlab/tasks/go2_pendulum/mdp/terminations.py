"""Task-specific termination predicates."""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor


def pendulum_fallen(env, asset_cfg: SceneEntityCfg, angle_rad: float) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.linalg.vector_norm(asset.data.joint_pos[:, asset_cfg.joint_ids], dim=-1) > angle_rad


def position_goal_violation(env, command_name: str, max_dist: float) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return torch.linalg.vector_norm(command[:, :2], dim=-1) > max_dist


def body_contact_force(env, sensor_name: str, threshold: float) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  return torch.linalg.vector_norm(sensor.data.force, dim=-1).max(dim=-1).values > threshold


class sustained:
  def __init__(self, cfg: TerminationTermCfg, env):
    inner = cfg.params["inner"]
    self._inner_func = inner["func"]
    for value in inner.get("params", {}).values():
      if isinstance(value, SceneEntityCfg):
        value.resolve(env.scene)
    self._hold_count = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

  def __call__(self, env, inner: dict, duration_s: float = 0.0, grace_period_s: float = 0.0) -> torch.Tensor:
    cond = self._inner_func(env, **inner.get("params", {}))
    duration_steps = int(round(float(duration_s) / env.step_dt))
    grace_steps = int(round(float(grace_period_s) / env.step_dt))
    past_grace = env.episode_length_buf >= grace_steps
    active_cond = cond & past_grace
    self._hold_count = torch.where(active_cond, self._hold_count + 1, torch.zeros_like(self._hold_count))
    if duration_steps <= 0:
      return active_cond
    return (self._hold_count >= duration_steps) & past_grace

  def reset(self, env_ids=None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._hold_count[env_ids] = 0
