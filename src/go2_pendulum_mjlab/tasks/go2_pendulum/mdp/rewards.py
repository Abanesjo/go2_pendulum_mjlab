"""Reward helpers ported from the current IsaacLab task."""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

from go2_pendulum_mjlab.tasks.go2_pendulum.mdp.gait import desired_contact_states, foot_phases, swing_phase_profile


def position_tracking(env, command_name: str, std: float) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return torch.exp(-torch.linalg.vector_norm(command[:, :2], dim=-1) / std)


def yaw_alignment(env, command_name: str, std: float) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return torch.exp(-torch.square(command[:, 2]) / (std * std))


class progress:
  def __init__(self, cfg: RewardTermCfg, env):
    self._command_name = cfg.params["command_name"]
    self._prev_dist = torch.zeros(env.num_envs, device=env.device)
    self._first_step = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)

  def __call__(self, env, command_name: str) -> torch.Tensor:
    del command_name
    command = env.command_manager.get_command(self._command_name)
    assert command is not None
    cur = torch.linalg.vector_norm(command[:, :2], dim=-1)
    delta = torch.where(self._first_step, torch.zeros_like(cur), self._prev_dist - cur)
    self._prev_dist = cur.clone()
    self._first_step[:] = False
    return delta

  def reset(self, env_ids=None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._prev_dist[env_ids] = 0.0
    self._first_step[env_ids] = True


def pendulum_upright(env, asset_cfg: SceneEntityCfg, std: float) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  error = torch.sum(torch.square(asset.data.joint_pos[:, asset_cfg.joint_ids]), dim=1)
  return torch.exp(-error / std)


def pendulum_velocity_l2(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def balanced_movement(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  pend_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
  pend_err = torch.sum(torch.square(pend_pos), dim=1)
  base_speed = torch.linalg.vector_norm(asset.data.root_link_lin_vel_w[:, :2], dim=-1)
  return torch.exp(-pend_err) * base_speed


def _command_active(env, command_name: str | None, threshold: float) -> torch.Tensor:
  if command_name is None:
    return torch.ones(env.num_envs, device=env.device)
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return (torch.linalg.vector_norm(command[:, :2], dim=1) > threshold).float()


def feet_clearance(
  env,
  asset_cfg: SceneEntityCfg,
  period_s: float = 1.0 / 3.0,
  offsets: tuple[float, ...] = (0.5, 0.0, 0.0, 0.5),
  swing_height: float = 0.08,
  stance_height: float = 0.02,
  command_name: str | None = None,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  phase = foot_phases(env, period_s=period_s, offsets=offsets)
  desired = desired_contact_states(phase)
  target_height = swing_height * swing_phase_profile(phase) + stance_height
  foot_height = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2]
  cost = torch.sum(torch.square(target_height - foot_height) * (1.0 - desired), dim=1)
  return cost * _command_active(env, command_name, command_threshold)


def tracking_contacts_shaped_force(
  env,
  sensor_name: str,
  period_s: float = 1.0 / 3.0,
  offsets: tuple[float, ...] = (0.5, 0.0, 0.0, 0.5),
  force_sigma: float = 100.0,
  command_name: str | None = None,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  desired = desired_contact_states(foot_phases(env, period_s=period_s, offsets=offsets))
  force = torch.linalg.vector_norm(sensor.data.force, dim=-1)
  penalty = -torch.sum((1.0 - desired) * (1.0 - torch.exp(-torch.square(force) / force_sigma)), dim=1) / force.shape[1]
  return penalty * _command_active(env, command_name, command_threshold)


def feet_air_time(env, sensor_name: str, target_air_time: float = 0.5, command_name: str | None = None, command_threshold: float = 0.1) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  first_contact = sensor.compute_first_contact(dt=env.step_dt)
  assert sensor.data.last_air_time is not None
  reward = torch.sum((sensor.data.last_air_time - target_air_time) * first_contact.float(), dim=1)
  return reward * _command_active(env, command_name, command_threshold)


def undesired_contacts(env, sensor_name: str, threshold: float = 1.0) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  return torch.sum((torch.linalg.vector_norm(sensor.data.force, dim=-1) > threshold).float(), dim=1)


def action_l2(env) -> torch.Tensor:
  return torch.sum(torch.square(env.action_manager.action), dim=1)


def action_rate_l2(env) -> torch.Tensor:
  return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


def action_acc_l2(env) -> torch.Tensor:
  acc = env.action_manager.action - 2.0 * env.action_manager.prev_action + env.action_manager.prev_prev_action
  return torch.sum(torch.square(acc), dim=1)


def action_soft_bound_l2(env, bound: float = 1.0) -> torch.Tensor:
  excess = torch.clamp(torch.abs(env.action_manager.action) - bound, min=0.0)
  return torch.sum(torch.square(excess), dim=1)


def _ordered_pd_action_term(env, action_name: str):
  term = env.action_manager.get_term(action_name)
  required = ("applied_action", "prev_applied_action", "prev_prev_applied_action", "target_pos")
  for name in required:
    if not hasattr(term, name):
      raise TypeError(f"Action term '{action_name}' is missing '{name}'")
  return term


def applied_action_l2(env, action_name: str = "joint_pos") -> torch.Tensor:
  term = _ordered_pd_action_term(env, action_name)
  return torch.sum(torch.square(term.applied_action), dim=1)


def applied_action_rate_l2(env, action_name: str = "joint_pos") -> torch.Tensor:
  term = _ordered_pd_action_term(env, action_name)
  return torch.sum(torch.square(term.applied_action - term.prev_applied_action), dim=1)


def applied_action_acc_l2(env, action_name: str = "joint_pos") -> torch.Tensor:
  term = _ordered_pd_action_term(env, action_name)
  acc = term.applied_action - 2.0 * term.prev_applied_action + term.prev_prev_applied_action
  return torch.sum(torch.square(acc), dim=1)


def action_filter_residual_l2(env, action_name: str = "joint_pos") -> torch.Tensor:
  term = _ordered_pd_action_term(env, action_name)
  return torch.sum(torch.square(term.raw_action - term.applied_action), dim=1)


def target_pos_rate_l2(env, action_name: str = "joint_pos") -> torch.Tensor:
  term = _ordered_pd_action_term(env, action_name)
  rate = (term.target_pos - term.prev_target_pos) / env.step_dt
  return torch.sum(torch.square(rate), dim=1)


def target_pos_acc_l2(env, action_name: str = "joint_pos") -> torch.Tensor:
  term = _ordered_pd_action_term(env, action_name)
  acc = (term.target_pos - 2.0 * term.prev_target_pos + term.prev_prev_target_pos) / (env.step_dt**2)
  return torch.sum(torch.square(acc), dim=1)


def target_delta_l2(env, action_name: str = "joint_pos") -> torch.Tensor:
  return target_pos_rate_l2(env, action_name=action_name)


def target_delta_delta_l2(env, action_name: str = "joint_pos") -> torch.Tensor:
  return target_pos_acc_l2(env, action_name=action_name)


def joint_actuator_effort_l2(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.qfrc_actuator[:, asset_cfg.joint_ids]), dim=1)


def torque_l2(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  return joint_actuator_effort_l2(env, asset_cfg=asset_cfg)


def torque_rate_l2(env, action_name: str = "joint_pos") -> torch.Tensor:
  term = env.action_manager.get_term(action_name)
  required = ("applied_torque", "prev_applied_torque")
  for name in required:
    if not hasattr(term, name):
      raise TypeError(f"Action term '{action_name}' is missing '{name}'")
  rate = (term.applied_torque - term.prev_applied_torque) / env.step_dt
  return torch.sum(torch.square(rate), dim=1)


def applied_torque_rate_l2(env, action_name: str = "joint_pos") -> torch.Tensor:
  return torque_rate_l2(env, action_name=action_name)


def flat_orientation_reward(env, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  orient_error = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
  return torch.exp(-orient_error / max(std, 1.0e-6))


def base_height_l2(env, target_height: float, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.exp(-torch.square(target_height - asset.data.root_link_pos_w[:, 2]) / (std * std))


def lin_vel_z_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_lin_vel_b[:, 2])


def ang_vel_xy_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)


def early_termination(env) -> torch.Tensor:
  return (env.termination_manager.terminated & ~env.termination_manager.time_outs).float()
