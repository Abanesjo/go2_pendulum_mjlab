"""Observation terms matching the IsaacLab/C++ policy ABI."""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor
from mjlab.utils.lab_api.math import euler_xyz_from_quat, quat_apply_inverse, quat_mul, wrap_to_pi


def _asset(env, asset_cfg: SceneEntityCfg) -> Entity:
  return env.scene[asset_cfg.name]


class finite_diff_base_lin_vel_b:
  """Base-frame linear velocity from finite-differenced root pose."""

  def __init__(self, cfg: ObservationTermCfg, env):
    asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
    self._asset: Entity = env.scene[asset_cfg.name]
    self._prev_pos_w = self._asset.data.root_link_pos_w.clone()
    self._has_prev = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  def __call__(self, env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = self._asset
    pos_w = asset.data.root_link_pos_w.clone()
    vel_w = torch.zeros_like(pos_w)
    valid = self._has_prev
    if torch.any(valid):
      vel_w[valid] = (pos_w[valid] - self._prev_pos_w[valid]) / env.step_dt
    self._prev_pos_w[:] = pos_w
    self._has_prev[:] = True
    return quat_apply_inverse(asset.data.root_link_quat_w, vel_w)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._prev_pos_w[env_ids] = self._asset.data.root_link_pos_w[env_ids]
    self._has_prev[env_ids] = False


def imu_ang_vel_b(env, sensor_name: str = "robot/imu_gyro") -> torch.Tensor:
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, BuiltinSensor)
  return sensor.data


def projected_gravity_from_imu(env, sensor_name: str = "robot/imu_quat") -> torch.Tensor:
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, BuiltinSensor)
  gravity_w = torch.tensor((0.0, 0.0, -1.0), device=env.device).repeat(env.num_envs, 1)
  return quat_apply_inverse(sensor.data, gravity_w)


def goal_error_b(env, command_name: str = "position_goal") -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return command


def joint_pos_rel(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  asset = _asset(env, asset_cfg)
  ids = asset_cfg.joint_ids
  return asset.data.joint_pos[:, ids] - asset.data.default_joint_pos[:, ids]


def joint_vel(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  return _asset(env, asset_cfg).data.joint_vel[:, asset_cfg.joint_ids]


def raw_last_action(env) -> torch.Tensor:
  return env.action_manager.action


def applied_last_action(env, action_name: str = "joint_pos") -> torch.Tensor:
  term = env.action_manager.get_term(action_name)
  if not hasattr(term, "applied_action"):
    raise TypeError(f"Action term '{action_name}' does not expose applied_action")
  return term.applied_action


def clock_inputs(env) -> torch.Tensor:
  step = env.episode_length_buf.to(dtype=torch.float32) * env.step_dt * 3.0
  gait = torch.remainder(step, 1.0)
  foot_indices = torch.stack((gait + 0.5, gait, gait, gait + 0.5), dim=-1)
  r = torch.remainder(foot_indices, 1.0)
  duration = 0.5
  remapped = torch.where(
    r < duration,
    r * (0.5 / duration),
    0.5 + (r - duration) * (0.5 / (1.0 - duration)),
  )
  return torch.sin(2.0 * torch.pi * remapped)


def yaw_from_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
  _, _, yaw = euler_xyz_from_quat(quat)
  return yaw


def noisy_quat(quat: torch.Tensor, roll_pitch_std: float, yaw_std: float = 0.0) -> torch.Tensor:
  from mjlab.utils.lab_api.math import quat_from_euler_xyz

  if roll_pitch_std <= 0.0 and yaw_std <= 0.0:
    return quat
  noise = quat_from_euler_xyz(
    torch.randn(quat.shape[0], device=quat.device) * roll_pitch_std,
    torch.randn(quat.shape[0], device=quat.device) * roll_pitch_std,
    torch.randn(quat.shape[0], device=quat.device) * yaw_std,
  )
  return quat_mul(noise, quat)


def clean_goal_error_from_pose(env, target_xy: torch.Tensor, target_yaw: torch.Tensor, pos_w: torch.Tensor, quat_w: torch.Tensor):
  yaw = yaw_from_quat_wxyz(quat_w)
  delta = target_xy - pos_w[:, :2]
  c = torch.cos(yaw)
  s = torch.sin(yaw)
  return torch.stack(
    (
      c * delta[:, 0] + s * delta[:, 1],
      -s * delta[:, 0] + c * delta[:, 1],
      wrap_to_pi(target_yaw - yaw),
    ),
    dim=-1,
  )
