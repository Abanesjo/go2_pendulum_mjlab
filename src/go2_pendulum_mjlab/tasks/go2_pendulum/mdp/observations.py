"""Observation terms matching the IsaacLab/C++ policy ABI."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from mjlab.entity import Entity
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor
from mjlab.utils.lab_api.math import euler_xyz_from_quat, quat_apply_inverse, quat_mul, wrap_to_pi


def _asset(env, asset_cfg: SceneEntityCfg) -> Entity:
  return env.scene[asset_cfg.name]


class delayed_noisy_observation:
  """Wrap an observation term with per-env delay, packet hold, noise, and bias."""

  def __init__(self, cfg: ObservationTermCfg, env):
    self._env = env
    self._source_params = dict(cfg.params.get("source_params", {}))
    for value in self._source_params.values():
      if isinstance(value, SceneEntityCfg):
        value.resolve(env.scene)
    source_func = cfg.params["source_func"]
    if isinstance(source_func, type):
      self._source = source_func(SimpleNamespace(params=self._source_params), env)
      self._call_source = lambda: self._source(env, **self._source_params)
    else:
      self._source = source_func
      self._call_source = lambda: self._source(env, **self._source_params)

    self._dim = int(cfg.params["dim"])
    self._delay_steps_range = tuple(cfg.params.get("delay_steps_range", (0, 0)))
    self._hold_prob = float(cfg.params.get("hold_prob", 0.0))
    self._noise = float(cfg.params.get("noise", 0.0))
    self._bias = float(cfg.params.get("bias", 0.0))
    lo, hi = self._delay_steps_range
    if lo < 0 or hi < lo:
      raise ValueError(f"Invalid delay_steps_range={self._delay_steps_range}")
    self._buffer = torch.zeros(env.num_envs, hi + 1, self._dim, device=env.device)
    self._delay_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    self._held_value = torch.zeros(env.num_envs, self._dim, device=env.device)
    self._bias_value = torch.zeros(env.num_envs, self._dim, device=env.device)
    self.reset()

  def __call__(self, env, **_) -> torch.Tensor:
    value = self._call_source()
    if self._buffer.shape[1] > 1:
      self._buffer[:, 1:, :] = self._buffer[:, :-1, :].clone()
    self._buffer[:, 0, :] = value
    env_ids = torch.arange(env.num_envs, device=env.device)
    delayed = self._buffer[env_ids, self._delay_steps]
    if self._hold_prob > 0.0:
      hold_mask = torch.rand((env.num_envs, 1), device=env.device) < self._hold_prob
      delayed = torch.where(hold_mask, self._held_value, delayed)
    self._held_value[:] = delayed
    out = delayed + self._bias_value
    if self._noise > 0.0:
      out = out + torch.empty_like(out).uniform_(-self._noise, self._noise)
    return out

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    if hasattr(self._source, "reset"):
      self._source.reset(env_ids)
    reset_ids = (
      torch.arange(self._env.num_envs, device=self._env.device)[env_ids]
      if isinstance(env_ids, slice)
      else env_ids
    )
    count = len(reset_ids)
    lo, hi = self._delay_steps_range
    self._delay_steps[env_ids] = torch.randint(
      low=lo,
      high=hi + 1,
      size=(count,),
      device=self._env.device,
    )
    if self._bias > 0.0:
      self._bias_value[env_ids] = torch.empty(
        count,
        self._dim,
        device=self._env.device,
      ).uniform_(-self._bias, self._bias)
    else:
      self._bias_value[env_ids] = 0.0
    value = self._call_source()[env_ids]
    self._buffer[env_ids, :, :] = value[:, None, :]
    self._held_value[env_ids] = value


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
