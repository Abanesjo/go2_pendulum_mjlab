"""Unitree MuJoCo realism helpers for the Go2 pendulum task."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import mujoco
import mujoco_warp as mjwarp
import torch

from mjlab.entity import Entity
from mjlab.sensor import Sensor, SensorCfg
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_from_euler_xyz, quat_mul

from go2_pendulum_mjlab.tasks.go2_pendulum.constants import DEFAULT_LEG_JOINT_POS, LEG_JOINT_NAMES

if TYPE_CHECKING:
  from mjlab.viewer.debug_visualizer import DebugVisualizer


@dataclass(frozen=True)
class UnitreeLowStateRealismCfg:
  publish_delay_ms: float = 8.0
  publish_jitter_ms: float = 2.0
  publish_dropout_prob: float = 0.001
  joint_pos_noise_std_rad: float = 0.0015
  joint_pos_bias_std_rad: float = 0.003
  joint_vel_noise_std_rad_s: float = 0.06
  joint_vel_bias_std_rad_s: float = 0.02
  joint_vel_lpf_tau_s: float = 0.015
  imu_gyro_noise_std_rad_s: float = 0.015
  imu_quat_noise_std_rad: float = 0.003


@dataclass(frozen=True)
class UnitreePoseRealismCfg:
  publish_delay_ms: float = 10.0
  publish_jitter_ms: float = 2.0
  publish_dropout_prob: float = 0.001
  position_noise_std_m: float = 0.002
  position_bias_std_m: float = 0.001
  orientation_noise_std_rad: float = 0.003


@dataclass(frozen=True)
class UnitreePendulumEePoseRealismCfg:
  publish_delay_ms: float = 15.0
  publish_jitter_ms: float = 3.0
  publish_dropout_prob: float = 0.0
  position_noise_std_m: float = 0.004
  position_bias_std_m: float = 0.002
  orientation_noise_std_rad: float = 0.004


@dataclass(frozen=True)
class UnitreeActuatorRealismCfg:
  command_delay_ms: float = 6.0
  command_jitter_ms: float = 2.0
  command_dropout_prob: float = 0.001
  command_hold_prob: float = 0.002
  torque_lpf_tau_s: float = 0.010
  torque_rate_limit_nm_per_s: float = 600.0
  torque_limit_scale_abduction: float = 0.85
  torque_limit_scale_hip: float = 0.85
  torque_limit_scale_knee: float = 0.75
  motor_strength_std: float = 0.04
  kp_scale_mean: float = 1.0
  kp_scale_std: float = 0.08
  kd_scale_mean: float = 1.0
  kd_scale_std: float = 0.12
  speed_limit_rad_s_abduction: float = 28.0
  speed_limit_rad_s_hip: float = 28.0
  speed_limit_rad_s_knee: float = 35.0
  min_speed_torque_scale: float = 0.35
  torque_deadband_nm: float = 0.15
  torque_noise_std_nm: float = 0.03


@dataclass(frozen=True)
class UnitreeContactRealismCfg:
  foot_friction: tuple[float, float, float] = (0.55, 0.025, 0.010)


@dataclass(frozen=True)
class UnitreePendulumEstimatorCfg:
  window: int = 7
  poly_order: int = 2
  delta_s: float = 0.005
  hinge_offset: tuple[float, float, float] = (-0.05, 0.0, 0.06)


@dataclass(frozen=True)
class UnitreeRealismCfg:
  lowstate: UnitreeLowStateRealismCfg = field(default_factory=UnitreeLowStateRealismCfg)
  base_pose: UnitreePoseRealismCfg = field(default_factory=UnitreePoseRealismCfg)
  pendulum_ee_pose: UnitreePendulumEePoseRealismCfg = field(default_factory=UnitreePendulumEePoseRealismCfg)
  actuator: UnitreeActuatorRealismCfg = field(default_factory=UnitreeActuatorRealismCfg)
  contact: UnitreeContactRealismCfg = field(default_factory=UnitreeContactRealismCfg)
  pendulum_estimator: UnitreePendulumEstimatorCfg = field(default_factory=UnitreePendulumEstimatorCfg)


UNITREE_REALISM = UnitreeRealismCfg()


def lpf_alpha(dt: float, tau: float) -> float:
  """Return the first-order low-pass update alpha for a timestep and time constant."""
  if tau <= 0.0:
    return 1.0
  return float(1.0 - math.exp(-float(dt) / float(tau)))


def delay_lag_range(delay_ms: float, jitter_ms: float, dt: float) -> tuple[int, int]:
  """Convert a delay and symmetric jitter into integer physics-step lags."""
  lo_s = max(0.0, (float(delay_ms) - float(jitter_ms)) * 1.0e-3)
  hi_s = max(0.0, (float(delay_ms) + float(jitter_ms)) * 1.0e-3)
  lo = int(round(lo_s / dt))
  hi = int(round(hi_s / dt))
  return max(0, lo), max(max(0, lo), hi)


def savgol_coeffs(window: int, poly_order: int, deriv: int, delta: float, pos: int | None = None) -> torch.Tensor:
  """Compute causal Savitzky-Golay coefficients matching scipy's ``use='dot'``."""
  if window <= 0 or window % 2 == 0:
    raise ValueError(f"window must be a positive odd integer, got {window}.")
  if poly_order >= window:
    raise ValueError(f"poly_order must be < window, got {poly_order} >= {window}.")
  if deriv > poly_order:
    return torch.zeros(window, dtype=torch.float32)
  if pos is None:
    pos = window // 2
  if not 0 <= pos < window:
    raise ValueError(f"pos must be in [0, {window}), got {pos}.")
  if delta <= 0.0:
    raise ValueError(f"delta must be positive, got {delta}.")

  x = torch.arange(window, dtype=torch.float64) - float(pos)
  order = torch.arange(poly_order + 1, dtype=torch.float64)
  design = x[:, None] ** order[None, :]
  pinv = torch.linalg.pinv(design)
  coeff = pinv[deriv] * (math.factorial(deriv) / (float(delta) ** deriv))
  return coeff.to(dtype=torch.float32)


def normalize_quat(quat: torch.Tensor) -> torch.Tensor:
  return quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(1.0e-9)


def perturb_quat(quat: torch.Tensor, std_rad: float) -> torch.Tensor:
  if std_rad <= 0.0:
    return quat
  noise = quat_from_euler_xyz(
    torch.randn(quat.shape[0], device=quat.device) * std_rad,
    torch.randn(quat.shape[0], device=quat.device) * std_rad,
    torch.randn(quat.shape[0], device=quat.device) * std_rad,
  )
  return normalize_quat(quat_mul(noise, quat))


def pendulum_angles_from_base_ee(
  base_pos_w: torch.Tensor,
  base_quat_w: torch.Tensor,
  ee_pos_w: torch.Tensor,
  hinge_offset: torch.Tensor | tuple[float, float, float] = UNITREE_REALISM.pendulum_estimator.hinge_offset,
) -> torch.Tensor:
  """Estimate pendulum hinge angles from base and end-effector poses."""
  offset = torch.as_tensor(hinge_offset, dtype=base_pos_w.dtype, device=base_pos_w.device)
  v = quat_apply_inverse(base_quat_w, ee_pos_w - base_pos_w) - offset
  joint1 = torch.atan2(-v[:, 1], v[:, 2])
  joint2 = torch.atan2(v[:, 0], torch.hypot(v[:, 1], v[:, 2]))
  return torch.stack((joint1, joint2), dim=-1)


class LaggedSignal:
  """Small per-env delayed stream with optional dropout-as-hold behavior."""

  def __init__(self, num_envs: int, dim: int, max_lag: int, device: str):
    self.history = torch.zeros(num_envs, max_lag + 1, dim, device=device)
    self.output = torch.zeros(num_envs, dim, device=device)
    self.last_input = torch.zeros(num_envs, dim, device=device)
    self.current_lags = torch.zeros(num_envs, dtype=torch.long, device=device)
    self._env_ids = torch.arange(num_envs, dtype=torch.long, device=device)

  def reset(self, values: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
    if env_ids is None:
      self.history[:] = values[:, None, :]
      self.output[:] = values
      self.last_input[:] = values
      self.current_lags[:] = 0
      return
    self.history[env_ids] = values[:, None, :]
    self.output[env_ids] = values
    self.last_input[env_ids] = values
    self.current_lags[env_ids] = 0

  def update(self, values: torch.Tensor, min_lag: int, max_lag: int, dropout_prob: float = 0.0) -> torch.Tensor:
    if dropout_prob > 0.0:
      dropout = torch.rand(values.shape[0], device=values.device) < dropout_prob
      values = torch.where(dropout[:, None], self.last_input, values)
    self.last_input[:] = values

    self.history[:, 1:] = self.history[:, :-1].clone()
    self.history[:, 0] = values
    if max_lag <= min_lag:
      self.current_lags[:] = min_lag
    else:
      self.current_lags[:] = torch.randint(
        min_lag,
        max_lag + 1,
        (values.shape[0],),
        dtype=torch.long,
        device=values.device,
      )
    self.output[:] = self.history[self._env_ids, self.current_lags]
    return self.output


class CausalSavitzkyGolayFilter:
  """Causal latest-sample Savitzky-Golay smoother and derivative."""

  def __init__(self, num_envs: int, dim: int, window: int, poly_order: int, delta: float, device: str):
    self.window = window
    self.history = torch.zeros(num_envs, window, dim, device=device)
    self.position = torch.zeros(num_envs, dim, device=device)
    self.velocity = torch.zeros(num_envs, dim, device=device)
    self._smooth_coeffs = savgol_coeffs(window, poly_order, deriv=0, delta=delta, pos=window - 1).to(device=device)
    self._vel_coeffs = savgol_coeffs(window, poly_order, deriv=1, delta=delta, pos=window - 1).to(device=device)

  def reset(self, values: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
    if env_ids is None:
      self.history[:] = values[:, None, :]
      self.position[:] = values
      self.velocity[:] = 0.0
      return
    self.history[env_ids] = values[:, None, :]
    self.position[env_ids] = values
    self.velocity[env_ids] = 0.0

  def update(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    self.history[:, :-1] = self.history[:, 1:].clone()
    self.history[:, -1] = values
    self.position[:] = torch.einsum("w,bwd->bd", self._smooth_coeffs, self.history)
    self.velocity[:] = torch.einsum("w,bwd->bd", self._vel_coeffs, self.history)
    return self.position, self.velocity


@dataclass
class UnitreeRealismSensorData:
  leg_joint_pos_rel: torch.Tensor
  leg_joint_vel: torch.Tensor
  imu_ang_vel_b: torch.Tensor
  projected_gravity_b: torch.Tensor
  base_pos_w: torch.Tensor
  base_quat_w: torch.Tensor
  pendulum_pos: torch.Tensor
  pendulum_vel: torch.Tensor
  clean_pendulum_pos: torch.Tensor
  clean_pendulum_vel: torch.Tensor


@dataclass
class UnitreeRealismSensorCfg(SensorCfg):
  asset_name: str = "robot"
  leg_joint_names: tuple[str, ...] = LEG_JOINT_NAMES
  pendulum_ee_body_name: str = "pendulum_ee"
  realism: UnitreeRealismCfg = UNITREE_REALISM

  def build(self) -> "UnitreeRealismSensor":
    return UnitreeRealismSensor(self)


class UnitreeRealismSensor(Sensor[UnitreeRealismSensorData]):
  """Internal high-rate state stream matching the Unitree MuJoCo deployment path."""

  def __init__(self, cfg: UnitreeRealismSensorCfg):
    super().__init__()
    self.cfg = cfg
    self._asset: Entity | None = None

  def edit_spec(self, scene_spec: mujoco.MjSpec, entities: dict[str, Entity]) -> None:
    del scene_spec
    self._asset = entities[self.cfg.asset_name]

  def initialize(
    self,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
    device: str,
  ) -> None:
    del mj_model, model, data
    if self._asset is None:
      raise RuntimeError("UnitreeRealismSensor must be attached to an entity before initialize().")
    joint_ids, joint_names = self._asset.find_joints(self.cfg.leg_joint_names, preserve_order=True)
    if tuple(joint_names) != tuple(self.cfg.leg_joint_names):
      raise RuntimeError(f"Resolved joint order mismatch: {joint_names}")
    body_ids, body_names = self._asset.find_bodies((self.cfg.pendulum_ee_body_name,), preserve_order=True)
    if tuple(body_names) != (self.cfg.pendulum_ee_body_name,):
      raise RuntimeError(f"Could not resolve pendulum EE body '{self.cfg.pendulum_ee_body_name}'.")

    num_envs = self._asset.data.joint_pos.shape[0]
    physics_dt = self.cfg.realism.pendulum_estimator.delta_s
    low_min, low_max = delay_lag_range(
      self.cfg.realism.lowstate.publish_delay_ms,
      self.cfg.realism.lowstate.publish_jitter_ms,
      physics_dt,
    )
    base_min, base_max = delay_lag_range(
      self.cfg.realism.base_pose.publish_delay_ms,
      self.cfg.realism.base_pose.publish_jitter_ms,
      physics_dt,
    )
    ee_min, ee_max = delay_lag_range(
      self.cfg.realism.pendulum_ee_pose.publish_delay_ms,
      self.cfg.realism.pendulum_ee_pose.publish_jitter_ms,
      physics_dt,
    )
    self._low_lag_range = (low_min, low_max)
    self._base_lag_range = (base_min, base_max)
    self._ee_lag_range = (ee_min, ee_max)

    self._joint_ids = torch.tensor(joint_ids, device=device, dtype=torch.long)
    self._pendulum_ee_body_id = int(body_ids[0])
    self._default_joint_pos = torch.tensor(DEFAULT_LEG_JOINT_POS, device=device).repeat(num_envs, 1)
    self._gravity_w = torch.tensor((0.0, 0.0, -1.0), device=device).repeat(num_envs, 1)
    self._hinge_offset = torch.tensor(self.cfg.realism.pendulum_estimator.hinge_offset, device=device)
    self._needs_reset = torch.ones(num_envs, device=device, dtype=torch.bool)

    self._joint_pos_bias = torch.zeros(num_envs, len(joint_ids), device=device)
    self._joint_vel_bias = torch.zeros_like(self._joint_pos_bias)
    self._joint_vel_lpf = torch.zeros_like(self._joint_pos_bias)
    self._base_pos_bias = torch.zeros(num_envs, 3, device=device)
    self._ee_pos_bias = torch.zeros(num_envs, 3, device=device)

    self._lowstate_stream = LaggedSignal(num_envs, len(joint_ids) * 2 + 7, low_max, device)
    self._base_pose_stream = LaggedSignal(num_envs, 7, base_max, device)
    self._ee_pos_stream = LaggedSignal(num_envs, 3, ee_max, device)
    est = self.cfg.realism.pendulum_estimator
    self._noisy_pendulum = CausalSavitzkyGolayFilter(num_envs, 2, est.window, est.poly_order, est.delta_s, device)
    self._clean_pendulum = CausalSavitzkyGolayFilter(num_envs, 2, est.window, est.poly_order, est.delta_s, device)

    self._data = UnitreeRealismSensorData(
      leg_joint_pos_rel=torch.zeros(num_envs, len(joint_ids), device=device),
      leg_joint_vel=torch.zeros(num_envs, len(joint_ids), device=device),
      imu_ang_vel_b=torch.zeros(num_envs, 3, device=device),
      projected_gravity_b=torch.zeros(num_envs, 3, device=device),
      base_pos_w=torch.zeros(num_envs, 3, device=device),
      base_quat_w=torch.zeros(num_envs, 4, device=device),
      pendulum_pos=torch.zeros(num_envs, 2, device=device),
      pendulum_vel=torch.zeros(num_envs, 2, device=device),
      clean_pendulum_pos=torch.zeros(num_envs, 2, device=device),
      clean_pendulum_vel=torch.zeros(num_envs, 2, device=device),
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      self._needs_reset[:] = True
    elif isinstance(env_ids, slice):
      self._needs_reset[env_ids] = True
    else:
      self._needs_reset[env_ids.to(device=self._needs_reset.device, dtype=torch.long)] = True

  def update(self, dt: float) -> None:
    super().update(dt)
    self._maybe_reset_buffers()
    self._advance(dt)

  def _compute_data(self) -> UnitreeRealismSensorData:
    self._maybe_reset_buffers()
    return self._data

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    del visualizer

  @property
  def _robot(self) -> Entity:
    assert self._asset is not None
    return self._asset

  def _maybe_reset_buffers(self) -> None:
    env_ids = torch.nonzero(self._needs_reset, as_tuple=False).squeeze(-1)
    if env_ids.numel() == 0:
      return
    self._sample_biases(env_ids)
    clean_q = self._robot.data.joint_pos[:, self._joint_ids]
    clean_dq = self._robot.data.joint_vel[:, self._joint_ids]
    self._joint_vel_lpf[env_ids] = clean_dq[env_ids]

    lowstate = self._sample_lowstate()
    base_pose = self._sample_base_pose()
    ee_pos = self._sample_ee_pos()
    self._lowstate_stream.reset(lowstate[env_ids], env_ids)
    self._base_pose_stream.reset(base_pose[env_ids], env_ids)
    self._ee_pos_stream.reset(ee_pos[env_ids], env_ids)

    clean_angles = self._clean_pendulum_angles()
    noisy_angles = pendulum_angles_from_base_ee(
      self._base_pose_stream.output[:, :3],
      normalize_quat(self._base_pose_stream.output[:, 3:7]),
      self._ee_pos_stream.output,
      self._hinge_offset,
    )
    self._clean_pendulum.reset(clean_angles[env_ids], env_ids)
    self._noisy_pendulum.reset(noisy_angles[env_ids], env_ids)
    self._sync_data_from_streams()
    self._needs_reset[env_ids] = False

  def _advance(self, dt: float) -> None:
    low = self.cfg.realism.lowstate
    alpha = lpf_alpha(dt, low.joint_vel_lpf_tau_s)
    clean_dq = self._robot.data.joint_vel[:, self._joint_ids]
    self._joint_vel_lpf += alpha * (clean_dq - self._joint_vel_lpf)

    self._lowstate_stream.update(
      self._sample_lowstate(),
      *self._low_lag_range,
      dropout_prob=low.publish_dropout_prob,
    )
    self._base_pose_stream.update(
      self._sample_base_pose(),
      *self._base_lag_range,
      dropout_prob=self.cfg.realism.base_pose.publish_dropout_prob,
    )
    self._ee_pos_stream.update(
      self._sample_ee_pos(),
      *self._ee_lag_range,
      dropout_prob=self.cfg.realism.pendulum_ee_pose.publish_dropout_prob,
    )

    self._clean_pendulum.update(self._clean_pendulum_angles())
    noisy_angles = pendulum_angles_from_base_ee(
      self._base_pose_stream.output[:, :3],
      normalize_quat(self._base_pose_stream.output[:, 3:7]),
      self._ee_pos_stream.output,
      self._hinge_offset,
    )
    self._noisy_pendulum.update(noisy_angles)
    self._sync_data_from_streams()

  def _sample_biases(self, env_ids: torch.Tensor) -> None:
    low = self.cfg.realism.lowstate
    base = self.cfg.realism.base_pose
    ee = self.cfg.realism.pendulum_ee_pose
    self._joint_pos_bias[env_ids] = torch.randn_like(self._joint_pos_bias[env_ids]) * low.joint_pos_bias_std_rad
    self._joint_vel_bias[env_ids] = torch.randn_like(self._joint_vel_bias[env_ids]) * low.joint_vel_bias_std_rad_s
    self._base_pos_bias[env_ids] = torch.randn_like(self._base_pos_bias[env_ids]) * base.position_bias_std_m
    self._ee_pos_bias[env_ids] = torch.randn_like(self._ee_pos_bias[env_ids]) * ee.position_bias_std_m

  def _sample_lowstate(self) -> torch.Tensor:
    low = self.cfg.realism.lowstate
    q = self._robot.data.joint_pos[:, self._joint_ids]
    dq = self._joint_vel_lpf
    gyro = self._robot.data.root_link_ang_vel_b
    quat = self._robot.data.root_link_quat_w
    q_noisy = q + self._joint_pos_bias + torch.randn_like(q) * low.joint_pos_noise_std_rad
    dq_noisy = dq + self._joint_vel_bias + torch.randn_like(dq) * low.joint_vel_noise_std_rad_s
    gyro_noisy = gyro + torch.randn_like(gyro) * low.imu_gyro_noise_std_rad_s
    quat_noisy = perturb_quat(quat, low.imu_quat_noise_std_rad)
    return torch.cat((q_noisy, dq_noisy, gyro_noisy, quat_noisy), dim=-1)

  def _sample_base_pose(self) -> torch.Tensor:
    cfg = self.cfg.realism.base_pose
    pos = self._robot.data.root_link_pos_w + self._base_pos_bias
    pos = pos + torch.randn_like(pos) * cfg.position_noise_std_m
    quat = perturb_quat(self._robot.data.root_link_quat_w, cfg.orientation_noise_std_rad)
    return torch.cat((pos, quat), dim=-1)

  def _sample_ee_pos(self) -> torch.Tensor:
    cfg = self.cfg.realism.pendulum_ee_pose
    ee_pos = self._robot.data.body_link_pos_w[:, self._pendulum_ee_body_id]
    return ee_pos + self._ee_pos_bias + torch.randn_like(ee_pos) * cfg.position_noise_std_m

  def _clean_pendulum_angles(self) -> torch.Tensor:
    ee_pos = self._robot.data.body_link_pos_w[:, self._pendulum_ee_body_id]
    return pendulum_angles_from_base_ee(
      self._robot.data.root_link_pos_w,
      self._robot.data.root_link_quat_w,
      ee_pos,
      self._hinge_offset,
    )

  def _sync_data_from_streams(self) -> None:
    low = self._lowstate_stream.output
    q = low[:, : len(self._joint_ids)]
    dq = low[:, len(self._joint_ids) : 2 * len(self._joint_ids)]
    gyro = low[:, 2 * len(self._joint_ids) : 2 * len(self._joint_ids) + 3]
    quat = normalize_quat(low[:, -4:])
    base_pose = self._base_pose_stream.output
    base_quat = normalize_quat(base_pose[:, 3:7])

    self._data.leg_joint_pos_rel[:] = q - self._default_joint_pos
    self._data.leg_joint_vel[:] = dq
    self._data.imu_ang_vel_b[:] = gyro
    self._data.projected_gravity_b[:] = quat_apply_inverse(quat, self._gravity_w)
    self._data.base_pos_w[:] = base_pose[:, :3]
    self._data.base_quat_w[:] = base_quat
    self._data.pendulum_pos[:] = self._noisy_pendulum.position
    self._data.pendulum_vel[:] = self._noisy_pendulum.velocity
    self._data.clean_pendulum_pos[:] = self._clean_pendulum.position
    self._data.clean_pendulum_vel[:] = self._clean_pendulum.velocity
