"""Custom action terms for exact Isaac/deploy action semantics."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

from go2_pendulum_mjlab.tasks.go2_pendulum.constants import (
  ACTION_SCALE,
  DEFAULT_LEG_JOINT_POS,
  LEG_JOINT_NAMES,
)
from go2_pendulum_mjlab.tasks.go2_pendulum.mdp.realism import (
  UNITREE_REALISM,
  UnitreeActuatorRealismCfg,
  delay_lag_range,
  lpf_alpha,
)

GO2_ABDUCTION_TORQUE_LIMIT_NM = 23.7
GO2_HIP_TORQUE_LIMIT_NM = 23.7
GO2_KNEE_TORQUE_LIMIT_NM = 45.43


def go2_joint_class_values(
  joint_names: tuple[str, ...],
  abduction: float,
  hip: float,
  knee: float,
) -> tuple[float, ...]:
  values = []
  for name in joint_names:
    if name.endswith("_hip_joint"):
      values.append(abduction)
    elif name.endswith("_thigh_joint"):
      values.append(hip)
    elif name.endswith("_calf_joint"):
      values.append(knee)
    else:
      raise ValueError(f"Unsupported Go2 leg joint name: {name}")
  return tuple(values)


def go2_torque_limits(
  joint_names: tuple[str, ...],
  realism: UnitreeActuatorRealismCfg = UNITREE_REALISM.actuator,
) -> tuple[float, ...]:
  return go2_joint_class_values(
    joint_names,
    GO2_ABDUCTION_TORQUE_LIMIT_NM * realism.torque_limit_scale_abduction,
    GO2_HIP_TORQUE_LIMIT_NM * realism.torque_limit_scale_hip,
    GO2_KNEE_TORQUE_LIMIT_NM * realism.torque_limit_scale_knee,
  )


def go2_speed_limits(
  joint_names: tuple[str, ...],
  realism: UnitreeActuatorRealismCfg = UNITREE_REALISM.actuator,
) -> tuple[float, ...]:
  return go2_joint_class_values(
    joint_names,
    realism.speed_limit_rad_s_abduction,
    realism.speed_limit_rad_s_hip,
    realism.speed_limit_rad_s_knee,
  )


class ActionCommandDelay:
  """High-rate command delay with hold/dropout semantics."""

  def __init__(self, num_envs: int, action_dim: int, device: str):
    self.current = torch.zeros(num_envs, action_dim, device=device)
    self.last_accepted = torch.zeros_like(self.current)
    self.pending = torch.zeros_like(self.current)
    self.pending_delay_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
    self.has_pending = torch.zeros(num_envs, dtype=torch.bool, device=device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.current[env_ids] = 0.0
    self.last_accepted[env_ids] = 0.0
    self.pending[env_ids] = 0.0
    self.pending_delay_steps[env_ids] = 0
    self.has_pending[env_ids] = False

  def push(
    self,
    actions: torch.Tensor,
    delay_steps: torch.Tensor,
    hold_mask: torch.Tensor | None = None,
    dropout_mask: torch.Tensor | None = None,
  ) -> None:
    keep_previous = torch.zeros(actions.shape[0], dtype=torch.bool, device=actions.device)
    if hold_mask is not None:
      keep_previous |= hold_mask
    if dropout_mask is not None:
      keep_previous |= dropout_mask
    accepted = torch.where(keep_previous[:, None], self.last_accepted, actions)
    self.last_accepted[:] = accepted
    self.pending[:] = accepted
    self.pending_delay_steps[:] = delay_steps.clamp_min(0)
    self.has_pending[:] = True

  def step(self) -> torch.Tensor:
    ready = self.has_pending & (self.pending_delay_steps <= 0)
    self.current[:] = torch.where(ready[:, None], self.pending, self.current)
    self.has_pending[ready] = False
    waiting = self.has_pending & (self.pending_delay_steps > 0)
    self.pending_delay_steps[waiting] -= 1
    return self.current


@dataclass(kw_only=True)
class OrderedGo2PdActionCfg(ActionTermCfg):
  """Raw 12D policy action -> ordered PD effort through XML motors."""

  joint_names: tuple[str, ...] = LEG_JOINT_NAMES
  default_joint_pos: tuple[float, ...] = DEFAULT_LEG_JOINT_POS
  action_scale: float = ACTION_SCALE
  stiffness: float = 25.0
  damping: float = 0.6
  realism: UnitreeActuatorRealismCfg = UNITREE_REALISM.actuator

  def build(self, env) -> "OrderedGo2PdAction":
    return OrderedGo2PdAction(self, env)


class OrderedGo2PdAction(ActionTerm):
  """Applies deployment-compatible PD torques while preserving policy order."""

  cfg: OrderedGo2PdActionCfg

  def __init__(self, cfg: OrderedGo2PdActionCfg, env):
    super().__init__(cfg=cfg, env=env)
    joint_ids, joint_names = self._entity.find_joints(cfg.joint_names, preserve_order=True)
    if tuple(joint_names) != tuple(cfg.joint_names):
      raise RuntimeError(f"Resolved joint order mismatch: {joint_names}")
    self._min_action_delay_steps, self._max_action_delay_steps = delay_lag_range(
      cfg.realism.command_delay_ms,
      cfg.realism.command_jitter_ms,
      env.physics_dt,
    )
    self._joint_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.long)
    self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
    self._target_pos = torch.tensor(cfg.default_joint_pos, device=self.device).repeat(self.num_envs, 1)
    self._default_pos = self._target_pos.clone()
    self._applied_action = torch.zeros_like(self._raw_actions)
    self._command_delay = ActionCommandDelay(self.num_envs, self.action_dim, self.device)
    self._all_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
    self._default_stiffness = torch.full(
      (self.num_envs, self.action_dim), cfg.stiffness, device=self.device
    )
    self._default_damping = torch.full(
      (self.num_envs, self.action_dim), cfg.damping, device=self.device
    )
    self.stiffness = self._default_stiffness.clone()
    self.damping = self._default_damping.clone()
    self._motor_strength = torch.ones(self.num_envs, self.action_dim, device=self.device)
    self._torque_limit = torch.tensor(go2_torque_limits(cfg.joint_names, cfg.realism), device=self.device)
    self._speed_limit = torch.tensor(go2_speed_limits(cfg.joint_names, cfg.realism), device=self.device)
    self._prev_torque = torch.zeros(self.num_envs, self.action_dim, device=self.device)
    self._sample_actuator_params(self._all_env_ids)

  @property
  def action_dim(self) -> int:
    return len(self.cfg.joint_names)

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  @property
  def applied_action(self) -> torch.Tensor:
    return self._applied_action

  @property
  def action_delay_steps(self) -> torch.Tensor:
    return self._command_delay.pending_delay_steps

  @property
  def joint_ids(self) -> torch.Tensor:
    return self._joint_ids

  @property
  def target_pos(self) -> torch.Tensor:
    return self._target_pos

  @property
  def default_stiffness(self) -> torch.Tensor:
    return self._default_stiffness

  @property
  def default_damping(self) -> torch.Tensor:
    return self._default_damping

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_actions[:] = actions
    delay_steps = self._sample_command_delay_steps()
    hold_mask = torch.rand(self.num_envs, device=self.device) < self.cfg.realism.command_hold_prob
    dropout_mask = torch.rand(self.num_envs, device=self.device) < self.cfg.realism.command_dropout_prob
    self._command_delay.push(actions, delay_steps, hold_mask=hold_mask, dropout_mask=dropout_mask)

  def apply_actions(self) -> None:
    self._applied_action[:] = self._command_delay.step()
    self._target_pos[:] = self._default_pos + self.cfg.action_scale * self._applied_action
    q = self._entity.data.joint_pos[:, self._joint_ids]
    dq = self._entity.data.joint_vel[:, self._joint_ids]
    torque = self.stiffness * (self._target_pos - q) - self.damping * dq
    torque = self._apply_torque_realism(torque, dq)
    self._entity.set_joint_effort_target(torque, joint_ids=self._joint_ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    env_ids = self._resolve_env_ids(env_ids)
    self._raw_actions[env_ids] = 0.0
    self._applied_action[env_ids] = 0.0
    self._command_delay.reset(env_ids)
    self._target_pos[env_ids] = self._default_pos[env_ids]
    self._prev_torque[env_ids] = 0.0
    self._sample_actuator_params(env_ids)

  def _resolve_env_ids(self, env_ids: torch.Tensor | slice | None) -> torch.Tensor:
    if env_ids is None:
      return self._all_env_ids
    if isinstance(env_ids, slice):
      return self._all_env_ids[env_ids]
    return env_ids.to(device=self.device, dtype=torch.long)

  def _sample_command_delay_steps(self) -> torch.Tensor:
    if self._min_action_delay_steps == self._max_action_delay_steps:
      return torch.full((self.num_envs,), self._min_action_delay_steps, dtype=torch.long, device=self.device)
    return torch.randint(
      self._min_action_delay_steps,
      self._max_action_delay_steps + 1,
      (self.num_envs,),
      device=self.device,
      dtype=torch.long,
    )

  def _sample_actuator_params(self, env_ids: torch.Tensor) -> None:
    cfg = self.cfg.realism
    self._motor_strength[env_ids] = 1.0 + torch.randn(
      (env_ids.numel(), self.action_dim), device=self.device
    ) * cfg.motor_strength_std
    kp_scale = cfg.kp_scale_mean + torch.randn(
      (env_ids.numel(), self.action_dim), device=self.device
    ) * cfg.kp_scale_std
    kd_scale = cfg.kd_scale_mean + torch.randn(
      (env_ids.numel(), self.action_dim), device=self.device
    ) * cfg.kd_scale_std
    self.stiffness[env_ids] = self._default_stiffness[env_ids] * kp_scale.clamp_min(0.0)
    self.damping[env_ids] = self._default_damping[env_ids] * kd_scale.clamp_min(0.0)

  def _apply_torque_realism(self, torque: torch.Tensor, joint_vel: torch.Tensor) -> torch.Tensor:
    cfg = self.cfg.realism
    torque = torque * self._motor_strength
    speed_ratio = (torch.abs(joint_vel) / self._speed_limit).clamp(0.0, 1.0)
    speed_scale = 1.0 - (1.0 - cfg.min_speed_torque_scale) * speed_ratio
    limit = self._torque_limit * speed_scale
    torque = torch.clamp(torque, -limit, limit)
    torque = torch.where(torch.abs(torque) < cfg.torque_deadband_nm, torch.zeros_like(torque), torque)
    if cfg.torque_noise_std_nm > 0.0:
      torque = torque + torch.randn_like(torque) * cfg.torque_noise_std_nm
    torque = torch.clamp(torque, -limit, limit)

    alpha = lpf_alpha(self._env.physics_dt, cfg.torque_lpf_tau_s)
    filtered = self._prev_torque + alpha * (torque - self._prev_torque)
    max_delta = cfg.torque_rate_limit_nm_per_s * self._env.physics_dt
    filtered = self._prev_torque + (filtered - self._prev_torque).clamp(-max_delta, max_delta)
    filtered = torch.clamp(filtered, -limit, limit)
    self._prev_torque[:] = filtered
    return filtered
