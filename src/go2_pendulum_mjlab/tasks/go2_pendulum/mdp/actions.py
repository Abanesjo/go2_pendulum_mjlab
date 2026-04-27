"""Custom action terms for exact Isaac/deploy action semantics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

from go2_pendulum_mjlab.tasks.go2_pendulum.constants import (
  ACTION_SCALE,
  DEFAULT_LEG_JOINT_POS,
  LEG_JOINT_NAMES,
)


@dataclass(kw_only=True)
class OrderedGo2PdActionCfg(ActionTermCfg):
  """Raw 12D policy action -> ordered PD effort through XML motors."""

  joint_names: tuple[str, ...] = LEG_JOINT_NAMES
  default_joint_pos: tuple[float, ...] = DEFAULT_LEG_JOINT_POS
  action_scale: float = ACTION_SCALE
  stiffness: float = 25.0
  damping: float = 0.6
  effort_limit: float = 23.5
  clip_actions: bool = True
  action_clip: float = 1.0
  enable_target_filter: bool = True
  target_lpf_tau_s: float = 0.06
  max_target_rate: float = 2.5
  latency_steps_range: tuple[int, int] = (0, 2)
  command_hold_prob: float = 0.02
  randomize_effort_limit: bool = True
  motor_strength: float = 1.0
  torque_response_tau_s: float = 0.0

  def build(self, env) -> "OrderedGo2PdAction":
    return OrderedGo2PdAction(self, env)


class OrderedGo2PdAction(ActionTerm):
  """Applies deployment-compatible PD torques while preserving policy order."""

  cfg: OrderedGo2PdActionCfg

  def __init__(self, cfg: OrderedGo2PdActionCfg, env):
    super().__init__(cfg=cfg, env=env)
    self._env = env
    joint_ids, joint_names = self._entity.find_joints(cfg.joint_names, preserve_order=True)
    if tuple(joint_names) != tuple(cfg.joint_names):
      raise RuntimeError(f"Resolved joint order mismatch: {joint_names}")
    self._joint_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.long)
    self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
    self._target_pos = torch.tensor(cfg.default_joint_pos, device=self.device).repeat(self.num_envs, 1)
    self._default_pos = self._target_pos.clone()
    self._filter_target_pos = self._target_pos.clone()
    self._applied_action = torch.zeros_like(self._raw_actions)
    self._prev_applied_action = torch.zeros_like(self._raw_actions)
    self._prev_prev_applied_action = torch.zeros_like(self._raw_actions)
    self._prev_target_pos = self._target_pos.clone()
    self._prev_prev_target_pos = self._target_pos.clone()
    lo, hi = cfg.latency_steps_range
    if lo < 0 or hi < lo:
      raise ValueError(f"Invalid latency_steps_range={cfg.latency_steps_range}")
    self._target_delay_buffer = torch.zeros(
      self.num_envs,
      hi + 1,
      self.action_dim,
      device=self.device,
    )
    self._target_delay_buffer[:] = self._target_pos[:, None, :]
    self._latency_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self._default_stiffness = torch.full(
      (self.num_envs, self.action_dim), cfg.stiffness, device=self.device
    )
    self._default_damping = torch.full(
      (self.num_envs, self.action_dim), cfg.damping, device=self.device
    )
    self._default_effort_limit = torch.full(
      (self.num_envs, self.action_dim), cfg.effort_limit, device=self.device
    )
    self._default_motor_strength = torch.full(
      (self.num_envs, self.action_dim), cfg.motor_strength, device=self.device
    )
    self._default_torque_response_tau_s = torch.full(
      (self.num_envs, self.action_dim), cfg.torque_response_tau_s, device=self.device
    )
    self.stiffness = self._default_stiffness.clone()
    self.damping = self._default_damping.clone()
    self.effort_limit = self._default_effort_limit.clone()
    self.motor_strength = self._default_motor_strength.clone()
    self.torque_response_tau_s = self._default_torque_response_tau_s.clone()
    self._applied_torque = torch.zeros_like(self._raw_actions)
    self._prev_applied_torque = torch.zeros_like(self._raw_actions)

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
  def prev_applied_action(self) -> torch.Tensor:
    return self._prev_applied_action

  @property
  def prev_prev_applied_action(self) -> torch.Tensor:
    return self._prev_prev_applied_action

  @property
  def joint_ids(self) -> torch.Tensor:
    return self._joint_ids

  @property
  def target_pos(self) -> torch.Tensor:
    return self._target_pos

  @property
  def prev_target_pos(self) -> torch.Tensor:
    return self._prev_target_pos

  @property
  def prev_prev_target_pos(self) -> torch.Tensor:
    return self._prev_prev_target_pos

  @property
  def default_stiffness(self) -> torch.Tensor:
    return self._default_stiffness

  @property
  def default_damping(self) -> torch.Tensor:
    return self._default_damping

  @property
  def effort_limit_tensor(self) -> torch.Tensor:
    return self.effort_limit

  @property
  def default_effort_limit(self) -> torch.Tensor:
    return self._default_effort_limit

  @property
  def default_motor_strength(self) -> torch.Tensor:
    return self._default_motor_strength

  @property
  def default_torque_response_tau_s(self) -> torch.Tensor:
    return self._default_torque_response_tau_s

  @property
  def applied_torque(self) -> torch.Tensor:
    return self._applied_torque

  @property
  def prev_applied_torque(self) -> torch.Tensor:
    return self._prev_applied_torque

  @property
  def latency_steps(self) -> torch.Tensor:
    return self._latency_steps

  def process_actions(self, actions: torch.Tensor) -> None:
    self._prev_prev_applied_action[:] = self._prev_applied_action
    self._prev_applied_action[:] = self._applied_action
    self._prev_prev_target_pos[:] = self._prev_target_pos
    self._prev_target_pos[:] = self._target_pos

    if self.cfg.clip_actions:
      self._raw_actions[:] = torch.clamp(actions, -self.cfg.action_clip, self.cfg.action_clip)
    else:
      self._raw_actions[:] = actions

    q_raw = self._default_pos + self.cfg.action_scale * self._raw_actions
    if self.cfg.enable_target_filter:
      tau = self.cfg.target_lpf_tau_s
      alpha = 1.0 if tau <= 0.0 else 1.0 - math.exp(-float(self._env.step_dt) / tau)
      q_lpf = self._filter_target_pos + alpha * (q_raw - self._filter_target_pos)
    else:
      q_lpf = q_raw

    max_delta = self.cfg.max_target_rate * float(self._env.step_dt)
    if max_delta > 0.0:
      q_limited = self._filter_target_pos + torch.clamp(
        q_lpf - self._filter_target_pos,
        min=-max_delta,
        max=max_delta,
      )
    else:
      q_limited = q_lpf
    self._filter_target_pos[:] = q_limited

    if self._target_delay_buffer.shape[1] > 1:
      self._target_delay_buffer[:, 1:, :] = self._target_delay_buffer[:, :-1, :].clone()
    self._target_delay_buffer[:, 0, :] = q_limited
    env_ids = torch.arange(self.num_envs, device=self.device)
    q_delayed = self._target_delay_buffer[env_ids, self._latency_steps]

    hold_prob = self.cfg.command_hold_prob
    if hold_prob > 0.0:
      hold_mask = torch.rand((self.num_envs, 1), device=self.device) < hold_prob
      q_target = torch.where(hold_mask, self._target_pos, q_delayed)
    else:
      q_target = q_delayed
    if max_delta > 0.0:
      q_target = self._target_pos + torch.clamp(
        q_target - self._target_pos,
        min=-max_delta,
        max=max_delta,
      )

    self._target_pos[:] = q_target
    self._applied_action[:] = (self._target_pos - self._default_pos) / self.cfg.action_scale

  def apply_actions(self) -> None:
    q = self._entity.data.joint_pos[:, self._joint_ids]
    dq = self._entity.data.joint_vel[:, self._joint_ids]
    desired_torque = self.motor_strength * (
      self.stiffness * (self._target_pos - q) - self.damping * dq
    )
    tau = self.torque_response_tau_s
    alpha = torch.where(
      tau <= 0.0,
      torch.ones_like(tau),
      1.0 - torch.exp(-float(self._env.step_dt) / tau),
    )
    torque = self._applied_torque + alpha * (desired_torque - self._applied_torque)
    torque = torch.minimum(torch.maximum(torque, -self.effort_limit), self.effort_limit)
    self._prev_applied_torque[:] = self._applied_torque
    self._applied_torque[:] = torque
    self._entity.set_joint_effort_target(torque, joint_ids=self._joint_ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    reset_ids = (
      torch.arange(self.num_envs, device=self.device)[env_ids]
      if isinstance(env_ids, slice)
      else env_ids
    )
    count = len(reset_ids)
    q_init = self._default_pos[env_ids].clone()
    try:
      q = self._entity.data.joint_pos[:, self._joint_ids][env_ids]
      q_init = torch.where(torch.isfinite(q), q, q_init)
    except (AttributeError, IndexError, RuntimeError):
      pass
    lo, hi = self.cfg.latency_steps_range
    self._latency_steps[env_ids] = torch.randint(
      low=lo,
      high=hi + 1,
      size=(count,),
      device=self.device,
    )
    self._raw_actions[env_ids] = 0.0
    self._filter_target_pos[env_ids] = q_init
    self._target_pos[env_ids] = q_init
    self._target_delay_buffer[env_ids, :, :] = q_init[:, None, :]
    self._applied_action[env_ids] = (q_init - self._default_pos[env_ids]) / self.cfg.action_scale
    self._prev_applied_action[env_ids] = self._applied_action[env_ids]
    self._prev_prev_applied_action[env_ids] = self._applied_action[env_ids]
    self._prev_target_pos[env_ids] = q_init
    self._prev_prev_target_pos[env_ids] = q_init
    self._applied_torque[env_ids] = 0.0
    self._prev_applied_torque[env_ids] = 0.0
