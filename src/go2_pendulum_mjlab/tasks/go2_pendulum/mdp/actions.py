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


@dataclass(kw_only=True)
class OrderedGo2PdActionCfg(ActionTermCfg):
  """Raw 12D policy action -> ordered PD effort through XML motors."""

  joint_names: tuple[str, ...] = LEG_JOINT_NAMES
  default_joint_pos: tuple[float, ...] = DEFAULT_LEG_JOINT_POS
  action_scale: float = ACTION_SCALE
  stiffness: float = 25.0
  damping: float = 0.6
  effort_limit: float = 23.5
  action_delay_steps_range: tuple[int, int] = (0, 2)

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
    min_delay, max_delay = cfg.action_delay_steps_range
    if min_delay < 0 or max_delay < 0:
      raise ValueError(f"action_delay_steps_range values must be non-negative, got {cfg.action_delay_steps_range}.")
    if max_delay < min_delay:
      raise ValueError(f"action_delay_steps_range max must be >= min, got {cfg.action_delay_steps_range}.")
    self._min_action_delay_steps = int(min_delay)
    self._max_action_delay_steps = int(max_delay)
    self._joint_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.long)
    self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
    self._target_pos = torch.tensor(cfg.default_joint_pos, device=self.device).repeat(self.num_envs, 1)
    self._default_pos = self._target_pos.clone()
    self._applied_action = torch.zeros_like(self._raw_actions)
    self._action_delay_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
    self._action_delay_history = torch.zeros(
      self.num_envs,
      self.action_dim,
      self._max_action_delay_steps + 1,
      device=self.device,
    )
    self._all_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
    self._sample_action_delay(self._all_env_ids)
    self._default_stiffness = torch.full(
      (self.num_envs, self.action_dim), cfg.stiffness, device=self.device
    )
    self._default_damping = torch.full(
      (self.num_envs, self.action_dim), cfg.damping, device=self.device
    )
    self.stiffness = self._default_stiffness.clone()
    self.damping = self._default_damping.clone()

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
    return self._action_delay_steps

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
    if self._action_delay_history.shape[-1] > 1:
      self._action_delay_history[:, :, 1:] = self._action_delay_history[:, :, :-1].clone()
    self._action_delay_history[:, :, 0] = actions
    delay_ids = self._action_delay_steps.view(self.num_envs, 1, 1).expand(-1, self.action_dim, 1)
    self._applied_action[:] = torch.gather(self._action_delay_history, dim=2, index=delay_ids).squeeze(-1)
    self._target_pos[:] = self._default_pos + self.cfg.action_scale * self._applied_action

  def apply_actions(self) -> None:
    q = self._entity.data.joint_pos[:, self._joint_ids]
    dq = self._entity.data.joint_vel[:, self._joint_ids]
    torque = self.stiffness * (self._target_pos - q) - self.damping * dq
    torque = torch.clamp(torque, -self.cfg.effort_limit, self.cfg.effort_limit)
    self._entity.set_joint_effort_target(torque, joint_ids=self._joint_ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    env_ids = self._resolve_env_ids(env_ids)
    self._raw_actions[env_ids] = 0.0
    self._applied_action[env_ids] = 0.0
    self._action_delay_history[env_ids] = 0.0
    self._sample_action_delay(env_ids)
    self._target_pos[env_ids] = self._default_pos[env_ids]

  def _resolve_env_ids(self, env_ids: torch.Tensor | slice | None) -> torch.Tensor:
    if env_ids is None:
      return self._all_env_ids
    if isinstance(env_ids, slice):
      return self._all_env_ids[env_ids]
    return env_ids.to(device=self.device, dtype=torch.long)

  def _sample_action_delay(self, env_ids: torch.Tensor) -> None:
    if self._min_action_delay_steps == self._max_action_delay_steps:
      self._action_delay_steps[env_ids] = self._min_action_delay_steps
      return
    self._action_delay_steps[env_ids] = torch.randint(
      self._min_action_delay_steps,
      self._max_action_delay_steps + 1,
      (env_ids.numel(),),
      device=self.device,
      dtype=torch.long,
    )
