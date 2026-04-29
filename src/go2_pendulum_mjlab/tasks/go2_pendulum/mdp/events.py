"""Reset and curriculum support events."""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from go2_pendulum_mjlab.tasks.go2_pendulum.mdp.actions import OrderedGo2PdAction


def randomize_ordered_pd_gains(
  env,
  env_ids: torch.Tensor | None,
  action_name: str,
  kp_range: tuple[float, float],
  kd_range: tuple[float, float],
) -> None:
  """Randomize the custom deployment-compatible PD action gains."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  term = env.action_manager.get_term(action_name)
  if not isinstance(term, OrderedGo2PdAction):
    raise TypeError(f"Expected OrderedGo2PdAction for '{action_name}', got {type(term).__name__}")
  kp = torch.empty((len(env_ids), 1), device=env.device).uniform_(*kp_range)
  kd = torch.empty((len(env_ids), 1), device=env.device).uniform_(*kd_range)
  term.stiffness[env_ids] = term.default_stiffness[env_ids] * kp
  term.damping[env_ids] = term.default_damping[env_ids] * kd


def reset_pendulum_by_sign_magnitude(
  env,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg,
  angle_range: tuple[float, float],
) -> None:
  """Sample a radial pendulum angle and map it to the two hinge joints."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  asset: Entity = env.scene[asset_cfg.name]
  joint_ids = asset_cfg.joint_ids
  joint_pos = asset.data.default_joint_pos[env_ids].clone()
  joint_vel = asset.data.default_joint_vel[env_ids].clone()

  count = joint_pos[:, joint_ids].shape[1]
  if count != 2:
    raise RuntimeError(f"Radial pendulum reset expects exactly two joints, got {count}.")
  lo, hi = angle_range
  if hi < lo:
    raise ValueError(f"angle_range max must be >= min, got {hi} < {lo}.")
  radius = torch.empty((len(env_ids),), device=env.device).uniform_(lo, hi)
  theta = torch.empty((len(env_ids),), device=env.device).uniform_(0.0, 2.0 * torch.pi)
  offsets = torch.stack((radius * torch.cos(theta), radius * torch.sin(theta)), dim=-1)
  joint_pos[:, joint_ids] = joint_pos[:, joint_ids] + offsets
  joint_vel[:, joint_ids] = 0.0
  asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def set_pendulum_joint_limits(
  env,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg,
  limit_range: tuple[float, float],
) -> None:
  """Update MJWarp joint position limits for pendulum joints."""
  del env_ids
  asset: Entity = env.scene[asset_cfg.name]
  ids = asset_cfg.joint_ids
  lo, hi = limit_range
  asset.data.joint_pos_limits[:, ids, 0] = lo
  asset.data.joint_pos_limits[:, ids, 1] = hi
  asset.data.soft_joint_pos_limits[:, ids, 0] = lo
  asset.data.soft_joint_pos_limits[:, ids, 1] = hi
