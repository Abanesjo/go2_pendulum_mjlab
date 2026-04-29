"""MJLab env config for the Isaac-compatible Go2 pendulum task."""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as env_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from go2_pendulum_mjlab.tasks.go2_pendulum.constants import (
  ACTION_SCALE,
  DEFAULT_LEG_JOINT_POS,
  DIFFICULTY_PRESETS,
  LEG_JOINT_NAMES,
  PENDULUM_JOINT_NAMES,
)
from go2_pendulum_mjlab.tasks.go2_pendulum.mdp import (
  OrderedGo2PdActionCfg,
  PositionGoalCommandCfg,
  action_acc_l2,
  action_l2,
  action_rate_l2,
  ang_vel_xy_l2,
  balanced_movement,
  base_height_l2,
  body_contact_force,
  clock_inputs,
  early_termination,
  feet_air_time,
  feet_clearance,
  flat_orientation_reward,
  finite_diff_base_lin_vel_b,
  goal_error_b,
  imu_ang_vel_b,
  isaac_difficulty,
  joint_actuator_effort_l2,
  joint_pos_rel,
  joint_vel,
  lin_vel_z_l2,
  pendulum_fallen,
  pendulum_upright,
  pendulum_velocity_l2,
  position_goal_violation,
  position_tracking,
  progress,
  projected_gravity_from_imu,
  raw_last_action,
  randomize_ordered_pd_gains,
  reset_pendulum_by_sign_magnitude,
  set_pendulum_joint_limits,
  sustained,
  tracking_contacts_shaped_force,
  undesired_contacts,
  yaw_alignment,
)
from go2_pendulum_mjlab.tasks.go2_pendulum.robot_cfg import get_go2_pendulum_robot_cfg

_LEG_CFG = SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)
_PEND_CFG = SceneEntityCfg("robot", joint_names=PENDULUM_JOINT_NAMES, preserve_order=True)
_FEET_GEOMS = ("FL", "FR", "RL", "RR")
_FEET_BODIES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
_THIGH_BODIES = ("FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh")


def _obs_terms(noisy: bool) -> dict[str, ObservationTermCfg]:
  return {
    "base_lin_vel_b": ObservationTermCfg(
      func=finite_diff_base_lin_vel_b,
      params={"asset_cfg": SceneEntityCfg("robot")},
      noise=Unoise(n_min=-0.02, n_max=0.02) if noisy else None,
    ),
    "base_ang_vel_b": ObservationTermCfg(
      func=imu_ang_vel_b,
      noise=Unoise(n_min=-0.2, n_max=0.2) if noisy else None,
    ),
    "projected_gravity_b": ObservationTermCfg(func=projected_gravity_from_imu),
    "goal_error_b": ObservationTermCfg(func=goal_error_b),
    "leg_joint_pos_rel": ObservationTermCfg(
      func=joint_pos_rel,
      params={"asset_cfg": _LEG_CFG},
      noise=Unoise(n_min=-math.radians(1.0), n_max=math.radians(1.0)) if noisy else None,
    ),
    "leg_joint_vel": ObservationTermCfg(
      func=joint_vel,
      params={"asset_cfg": _LEG_CFG},
      noise=Unoise(n_min=-math.radians(5.0), n_max=math.radians(5.0)) if noisy else None,
    ),
    "pendulum_pos": ObservationTermCfg(
      func=joint_pos_rel,
      params={"asset_cfg": _PEND_CFG},
      noise=Unoise(n_min=-math.radians(0.5), n_max=math.radians(0.5)) if noisy else None,
    ),
    "pendulum_vel": ObservationTermCfg(
      func=joint_vel,
      params={"asset_cfg": _PEND_CFG},
      noise=Unoise(n_min=-math.radians(2.0), n_max=math.radians(2.0)) if noisy else None,
    ),
    "last_action": ObservationTermCfg(func=raw_last_action),
    "clock_inputs": ObservationTermCfg(func=clock_inputs),
  }


def go2_pendulum_mjlab_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  preset = DIFFICULTY_PRESETS[5] if play else DIFFICULTY_PRESETS[1]

  observations = {
    "policy": ObservationGroupCfg(
      terms=_obs_terms(noisy=True),
      concatenate_terms=True,
      enable_corruption=True,
      nan_policy="sanitize",
    ),
    "critic": ObservationGroupCfg(
      terms=_obs_terms(noisy=False),
      concatenate_terms=True,
      enable_corruption=False,
      nan_policy="sanitize",
    ),
  }

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": OrderedGo2PdActionCfg(
      entity_name="robot",
      joint_names=LEG_JOINT_NAMES,
      default_joint_pos=DEFAULT_LEG_JOINT_POS,
      action_scale=ACTION_SCALE,
      stiffness=25.0,
      damping=0.6,
      effort_limit=23.5,
    )
  }

  commands: dict[str, CommandTermCfg] = {
    "position_goal": PositionGoalCommandCfg(
      entity_name="robot",
      resampling_time_range=(1.0e9, 1.0e9),
      dist_range=preset["goal_dist"],
      bearing_range=preset["goal_bearing"],
      yaw_range=preset["goal_yaw"],
      debug_vis=True,
    )
  }

  events = {
    "reset_scene_to_default": EventTermCfg(func=env_mdp.reset_scene_to_default, mode="reset"),
    "pendulum_limits": EventTermCfg(
      func=set_pendulum_joint_limits,
      mode="reset",
      params={"asset_cfg": _PEND_CFG, "limit_range": preset["pendulum_limits"]},
    ),
    "reset_pendulum": EventTermCfg(
      func=reset_pendulum_by_sign_magnitude,
      mode="reset",
      params={"asset_cfg": _PEND_CFG, "angle_range": preset["pendulum_reset"]},
    ),
    "pendulum_damping": EventTermCfg(
      mode="reset",
      func=dr.joint_damping,
      params={
        "asset_cfg": _PEND_CFG,
        "operation": "abs",
        "ranges": (0.001, 0.05),
        "shared_random": True,
      },
    ),
    "pendulum_mass": EventTermCfg(
      mode="startup",
      func=dr.body_mass,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("pendulum_ee",)),
        "operation": "scale",
        "ranges": (0.5, 3.0),
      },
    ),
    "base_mass": EventTermCfg(
      mode="startup",
      func=dr.body_mass,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
        "operation": "scale",
        "ranges": (0.9, 1.2),
      },
    ),
    "motor_gains": EventTermCfg(
      mode="reset",
      func=randomize_ordered_pd_gains,
      params={
        "action_name": "joint_pos",
        "kp_range": (0.9, 1.1),
        "kd_range": (0.9, 1.1),
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-math.radians(1.0), math.radians(1.0)),
      },
    ),
    "push_robot": EventTermCfg(
      func=env_mdp.apply_body_impulse,
      mode="step",
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
        "force_range": preset["push_force_xy"],
        "torque_range": (0.0, 0.0),
        "duration_s": (0.05, 0.15),
        "cooldown_s": (5.0, 10.0),
      },
    ),
  }
  if play:
    events.pop("push_robot")
    events.pop("base_mass")
    events.pop("motor_gains")
    events.pop("encoder_bias")
    events.pop("pendulum_damping")
    events.pop("pendulum_mass")

  rewards = {
    "position_tracking": RewardTermCfg(
      func=position_tracking,
      weight=0.4,
      params={"command_name": "position_goal", "std": 0.3},
    ),
    "progress": RewardTermCfg(
      func=progress,
      weight=10.0,
      params={"command_name": "position_goal"},
    ),
    "yaw_alignment": RewardTermCfg(
      func=yaw_alignment,
      weight=0.3,
      params={"command_name": "position_goal", "std": 0.2},
    ),
    "pendulum_upright": RewardTermCfg(
      func=pendulum_upright,
      weight=0.45,
      params={"asset_cfg": _PEND_CFG, "std": 0.15},
    ),
    "pendulum_velocity": RewardTermCfg(
      func=pendulum_velocity_l2,
      weight=-0.1,
      params={"asset_cfg": _PEND_CFG},
    ),
    "balanced_movement": RewardTermCfg(
      func=balanced_movement,
      weight=0.1,
      params={"asset_cfg": _PEND_CFG},
    ),
    "action_magnitude": RewardTermCfg(
      func=action_l2,
      weight=-0.1 * (ACTION_SCALE**2),
    ),
    "rew_action_rate": RewardTermCfg(
      func=action_rate_l2,
      weight=-0.1 * (ACTION_SCALE**2),
    ),
    "action_acc": RewardTermCfg(
      func=action_acc_l2,
      weight=-0.1 * (ACTION_SCALE**2),
    ),
    "torque": RewardTermCfg(
      func=joint_actuator_effort_l2,
      weight=-0.0005,
      params={"asset_cfg": _LEG_CFG},
    ),
    "orient": RewardTermCfg(func=flat_orientation_reward, weight=0.8, params={"std": 0.05}),
    "base_height": RewardTermCfg(
      func=base_height_l2,
      weight=0.2,
      params={"target_height": 0.33, "std": 0.1},
    ),
    "lin_vel_z": RewardTermCfg(func=lin_vel_z_l2, weight=-2.0),
    "dof_vel": RewardTermCfg(func=env_mdp.joint_vel_l2, weight=-0.003, params={"asset_cfg": _LEG_CFG}),
    "dof_acc": RewardTermCfg(func=env_mdp.joint_acc_l2, weight=-2.5e-7, params={"asset_cfg": _LEG_CFG}),
    "ang_vel_xy": RewardTermCfg(func=ang_vel_xy_l2, weight=-0.01),
    "feet_clearance": RewardTermCfg(
      func=feet_clearance,
      weight=-20.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=_FEET_BODIES, preserve_order=True),
        "period_s": 1.0 / 3.0,
        "offsets": (0.5, 0.0, 0.0, 0.5),
        "command_name": "position_goal",
        "command_threshold": 0.1,
      },
    ),
    "feet_air_time": RewardTermCfg(
      func=feet_air_time,
      weight=0.1,
      params={"sensor_name": "feet_ground_contact", "command_name": "position_goal", "command_threshold": 0.1},
    ),
    "tracking_contacts_shaped_force": RewardTermCfg(
      func=tracking_contacts_shaped_force,
      weight=1.0,
      params={
        "sensor_name": "feet_ground_contact",
        "period_s": 1.0 / 3.0,
        "offsets": (0.5, 0.0, 0.0, 0.5),
        "command_name": "position_goal",
        "command_threshold": 0.1,
      },
    ),
    "undesired_contacts": RewardTermCfg(
      func=undesired_contacts,
      weight=-1.0,
      params={"sensor_name": "undesired_ground_contact"},
    ),
    "termination_penalty": RewardTermCfg(func=early_termination, weight=-5.0),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=env_mdp.time_out, time_out=True),
    "base_tilt": TerminationTermCfg(func=env_mdp.bad_orientation, params={"limit_angle": math.pi / 3}),
    "base_contact": TerminationTermCfg(
      func=sustained,
      params={
        "inner": {"func": body_contact_force, "params": {"sensor_name": "base_contact", "threshold": 1.0}},
        "duration_s": 0.0,
        "grace_period_s": 0.5,
      },
    ),
    "pendulum_contact": TerminationTermCfg(
      func=sustained,
      params={
        "inner": {
          "func": body_contact_force,
          "params": {"sensor_name": "pendulum_contact", "threshold": 1.0},
        },
        "duration_s": 0.0,
        "grace_period_s": 3.0,
      },
    ),
    "pendulum_fallen": TerminationTermCfg(
      func=sustained,
      params={
        "inner": {"func": pendulum_fallen, "params": {"asset_cfg": _PEND_CFG, "angle_rad": preset["pendulum_terminate_angle"]}},
        "duration_s": preset["pendulum_terminate_duration"],
        "grace_period_s": 3.0,
      },
    ),
    "base_height_too_low": TerminationTermCfg(
      func=sustained,
      params={
        "inner": {"func": env_mdp.root_height_below_minimum, "params": {"minimum_height": 0.28}},
        "duration_s": 10.0,
        "grace_period_s": 0.1,
      },
    ),
    "position_goal_violation": TerminationTermCfg(
      func=sustained,
      params={
        "inner": {
          "func": position_goal_violation,
          "params": {"command_name": "position_goal", "max_dist": preset["position_tolerance"]},
        },
        "duration_s": 15.0,
        "grace_period_s": 0.1,
      },
    ),
  }
  if play:
    terminations.pop("base_contact", None)
    terminations.pop("position_goal_violation", None)

  curriculum = {}
  if not play:
    curriculum = {
      "isaac_difficulty": CurriculumTermCfg(
        func=isaac_difficulty,
        params={
          "command_name": "position_goal",
          "pendulum_reset_event_name": "reset_pendulum",
          "pendulum_limits_event_name": "pendulum_limits",
          "pendulum_termination_name": "pendulum_fallen",
          "position_termination_name": "position_goal_violation",
          "push_event_name": "push_robot",
          "total_steps": 25_000 * 32,
          "override_level": -1,
        },
      )
    }

  sensors = (
    ContactSensorCfg(
      name="feet_ground_contact",
      primary=ContactMatch(mode="geom", pattern=_FEET_GEOMS, entity="robot"),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"),
      reduce="netforce",
      num_slots=1,
      track_air_time=True,
    ),
    ContactSensorCfg(
      name="base_contact",
      primary=ContactMatch(mode="body", pattern="base_link", entity="robot"),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"),
      reduce="netforce",
      num_slots=1,
    ),
    ContactSensorCfg(
      name="pendulum_contact",
      primary=ContactMatch(mode="body", pattern="pendulum_ee", entity="robot"),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"),
      reduce="netforce",
      num_slots=1,
    ),
    ContactSensorCfg(
      name="undesired_ground_contact",
      primary=ContactMatch(mode="body", pattern=_THIGH_BODIES, entity="robot"),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"),
      reduce="netforce",
      num_slots=1,
    ),
  )

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane", terrain_generator=None),
      entities={"robot": get_go2_pendulum_robot_cfg()},
      sensors=sensors,
      num_envs=1 if play else 4096,
      env_spacing=4.0,
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="base_link",
      distance=2.5,
      elevation=-10.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=1500,
      contact_sensor_maxmatch=64,
      mujoco=MujocoCfg(
        timestep=0.005,
        cone="elliptic",
        impratio=100.0,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=10.0,
    scale_rewards_by_dt=False,
  )
