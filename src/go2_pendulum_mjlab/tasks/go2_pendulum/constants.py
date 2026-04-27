"""Policy ABI constants for the IsaacLab-compatible Go2 pendulum task."""

from __future__ import annotations

import math

OBS_DIM = 56
ACTION_DIM = 12
ACTION_SCALE = 0.25

LEG_JOINT_NAMES: tuple[str, ...] = (
  "FL_hip_joint",
  "FR_hip_joint",
  "RL_hip_joint",
  "RR_hip_joint",
  "FL_thigh_joint",
  "FR_thigh_joint",
  "RL_thigh_joint",
  "RR_thigh_joint",
  "FL_calf_joint",
  "FR_calf_joint",
  "RL_calf_joint",
  "RR_calf_joint",
)

DEFAULT_LEG_JOINT_POS: tuple[float, ...] = (
  0.1,
  -0.1,
  0.1,
  -0.1,
  0.8,
  0.8,
  1.0,
  1.0,
  -1.5,
  -1.5,
  -1.5,
  -1.5,
)

PENDULUM_JOINT_NAMES: tuple[str, ...] = ("pendulum_joint1", "pendulum_joint2")
PENDULUM_PHYSICAL_LIMIT_RAD = math.radians(90.0)

# Pendulum angle magnitudes are radial angles from vertical:
# sqrt(pendulum_joint1**2 + pendulum_joint2**2).
# ``pendulum_limits`` remain per-joint simulator limits.

DEFAULT_JOINT_POS: dict[str, float] = {
  **dict(zip(LEG_JOINT_NAMES, DEFAULT_LEG_JOINT_POS, strict=True)),
  "pendulum_joint1": 0.0,
  "pendulum_joint2": 0.0,
}

DIFFICULTY_PRESETS: dict[int, dict[str, object]] = {
  1: dict(
    goal_dist=(0.0, 0.1),
    goal_bearing=(math.radians(-180), math.radians(180)),
    goal_yaw=(0.0, 0.0),
    pendulum_reset=(math.radians(0.0), math.radians(5.0)),
    pendulum_limits=(-PENDULUM_PHYSICAL_LIMIT_RAD, PENDULUM_PHYSICAL_LIMIT_RAD),
    pendulum_terminate_angle=math.radians(19.0),
    pendulum_terminate_duration=0.5,
    position_tolerance=1.0,
    push_force_xy=(0.0, 0.0),
  ),
  2: dict(
    goal_dist=(0.0, 0.15),
    goal_bearing=(math.radians(-180), math.radians(180)),
    goal_yaw=(math.radians(-15), math.radians(15)),
    pendulum_reset=(math.radians(0.0), math.radians(10.0)),
    pendulum_limits=(-PENDULUM_PHYSICAL_LIMIT_RAD, PENDULUM_PHYSICAL_LIMIT_RAD),
    pendulum_terminate_angle=math.radians(19.0),
    pendulum_terminate_duration=0.5,
    position_tolerance=0.5,
    push_force_xy=(0.0, 0.0),
  ),
  3: dict(
    goal_dist=(0.1, 0.3),
    goal_bearing=(math.radians(-180), math.radians(180)),
    goal_yaw=(math.radians(-30), math.radians(30)),
    pendulum_reset=(math.radians(5.0), math.radians(15.0)),
    pendulum_limits=(-PENDULUM_PHYSICAL_LIMIT_RAD, PENDULUM_PHYSICAL_LIMIT_RAD),
    pendulum_terminate_angle=math.radians(19.0),
    pendulum_terminate_duration=0.5,
    position_tolerance=0.3,
    push_force_xy=(0.0, 0.0),
  ),
  4: dict(
    goal_dist=(0.2, 0.5),
    goal_bearing=(math.radians(-180), math.radians(180)),
    goal_yaw=(math.radians(-45), math.radians(45)),
    pendulum_reset=(math.radians(10.0), math.radians(19.9)),
    pendulum_limits=(-PENDULUM_PHYSICAL_LIMIT_RAD, PENDULUM_PHYSICAL_LIMIT_RAD),
    pendulum_terminate_angle=math.radians(19.0),
    pendulum_terminate_duration=0.5,
    position_tolerance=0.2,
    push_force_xy=(-5.0, 5.0),
  ),
  5: dict(
    goal_dist=(0.3, 0.5),
    goal_bearing=(math.radians(-180), math.radians(180)),
    goal_yaw=(math.radians(-60), math.radians(60)),
    pendulum_reset=(math.radians(19.8), math.radians(19.9)),
    pendulum_limits=(-PENDULUM_PHYSICAL_LIMIT_RAD, PENDULUM_PHYSICAL_LIMIT_RAD),
    pendulum_terminate_angle=math.radians(19.0),
    pendulum_terminate_duration=0.5,
    position_tolerance=0.2,
    push_force_xy=(0.0, 0.0),
  ),
}
