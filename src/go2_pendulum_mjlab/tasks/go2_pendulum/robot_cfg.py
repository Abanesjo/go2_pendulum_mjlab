"""Robot/entity configuration using the deployment MuJoCo asset."""

from __future__ import annotations

from pathlib import Path

import mujoco

from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from go2_pendulum_mjlab.tasks.go2_pendulum.constants import DEFAULT_JOINT_POS, LEG_JOINT_NAMES

ASSET_DIR = Path(__file__).resolve().parents[2] / "assets" / "go2_pendulum"
GO2_PENDULUM_XML = ASSET_DIR / "go2_pendulum.xml"
LEG_ARMATURE = 0.0
LEG_FRICTIONLOSS = 0.0
LEG_VISCOUS_DAMPING = 0.0


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(GO2_PENDULUM_XML))
  # MJWarp native CCD does not support nonzero geom margins. Normalize the
  # local training copy only; the deployment XML remains untouched.
  for geom in spec.geoms:
    geom.margin = 0.0
  while spec.keys:
    spec.delete(spec.keys[0])
  return spec


INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.4),
  joint_pos=DEFAULT_JOINT_POS,
  joint_vel={".*": 0.0},
)


GO2_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    XmlActuatorCfg(
      target_names_expr=LEG_JOINT_NAMES,
      command_field="effort",
      armature=LEG_ARMATURE,
      frictionloss=LEG_FRICTIONLOSS,
      viscous_damping=LEG_VISCOUS_DAMPING,
    ),
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_go2_pendulum_robot_cfg() -> EntityCfg:
  """Return a fresh local Go2+pendulum entity config."""
  return EntityCfg(
    init_state=INIT_STATE,
    spec_fn=get_spec,
    articulation=GO2_ARTICULATION,
  )
