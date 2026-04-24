"""Go2 pendulum MJLab task registration."""

from mjlab.tasks.registry import register_mjlab_task

from go2_pendulum_mjlab.tasks.go2_pendulum.config.go2.env_cfg import (
  go2_pendulum_mjlab_env_cfg,
)
from go2_pendulum_mjlab.tasks.go2_pendulum.config.go2.rl_cfg import (
  go2_pendulum_mjlab_ppo_runner_cfg,
)

TASK_ID = "Go2-Pendulum-MJLab-v0"

register_mjlab_task(
  task_id=TASK_ID,
  env_cfg=go2_pendulum_mjlab_env_cfg(play=False),
  play_env_cfg=go2_pendulum_mjlab_env_cfg(play=True),
  rl_cfg=go2_pendulum_mjlab_ppo_runner_cfg(),
)

__all__ = ["TASK_ID"]

