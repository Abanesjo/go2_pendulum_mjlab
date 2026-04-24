"""Position-goal command matching the IsaacLab/C++ body-frame goal error."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import wrap_to_pi


@dataclass(kw_only=True)
class PositionGoalCommandCfg(CommandTermCfg):
  entity_name: str
  dist_range: tuple[float, float] = (0.0, 0.0)
  bearing_range: tuple[float, float] = (-math.pi, math.pi)
  yaw_range: tuple[float, float] = (0.0, 0.0)
  marker_height: float = 0.08
  marker_length: float = 0.35
  marker_width: float = 0.025
  marker_color: tuple[float, float, float, float] = (1.0, 0.55, 0.0, 0.9)
  gui_position_range: float = 2.0

  def build(self, env) -> "PositionGoalCommand":
    return PositionGoalCommand(self, env)


class PositionGoalCommand(CommandTerm):
  cfg: PositionGoalCommandCfg

  def __init__(self, cfg: PositionGoalCommandCfg, env):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.target_pos_w = torch.zeros(env.num_envs, 2, device=env.device)
    self.target_yaw_w = torch.zeros(env.num_envs, device=env.device)
    self._state_error = torch.zeros(env.num_envs, 3, device=env.device)
    self.metrics["error_pos_xy"] = torch.zeros(env.num_envs, device=env.device)
    self.metrics["error_yaw"] = torch.zeros(env.num_envs, device=env.device)
    self._gui_get_env_idx: Callable[[], int] | None = None
    self._gui_on_change: Callable[[], None] | None = None
    self._gui_request_action: Callable[[str, Any], None] | None = None
    self._gui_x = None
    self._gui_y = None
    self._gui_yaw = None
    self._gui_syncing = False

  @property
  def command(self) -> torch.Tensor:
    return self._state_error

  def _update_metrics(self) -> None:
    dist = torch.linalg.vector_norm(self._state_error[:, :2], dim=-1)
    max_step = self.cfg.resampling_time_range[1] / self._env.step_dt
    self.metrics["error_pos_xy"] += dist / max_step
    self.metrics["error_yaw"] += self._state_error[:, 2].abs() / max_step

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    d = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.dist_range)
    bearing = torch.empty_like(d).uniform_(*self.cfg.bearing_range)
    yaw_offset = torch.empty_like(d).uniform_(*self.cfg.yaw_range)
    root_pos = self.robot.data.root_link_pos_w[env_ids, :2]
    root_yaw = self.robot.data.heading_w[env_ids]
    self.target_pos_w[env_ids, 0] = root_pos[:, 0] + d * torch.cos(bearing)
    self.target_pos_w[env_ids, 1] = root_pos[:, 1] + d * torch.sin(bearing)
    self.target_yaw_w[env_ids] = wrap_to_pi(root_yaw + yaw_offset)

  def _update_command(self) -> None:
    root_pos = self.robot.data.root_link_pos_w[:, :2]
    yaw = self.robot.data.heading_w
    delta = self.target_pos_w - root_pos
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    self._state_error[:, 0] = c * delta[:, 0] + s * delta[:, 1]
    self._state_error[:, 1] = -s * delta[:, 0] + c * delta[:, 1]
    self._state_error[:, 2] = wrap_to_pi(self.target_yaw_w - yaw)

  def _selected_gui_env_id(self) -> int:
    if self._gui_get_env_idx is None:
      return 0
    return max(0, min(int(self._gui_get_env_idx()), self.num_envs - 1))

  @staticmethod
  def _wrap_angle_rad(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi

  def _sync_gui_from_selected_env(self) -> None:
    if self._gui_x is None or self._gui_y is None or self._gui_yaw is None:
      return
    env_id = self._selected_gui_env_id()
    target_xy = self.target_pos_w[env_id].detach().cpu()
    target_yaw_deg = math.degrees(float(self.target_yaw_w[env_id].detach().cpu()))
    self._gui_syncing = True
    self._gui_x.value = float(target_xy[0])
    self._gui_y.value = float(target_xy[1])
    self._gui_yaw.value = target_yaw_deg
    self._gui_syncing = False

  def _apply_gui_target_to_selected_env(self) -> None:
    if self._gui_syncing:
      return
    if self._gui_x is None or self._gui_y is None or self._gui_yaw is None:
      return
    env_id = self._selected_gui_env_id()
    yaw = self._wrap_angle_rad(math.radians(float(self._gui_yaw.value)))
    with torch.no_grad():
      self.target_pos_w[env_id, 0] = float(self._gui_x.value)
      self.target_pos_w[env_id, 1] = float(self._gui_y.value)
      self.target_yaw_w[env_id] = yaw
      self._update_command()
    if self._gui_on_change is not None:
      self._gui_on_change()

  def create_gui(
    self,
    name: str,
    server,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    """Create Viser controls for manually setting the world-frame goal pose."""
    self._gui_get_env_idx = get_env_idx
    self._gui_on_change = on_change
    self._gui_request_action = request_action

    limit = self.cfg.gui_position_range
    with server.gui.add_folder(name.replace("_", " ").capitalize()):
      self._gui_x = server.gui.add_slider(
        "Target x",
        min=-limit,
        max=limit,
        step=0.01,
        initial_value=0.0,
      )
      self._gui_y = server.gui.add_slider(
        "Target y",
        min=-limit,
        max=limit,
        step=0.01,
        initial_value=0.0,
      )
      self._gui_yaw = server.gui.add_slider(
        "Target yaw deg",
        min=-180.0,
        max=180.0,
        step=1.0,
        initial_value=0.0,
      )
      sync_btn = server.gui.add_button("Sync from selected")
      reset_btn = server.gui.add_button("Reset selected with GUI goal")

    self._sync_gui_from_selected_env()

    for slider in (self._gui_x, self._gui_y, self._gui_yaw):
      @slider.on_update
      def _(_) -> None:
        self._apply_gui_target_to_selected_env()

    @sync_btn.on_click
    def _(_) -> None:
      self._sync_gui_from_selected_env()
      if self._gui_on_change is not None:
        self._gui_on_change()

    @reset_btn.on_click
    def _(_) -> None:
      if self._gui_request_action is not None:
        self._gui_request_action("CUSTOM", {"type": "gui_reset", "all_envs": False})

  def apply_gui_reset(self, env_ids: torch.Tensor) -> bool:
    """Preserve the manually selected GUI goal after a Viser reset."""
    if self._gui_x is None or self._gui_y is None or self._gui_yaw is None:
      return False
    del env_ids
    self._apply_gui_target_to_selected_env()
    return True

  def _debug_vis_impl(self, visualizer) -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    targets = self.target_pos_w.detach().cpu().numpy()
    yaws = self.target_yaw_w.detach().cpu().numpy()
    z = self.cfg.marker_height
    length = self.cfg.marker_length

    for env_id in env_indices:
      yaw = float(yaws[env_id])
      start = np.array((targets[env_id, 0], targets[env_id, 1], z), dtype=np.float32)
      end = start + length * np.array((math.cos(yaw), math.sin(yaw), 0.0), dtype=np.float32)
      visualizer.add_arrow(
        start=start,
        end=end,
        color=self.cfg.marker_color,
        width=self.cfg.marker_width,
        label=f"position_goal_{env_id}",
      )
