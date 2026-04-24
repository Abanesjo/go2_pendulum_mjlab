"""Gait helper functions matching the Isaac/C++ trot clock."""

from __future__ import annotations

import torch


def foot_phases(env, period_s: float = 1.0 / 3.0, offsets: tuple[float, ...] = (0.5, 0.0, 0.0, 0.5)) -> torch.Tensor:
  base = torch.remainder(env.episode_length_buf.to(torch.float32) * env.step_dt / period_s, 1.0)
  return torch.remainder(base[:, None] + torch.tensor(offsets, device=env.device), 1.0)


def clock_inputs_from_phase(phase: torch.Tensor, duty_cycle: float = 0.5) -> torch.Tensor:
  return torch.sin(2.0 * torch.pi * swing_phase_profile(phase, duty_cycle=duty_cycle))


def swing_phase_profile(phase: torch.Tensor, duty_cycle: float = 0.5) -> torch.Tensor:
  return torch.where(
    phase < duty_cycle,
    phase * (0.5 / duty_cycle),
    0.5 + (phase - duty_cycle) * (0.5 / (1.0 - duty_cycle)),
  )


def desired_contact_states(phase: torch.Tensor, duty_cycle: float = 0.5) -> torch.Tensor:
  kappa = 0.07
  normal = torch.distributions.normal.Normal(0.0, kappa)
  return normal.cdf(phase) * (1.0 - normal.cdf(phase - duty_cycle)) + normal.cdf(phase - 1.0) * (
    1.0 - normal.cdf(phase - duty_cycle - 1.0)
  )

