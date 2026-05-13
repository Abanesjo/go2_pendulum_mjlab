# Unitree MuJoCo Realism Defaults

These values mirror `references/unitree_mujoco/simulate/config.yaml` for the
`go2_pendulum` deployment path while preserving the MJLab policy ABI.

## Timing

| Parameter | Value |
| --- | --- |
| MuJoCo timestep | `0.005 s` |
| Control decimation | `4` |
| Policy/control period | `0.020 s` |
| Pendulum estimator rate | `200 Hz` |

## Observation Realism

Policy observations use delayed/noisy Unitree-style streams. Critic observations
use clean geometric equivalents in the same 56-D order.

| Stream | Value |
| --- | --- |
| Leg joint position noise | Gaussian std `0.0015 rad` |
| Leg joint position bias | Per-episode Gaussian std `0.003 rad` |
| Leg joint velocity noise | Gaussian std `0.06 rad/s` |
| Leg joint velocity bias | Per-episode Gaussian std `0.02 rad/s` |
| Leg joint velocity LPF | `tau = 0.015 s` |
| Lowstate delay / jitter / dropout | `8 ms` / `2 ms` / `0.001` |
| IMU gyro noise | Gaussian std `0.015 rad/s` |
| IMU quaternion perturbation | Gaussian std `0.003 rad` |
| Base pose delay / jitter / dropout | `10 ms` / `2 ms` / `0.001` |
| Base pose position noise / bias | `0.002 m` / `0.001 m` |
| Pendulum EE pose delay / jitter | `15 ms` / `3 ms` |
| Pendulum EE position noise / bias | `0.004 m` / `0.002 m` |

Pendulum position and velocity are estimated from base and pendulum end-effector
poses:

```text
v = R_base^T * (p_ee - p_base) - [-0.05, 0.0, 0.06]
joint1 = atan2(-v_y, v_z)
joint2 = atan2(v_x, hypot(v_y, v_z))
```

The estimator uses a causal Savitzky-Golay filter with `window=7`, `poly=2`,
`delta=0.005`, and `pos=window-1`.

## Actuator Realism

| Parameter | Value |
| --- | --- |
| Command delay / jitter / dropout / hold | `6 ms` / `2 ms` / `0.001` / `0.002` |
| Nominal PD gains | `Kp = 25.0`, `Kd = 0.6` |
| Kp scale | Gaussian mean `1.0`, std `0.08` |
| Kd scale | Gaussian mean `1.0`, std `0.12` |
| Motor strength | Gaussian std `0.04` around `1.0` |
| Torque limit scales | abduction `0.85`, hip `0.85`, knee `0.75` |
| Speed limits | abduction `28 rad/s`, hip `28 rad/s`, knee `35 rad/s` |
| Minimum speed-torque scale | `0.35` |
| Torque LPF | `tau = 0.010 s` |
| Torque rate limit | `600 Nm/s` |
| Torque deadband | `0.15 Nm` |
| Torque noise | Gaussian std `0.03 Nm` |

## Contact

| Parameter | Value |
| --- | --- |
| Foot friction | `0.55 0.025 0.010` on all four foot geoms |

The previous broad base and pendulum mass randomizations are disabled by default
because they are not present in the Unitree MuJoCo realism config.
