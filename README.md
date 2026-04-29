# Go2 Pendulum MJLab

MJLab port of the IsaacLab Go2 pendulum task.

The registered task id is:

```text
Go2-Pendulum-MJLab-v0
```

The policy ABI is kept compatible with the IsaacLab-trained policy and the
`go2_mujoco` deploy code:

- observation dimension: `56`
- action dimension: `12`
- action order:
  `FL_hip_joint, FR_hip_joint, RL_hip_joint, RR_hip_joint,
  FL_thigh_joint, FR_thigh_joint, RL_thigh_joint, RR_thigh_joint,
  FL_calf_joint, FR_calf_joint, RL_calf_joint, RR_calf_joint`
- action mapping: `q_des = default_joint_pos + 0.25 * raw_action`

## Setup

From the workspace:

```bash
cd ~/Documents/mjlab_projects
conda activate mujoco
```

Editable install is optional because the scripts add `src` to `PYTHONPATH`
themselves. If you want imports to work from any directory:

```bash
cd ~/Documents/mjlab_projects/go2_pendulum_mjlab
pip install -e .
```

Useful render/cache environment variables:

```bash
export MUJOCO_GL=egl
export XDG_CACHE_HOME=/tmp
export MPLCONFIGDIR=/tmp/mpl
export WARP_CACHE_PATH=/tmp/warp
```

## Train

Example from-scratch training command with all wrapper arguments shown:

```bash
cd ~/Documents/mjlab_projects
conda activate mujoco

python scripts/train.py \
  --device cuda:0 \
  --resume-file None \
  --video True \
  --video-length 200 \
  --video-interval 2000 \
  --gpu-ids 0 \
  --env.scene.num-envs 4096 \
  --env.seed 42 \
  --agent.seed 42 \
  --agent.max-iterations 25000 \
  --agent.run-name finetune_v1
```

### `train.py` arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `--device STR` | `None` | Force a device such as `cuda:0` or `cpu`. If omitted, the script uses CUDA when visible, otherwise CPU. |
| `--resume-file STR` | `None` | Explicit checkpoint path to load before training. |
| `--video {True,False}` | `false` | Enable MJLab offscreen video recording during training. |
| `--video-length INT` | `200` | Number of frames per recorded training video. |
| `--video-interval INT` | `2000` | Start a new video every N environment steps. |
| `--gpu-ids [INT,...]` | `[0]` | GPUs exposed through `CUDA_VISIBLE_DEVICES`. Use `None` for CPU or `all` for all visible GPUs. |
| `--env.*` | task config | Any nested field on `ManagerBasedRlEnvCfg`, for example `--env.scene.num-envs 1024`. |
| `--agent.*` | PPO config | Any nested field on `RslRlBaseRunnerCfg`, for example `--agent.max-iterations 1500`. |

Set `--resume-file /path/to/model_1000.pt` to continue training from a checkpoint. To see the full generated list of nested `--env.*` and `--agent.*` options:

```bash
python scripts/train.py --help
```

## Play

Example playback/export command with all wrapper arguments shown:

```bash
cd ~/Documents/mjlab_projects
conda activate mujoco

python scripts/play.py \
  --checkpoint-file /path/to/model_1000.pt \
  --agent trained \
  --num-envs 1 \
  --device cuda:0 \
  --viewer auto \
  --export false
```

Set `--agent zero` or `--agent random` for dummy policies. Set `--viewer native` or `--viewer viser` to force a viewer. Set `--export true` to write:

```text
policy.pt
policy.onnx
```

next to the checkpoint file.

### `play.py` arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `--checkpoint-file STR` | `None` | Checkpoint to load for `--agent trained` or export. Required for trained policy playback/export. |
| `--agent STR` | `trained` | Policy source. Supported values are `trained`, `zero`, and `random`. |
| `--num-envs INT` | `1` | Number of parallel play environments. |
| `--device STR` | `None` | Force a device such as `cuda:0` or `cpu`. If omitted, the script uses CUDA when available. |
| `--viewer STR` | `auto` | Viewer backend. Supported values are `auto`, `native`, and `viser`. |
| `--export {true,false}` | `false` | Export `policy.pt` and `policy.onnx`, then exit. |

To show the generated play CLI:

```bash
python scripts/play.py --help
```

## Outputs

Training logs are written under:

```text
logs/rsl_rl/go2_pendulum_mjlab/<timestamp>/
```

Training videos, when enabled, are written under:

```text
logs/rsl_rl/go2_pendulum_mjlab/<timestamp>/videos/train/
```
