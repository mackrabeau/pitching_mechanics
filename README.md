# pitching_mechanics

Physics-based pitch delivery replay and torque analysis using [Driveline OpenBiomechanics Project (OBP)](https://github.com/drivelinebaseball/openbiomechanics) landmark data and [MuJoCo](https://mujoco.org/).

## Overview

This project loads motion-capture joint-center landmarks from the OBP baseball pitching dataset, builds a pitcher-specific MuJoCo model scaled to the athlete's actual segment lengths, and replays the full delivery (foot plant → ball release → follow-through) as a kinematic playback with inverse-dynamics torque computation.

The goal is to produce a reference torque baseline that serves as the foundation for reinforcement-learning-based pitch optimization (see `TO_DO.txt`).

## Project Structure

```
mujoco/                          # https://github.com/google-deepmind/mujoco
openbiomechanics/                # https://github.com/drivelineresearch/openbiomechanics
pitching_mechanics/
├── __init__.py
├── obp_fullsig.py               # Load OBP full-signal landmarks + events
├── site_ik.py                    # Damped least-squares IK solver (mj_jacSite)
├── build_pitcher_fullbody.py     # Generate full-body MJCF from OBP landmarks
├── replay_pitcher_fullbody.py    # Replay + inverse dynamics + ball release
├── ball_flight.py               # Ball release speed, strike check, RL reward
├── models/
│   └── pitcher_fullbody_*.xml    # Generated per-pitch MJCF models
└── logs/
    └── torques_*.csv             # Inverse-dynamics torque logs
TO_DO.txt                         # RL roadmap and current status
```

## How It Works

1. **OBP Data Loading** (`obp_fullsig.py`): Reads `landmarks.zip` and `force_plate.zip` from the OBP dataset. Extracts 12 joint-center trajectories (throwing arm, glove arm, both legs, both hips) and event timestamps (foot plant, MER, ball release, MIR).

2. **Model Generation** (`build_pitcher_fullbody.py`): Measures segment lengths and body offsets from OBP landmarks at a reference event (default: foot plant). Generates a full-body MJCF with:
   - Pelvis (freejoint — 6 DOF root)
   - Trunk (3 DOF: yaw / pitch / roll)
   - Throwing arm (3 DOF shoulder + 1 DOF elbow)
   - Glove arm (3 DOF shoulder + 1 DOF elbow)
   - Rear leg (3 DOF hip + 1 DOF knee + 1 DOF ankle)
   - Lead leg (3 DOF hip + 1 DOF knee + 1 DOF ankle)
   - **21 hinge DOFs total**, position actuators on every joint

3. **Replay + Inverse Dynamics** (`replay_pitcher_fullbody.py`):
   - Pre-computes IK trajectory for the full replay window (440 frames at 1 kHz)
   - Smooths the trajectory, then derives qvel/qacc via finite differences
   - Runs `mj_inverse` at each frame to compute the exact joint torques needed to produce the observed motion
   - Computes ball release speed and strike-zone feasibility at BR_time
   - Plays back the trajectory in the MuJoCo viewer (kinematic — positions set directly)
   - Writes a torque CSV log when `--log-torques` is specified

4. **Ball Release & Strike Check** (`ball_flight.py`):
   - Estimates ball speed from hand jc speed × 1.5 wrist/finger ratio (validated: 87.0 mph estimated vs 85.3 mph actual, 2% error)
   - Checks release feasibility: height (1.0–2.2m) and forward direction (≥80% of speed toward plate)
   - Computes a scalar RL reward: `speed_mph + 10 × strike_quality`

5. **IK Solver** (`site_ik.py`): Damped least-squares solver using `mj_jacSite` — tracks 10 joint-center sites (hands, elbows, shoulders, knees, ankles for both sides) while regularizing toward a neutral pose.

## Quick Start

### Prerequisites

- Python 3.10+
- macOS: `mjpython` (bundled with the `mujoco` pip package) is required for the viewer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install mujoco numpy
```

### 1. Build the model

```bash
python -m pitching_mechanics.build_pitcher_fullbody \
  --root . --session-pitch 1623_3 --event fp_10_time
```

### 2. Replay with torque logging

```bash
mjpython -m pitching_mechanics.replay_pitcher_fullbody \
  --root . --session-pitch 1623_3 \
  --realtime --loop --sleep-mult 3.0 \
  --log-torques pitching_mechanics/logs/torques_1623_3.csv
```

The replay window defaults to `fp_10_time` → `MIR_time + 0.25s` (~0.44s of the delivery). Torque summary, ball release speed, and strike check are printed to the console. A per-frame torque CSV is saved.

### 3. Replay only (no torque computation)

```bash
mjpython -m pitching_mechanics.replay_pitcher_fullbody \
  --root . --session-pitch 1623_3 \
  --realtime --loop --sleep-mult 3.0
```

## Pitch Selection

The default pitch (`1623_3`) was chosen as a median-velocity, right-handed, college-level pitcher from the OBP dataset. To use a different pitch, pass a different `--session-pitch` value (format: `<session>_<pitch_number>`, found in `openbiomechanics/baseball_pitching/data/metadata.csv`).

## Data Source

Joint-center landmarks and event timestamps come from the [Driveline OpenBiomechanics Project](https://github.com/drivelinebaseball/openbiomechanics) (`openbiomechanics/baseball_pitching/data/full_sig/`). The OBP global coordinate system is X = toward home plate, Y = pitcher's left, Z = up — which matches MuJoCo's default world frame, so no coordinate rotation is needed.

## RL Roadmap

See `TO_DO.txt` for the full roadmap. Current status:

| Step | Description | Status |
|------|-------------|--------|
| 1 | Inverse-dynamics torque baseline | Done |
| 2 | Ball release + strike zone | Done |
| 3 | Gym environment wrapper | Next |
| 4 | PPO training loop | Planned |
| 5 | Polish (wider window, contacts) | Planned |
