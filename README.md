# pitching_mechanics

Physics-tracked pitch delivery replay using [Driveline OpenBiomechanics Project (OBP)](https://github.com/drivelinebaseball/openbiomechanics) landmark data and [MuJoCo](https://mujoco.org/).

## Overview

This project loads motion-capture joint-center landmarks from the OBP baseball pitching dataset, builds a pitcher-specific MuJoCo model scaled to the athlete's actual segment lengths, and replays the full delivery (foot plant → ball release → follow-through) as either a kinematic playback or a physics-tracked simulation with position actuators.

The goal is to produce a meaningful force/torque baseline that can later serve as the foundation for reinforcement-learning-based pitch optimization (Plan A).

## Project Structure

```
pitching_mechanics/
├── __init__.py
├── obp_fullsig.py               # Load OBP full-signal landmarks + events
├── site_ik.py                    # Damped least-squares IK solver (mj_jacSite)
├── build_pitcher_fullbody.py     # Generate full-body MJCF from OBP landmarks
├── replay_pitcher_fullbody.py    # Full-body replay (kinematic or physics-tracked)
└── models/
    └── pitcher_fullbody_*.xml    # Generated per-pitch MJCF models
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

3. **Replay** (`replay_pitcher_fullbody.py`): For each simulation timestep:
   - Sets the pelvis position and orientation from OBP hip landmarks.
   - Solves site-based IK (`site_ik.py`) over 21 hinge DOFs to match 10 tracked joint centers (elbows, hands, shoulders, knees, ankles).
   - In **kinematic** mode: sets `qpos` directly (validates geometry).
   - In **physics** mode: feeds IK joint angles to position actuators and steps the simulation (produces forces/torques).

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

### 2. Replay (kinematic)

```bash
mjpython -m pitching_mechanics.replay_pitcher_fullbody \
  --root . --session-pitch 1623_3 \
  --kinematic --realtime --loop --sleep-mult 3.0
```

### 3. Replay (physics-tracked)

```bash
mjpython -m pitching_mechanics.replay_pitcher_fullbody \
  --root . --session-pitch 1623_3 \
  --realtime --loop --sleep-mult 2.0
```

## Pitch Selection

The default pitch (`1623_3`) was chosen as a median-velocity, right-handed, college-level pitcher from the OBP dataset. To use a different pitch, pass a different `--session-pitch` value (format: `<session>_<pitch_number>`, found in `openbiomechanics/baseball_pitching/data/metadata.csv`).

## Data Source

Joint-center landmarks and event timestamps come from the [Driveline OpenBiomechanics Project](https://github.com/drivelinebaseball/openbiomechanics) (`openbiomechanics/baseball_pitching/data/full_sig/`). The OBP global coordinate system is X = toward home plate, Y = pitcher's left, Z = up — which matches MuJoCo's default world frame, so no coordinate rotation is needed.
