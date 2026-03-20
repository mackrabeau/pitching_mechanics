# PitchLens

Pitching mechanics analysis and velocity diagnostics built on the [Driveline OpenBiomechanics Project (OBP)](https://github.com/drivelineresearch/openbiomechanics) dataset. Inspired by Driveline's Launchpad software.

## What It Does

Given a pitcher's biomechanical data from the OBP mocap dataset (or, in Phase 3, from a phone camera), PitchLens produces:

1. **Expected velocity from mechanics** — what does your delivery pattern predict you should throw? (CV R²=0.55, RMSE=3.1 mph)
2. **Expected velocity from physical capacity** — what does your strength and athleticism predict? (CV R²=0.34, RMSE=5.2 mph)
3. **The gap** — the core Launchpad diagnostic. Positive gap = mechanics are the limiter. Negative gap = physical development is the limiter.
4. **Composite mechanic scores** — five 0–100 percentile scores (Arm Action, Block, Posture, Rotation, Momentum) vs. the OBP elite cohort.
5. **Injury risk flags** — elbow varus moment (UCL stress proxy) and shoulder IR moment (rotator cuff stress proxy) flagged against clinical thresholds.
6. **Mechanical peer comps** — the most mechanically similar pitchers in the OBP dataset via cosine similarity, and what velocity range they achieve.

## Project Structure

```
pitchlens/
├── data/
│   ├── poi_metrics.py          Load & join OBP biomechanics + metadata + hp datasets
│   └── statcast.py             (planned) Baseball Savant pitch outcome data
│
├── analytics/
│   ├── velo_model.py           Expected velo models (biomechanics + strength, XGBoost + Ridge)
│   ├── scoring.py              Composite mechanic scores, percentile ranking, injury flags
│   └── peer_match.py           Cosine-similarity peer matching
│
├── simulation/                 MuJoCo-based 3D replay (repurposed as visualizer)
│   ├── build_pitcher_fullbody.py
│   ├── replay_pitcher_fullbody.py
│   ├── ball_flight.py
│   ├── site_ik.py
│   └── trajectory.py
│
├── dashboard/
│   └── app.py                  (Phase 2) Streamlit UI
│
├── cv/
│   └── pose_pipeline.py        (Phase 3) Video → POI metrics via MediaPipe
│
├── models/                     Generated MuJoCo MJCF files
├── logs/                       Inverse-dynamics torque CSVs
└── tests/
```

## Data Sources

| Dataset | Path | Rows | Key Contents |
|---------|------|------|--------------|
| `poi_metrics.csv` | `openbiomechanics/baseball_pitching/data/poi/` | 411 | ~75 biomechanical POI variables per pitch |
| `metadata.csv` | `openbiomechanics/baseball_pitching/data/` | 411 | Age, height, weight, playing level per pitch |
| `hp_obp.csv` | `openbiomechanics/high_performance/data/` | 1934 | CMJ, IMTP, hop test, relative strength, shoulder ROM |

`poi_metrics` and `metadata` are joined on `session_pitch` and drive the biomechanics model. `hp_obp` drives the strength model. The gap between their predictions on the same pitcher is the core actionable output.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run all smoke tests in order

```bash
python -m pitchlens.data.poi_metrics .
python -m pitchlens.analytics.velo_model .
python -m pitchlens.analytics.scoring .
python -m pitchlens.analytics.peer_match .
```

Expected output from `velo_model`:
```
BiomechanicsVeloModel fitted on 403 pitches, 78 features.  CV R²=0.550  RMSE=3.06 mph
StrengthVeloModel fitted on 425 athletes, 20 features.    CV R²=0.342  RMSE=5.19 mph
```

## The Launchpad Diagnostic

The core output mirrors Driveline's most important insight — comparing what a pitcher's mechanics predict vs. what their physical capacity predicts:

```
Biomechanics model  → expected velo from mechanics = 90.2 mph
Actual velo         = 90.4 mph
Gap                 = -0.2 mph → mechanics and physical capacity are well balanced

Top mechanical drivers (SHAP values):
  ↑ elbow_transfer_fp_br                      +0.58 mph
  ↑ thorax_distal_transfer_fp_br              +0.06 mph
  ↓ max_shoulder_internal_rotational_velo     -0.05 mph
```

## Model Details

### Biomechanics Velo Model
- **Features**: 78 total — kinematics (joint angles, velocities), energy flow (segment transfer/generation/absorption), ground reaction forces, plus anthropometrics (mass, height, BMI, playing level)
- **Algorithm**: `HistGradientBoostingRegressor` (handles NaN natively, outperforms Ridge on this dataset)
- **CV**: GroupKFold 5-fold, grouped by session to prevent data leakage across pitches from the same athlete
- **Explainability**: SHAP `TreeExplainer` for per-pitcher signed feature contributions
- **Top predictors**: `elbow_transfer_fp_br`, `thorax_distal_transfer_fp_br`, `max_shoulder_internal_rotational_velo`, `shoulder_horizontal_abduction_fp`, BMI

### Strength Velo Model
- **Features**: 20 total — CMJ metrics, IMTP peak force, hop test RSI, relative strength, body weight, shoulder/thoracic ROM, playing level (ordinal encoded)
- **Algorithm**: `HistGradientBoostingRegressor`
- **CV**: 5-fold KFold (shuffled); filtered to `pitch_speed_mph >= 60` to remove non-pitcher assessments
- **Top predictors**: `peak_power_[w]_mean_cmj`, hop test RSI, shoulder IR ROM, jump height

### Mechanics Scorer
- Fits empirical CDFs from the full OBP cohort (411 pitches)
- Converts raw POI values to 0–100 percentiles vs. cohort
- Groups into 5 composite scores: Arm Action, Block, Posture, Rotation, Momentum
- Score distributions are well-calibrated (mean ~50 per category)
- Flags `elbow_varus_moment` and `shoulder_internal_rotation_moment` as injury risk signals

### Peer Matcher
- Cosine similarity on 62 normalized kinematic + energy flow features
- Returns top-N comps with similarity score, velo, playing level, and key metric values
- Includes `velo_range_for_mechanics()` and `level_breakdown()` for cohort context

## Injury Risk Flags

`elbow_varus_moment` and `shoulder_internal_rotation_moment` are the two primary clinical markers for UCL and rotator cuff stress. PitchLens flags these against conservative thresholds:

| Joint | Normal | Elevated | High Risk |
|-------|--------|----------|-----------|
| Elbow varus (UCL proxy) | < 50 Nm | 50–80 Nm | > 80 Nm |
| Shoulder IR (rotator cuff proxy) | < 40 Nm | 40–60 Nm | > 60 Nm |

Note: the OBP college cohort mean elbow varus moment is ~130–140 Nm, which exceeds these thresholds. This is consistent with the biomechanics literature — high-velocity pitching inherently produces large elbow loads. The flags are most useful for relative comparison and trend tracking across sessions, not as absolute injury predictors.

## Prior Work: MuJoCo Simulation

The `simulation/` directory contains a full-body pitcher MuJoCo model built from OBP landmark data, including a damped least-squares IK solver, inverse-dynamics torque computation via `mj_inverse`, and ball release speed estimation validated to within 2% of radar gun readings. Originally built as a reinforcement learning environment; repurposed as a 3D visualization layer for the dashboard (Phase 2).

## CV Layer (Phase 3 — Thesis Target)

The goal is a phone-deployable tool for use at practice: record a bullpen session from the side, extract POI-equivalent metrics automatically, run them through the Phase 1 models. Makes the full diagnostic accessible without a $50k motion capture setup.

The OBP repo includes camera calibration code and pose tracking starter scripts in `openbiomechanics/computer_vision/` that serve as the CV foundation.

## Dependencies

```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
scipy>=1.11
shap>=0.45
mujoco>=3.0
streamlit>=1.32      # Phase 2
plotly>=5.18         # Phase 2
mediapipe>=0.10      # Phase 3
opencv-python>=4.9   # Phase 3
```