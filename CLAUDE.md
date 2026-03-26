# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PitchLens** is a pitching mechanics analysis system built on the Driveline OpenBiomechanics Project (OBP) dataset (411 pitches). It produces a 7-section diagnostic report — inspired by Driveline's Launchpad software — analyzing velocity potential, mechanical efficiency, and injury risk.

**Three phases:**
- Phase 1 (done): Analytics pipeline (biomechanics & strength velocity models)
- Phase 2 (done): Streamlit dashboard with 7 diagnostic sections
- Phase 3 (in progress): CV pipeline to extract POI-equivalent metrics from phone video

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Common Commands

**Run the dashboard:**
```bash
streamlit run pitchlens/dashboard/app.py
```

**Smoke-test the analytics pipeline:**
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

**Run tests:**
```bash
pytest
```

## Architecture

### Data Flow

```
openbiomechanics/ (gitignored dataset)
  ├── poi_metrics.csv (411 pitches, ~75 features)
  ├── metadata.csv (pitcher demographics)
  ├── hp_obp.csv (1934 strength records)
  └── forces_moments.zip (247k frames, inverse dynamics)
        │
pitchlens/data/          ← load & join datasets
        │
pitchlens/analytics/     ← ML models & scoring
        │
pitchlens/dashboard/     ← Streamlit app (7 sections)
```

### Key Modules

**`pitchlens/data/`**
- `poi_metrics.py` — `load_poi()`, `load_hp()`, `load_combined()`. Defines all column group constants (`KINEMATIC_COLS`, `KINETIC_COLS`, `ENERGY_COLS`, `GRF_COLS`, `HP_STRENGTH_COLS`, `HP_ROM_COLS`).
- `full_sig_moments.py` — Loads per-frame joint moments from the 247k-row zip; caches in memory.
- `obp_fullsig.py` — Full-signal landmark data for the simulation layer.

**`pitchlens/analytics/`**
- `velo_model.py` — `BiomechanicsVeloModel` (78 features, GroupKFold CV grouped by session to avoid data leakage, R²=0.55) and `StrengthVeloModel` (20 features, R²=0.34). Both use `HistGradientBoostingRegressor` with SHAP explainers. Top predictor: `elbow_transfer_fp_br` (r=0.69).
- `scoring.py` — `MechanicsScorer`: empirical CDF → 0–100 percentile for 5 composite scores (Arm Action, Block, Posture, Rotation, Momentum) plus injury flags (elbow varus UCL proxy, shoulder IR rotator cuff proxy).
- `peer_match.py` — `PeerMatcher`: cosine similarity on 62 normalized features; returns `PitcherComp` objects.

**`pitchlens/dashboard/app.py`** — Main Streamlit app, 7 sections:
1. Velocity Diagnostic (actual vs. bio-expected vs. strength-expected, SHAP chart)
2. Mechanic Scores (radar chart)
3. Injury Risk (moment gauges)
4. Peer Comps (top 5 mechanically similar pitchers)
5. Kinematic Profile (hip-shoulder separation, torso rotational velocity percentiles)
6. Kinetic Chain Efficiency (energy per segment vs. cohort average)
7. Joint Moment Time-Series (per-frame moments from OBP inverse dynamics, event markers)

**`pitchlens/cv/pose_pipeline.py`** — MediaPipe pose extraction from video; Phase 3 work.

**`pitchlens/simulation/`** — MuJoCo full-body pitcher model, retained for visualization. Built from OBP landmark data; ball release validated within 2% of radar gun.

### Key Data Notes

- `openbiomechanics/` and `mujoco/` are gitignored (large external repos).
- Generated MJCF XML files go in `pitchlens/models/` (gitignored).
- `pitchlens/logs/` holds generated torque CSVs and video landmarks (gitignored).
- `StrengthVeloModel` filters to `pitch_speed_mph >= 60` to exclude non-pitchers; uses `playing_level_enc` to handle HS/College/Pro heterogeneity (plain Ridge regression fails catastrophically on this mixed population).
- Hip-to-shoulder timing (`timing_peak_torso_to_peak_pelvis_rot_velo`) has r=0.036 with velocity despite coaching prominence — energy transfer features are the dominant predictors.
