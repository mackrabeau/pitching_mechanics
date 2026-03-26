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
pip install -e ".[dev]"       # installs pitchlens + pytest
pip install -e ".[cv]"        # adds mediapipe + opencv (Phase 3)
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
BiomechanicsVeloModel fitted on 403 pitches, 78 features. CV R²=0.550 RMSE=3.06 mph
StrengthVeloModel fitted on 425 athletes, 20 features. CV R²=0.342 RMSE=5.19 mph
```

**Run tests:**
```bash
pytest
```

Tests use synthetic data fixtures (no OBP dataset required).

## Architecture

### Package Structure

```
pitchlens/
├── __init__.py            ← version + top-level docstring
├── config.py              ← centralized path resolution (get_project_root, get_*_dir)
├── constants.py           ← TARGET_COL, LEVEL_MAP (single source of truth)
│
├── data/                  ← load & join datasets
│   ├── __init__.py        ← re-exports: load_poi, load_hp, column group constants
│   ├── poi_metrics.py     ← load_poi(), load_hp(), column group definitions
│   ├── full_sig_moments.py← per-frame joint moments from forces_moments.zip
│   └── obp_fullsig.py    ← full-signal landmarks for simulation layer
│
├── analytics/             ← ML models & scoring
│   ├── __init__.py        ← re-exports: all model classes, scorer, matcher
│   ├── base_model.py      ← shared utilities (CV, SHAP, feature importance)
│   ├── velo_model.py      ← BiomechanicsVeloModel, StrengthVeloModel
│   ├── scoring.py         ← MechanicsScorer (percentile scoring engine)
│   ├── scoring_config.py  ← component weights & injury thresholds (tunable)
│   ├── peer_match.py      ← PeerMatcher (cosine similarity comps)
│   └── chain_models.py    ← two-chain research models (GRF + rotational)
│
├── dashboard/             ← Streamlit app
│   ├── __init__.py
│   ├── app.py             ← thin orchestrator (~130 lines)
│   ├── styles.css         ← all dashboard CSS
│   ├── charts.py          ← reusable Plotly figure builders
│   └── sections/          ← one module per diagnostic section
│       ├── __init__.py
│       ├── velocity_diagnostic.py   ← Section 1
│       ├── mechanic_scores.py       ← Section 2
│       ├── injury_risk.py           ← Section 3
│       ├── peer_comps.py            ← Section 4
│       ├── kinematic_profile.py     ← Section 5
│       ├── kinetic_chain.py         ← Section 6
│       └── joint_moments.py         ← Section 7
│
├── cv/                    ← Phase 3: video pose extraction
│   ├── __init__.py
│   └── pose_pipeline.py
│
└── simulation/            ← MuJoCo full-body pitcher model
    ├── __init__.py
    ├── ball_flight.py
    ├── build_pitcher_fullbody.py
    ├── replay_pitcher_fullbody.py
    ├── site_ik.py
    └── trajectory.py

tests/
├── conftest.py            ← synthetic data fixtures (no OBP needed)
├── test_data.py           ← column group integrity
├── test_scoring.py        ← MechanicsScorer round-trip
└── test_peer_match.py     ← PeerMatcher round-trip
```

### Data Flow

```
openbiomechanics/ (gitignored dataset)
 ├── poi_metrics.csv (411 pitches, ~75 features)
 ├── metadata.csv (pitcher demographics)
 ├── hp_obp.csv (1934 strength records)
 └── forces_moments.zip (247k frames, inverse dynamics)
 │
pitchlens/data/ ← load & join datasets
 │
pitchlens/analytics/ ← ML models & scoring
 │
pitchlens/dashboard/ ← Streamlit app (7 sections)
```

### Key Modules

**`pitchlens/config.py`** — `get_project_root()`, `get_obp_data_root()`, `get_logs_dir()`, `get_figures_dir()`, `get_models_dir()`. All path resolution goes through here. Supports `PITCHLENS_ROOT` env var override.

**`pitchlens/constants.py`** — `TARGET_COL`, `LEVEL_MAP`, `LEVEL_MAP_TITLECASE`. Single source of truth for values that were previously duplicated across modules.

**`pitchlens/analytics/base_model.py`** — `run_cross_validation()`, `compute_shap_background()`, `compute_shap_values()`, `shap_feature_importance()`, `extract_session_groups()`. Shared ML utilities used by both velo models and chain models.

**`pitchlens/analytics/scoring_config.py`** — Scoring component weights and injury thresholds, separated from the scoring engine so domain weights can be tuned without touching logic.

**`pitchlens/dashboard/charts.py`** — `make_shap_bar()`, `make_radar()`, `make_gauge()`, `make_level_bar()`, `make_kinematic_scatter()`, `make_chain_bar()`. Reusable Plotly figure builders.

**`pitchlens/data/poi_metrics.py`** — `load_poi()`, `load_hp()`, `load_combined()`. Defines all column group constants (`KINEMATIC_COLS`, `KINETIC_COLS`, `ENERGY_COLS`, `GRF_COLS`, `HP_STRENGTH_COLS`, `HP_ROM_COLS`).

**`pitchlens/analytics/velo_model.py`** — `BiomechanicsVeloModel` (78 features, GroupKFold CV grouped by session, R²=0.55) and `StrengthVeloModel` (20 features, R²=0.34). Both use `HistGradientBoostingRegressor` with SHAP explainers. Top predictor: `elbow_transfer_fp_br` (r=0.69).

**`pitchlens/analytics/scoring.py`** — `MechanicsScorer`: empirical CDF → 0–100 percentile for 5 composite scores (Arm Action, Block, Posture, Rotation, Momentum) plus injury flags.

**`pitchlens/analytics/peer_match.py`** — `PeerMatcher`: cosine similarity on 62 normalized features; returns `PitcherComp` objects.

**`pitchlens/dashboard/app.py`** — Thin orchestrator (~130 lines) that loads models, renders sidebar, and delegates each section to `pitchlens.dashboard.sections.*`.

### Key Data Notes

- `openbiomechanics/` and `mujoco/` are gitignored (large external repos).
- Generated MJCF XML files go in `pitchlens/models/` (gitignored).
- `pitchlens/logs/` holds generated torque CSVs and video landmarks (gitignored).
- `StrengthVeloModel` filters to `pitch_speed_mph >= 60` to exclude non-pitchers; uses `playing_level_enc` to handle HS/College/Pro heterogeneity.
- Hip-to-shoulder timing (`timing_peak_torso_to_peak_pelvis_rot_velo`) has r=0.036 with velocity despite coaching prominence — energy transfer features are the dominant predictors.
