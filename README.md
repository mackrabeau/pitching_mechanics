# PitchLens

Pitching mechanics analysis and velocity diagnostics built on the [Driveline OpenBiomechanics Project (OBP)](https://github.com/drivelineresearch/openbiomechanics) dataset. Inspired by Driveline's Launchpad software.

## What It Does

Given a pitcher's biomechanical data from the OBP mocap dataset (or, in Phase 3, from a phone camera), PitchLens produces a 7-section diagnostic report:

1. **Velocity diagnostic** — expected velo from mechanics (CV R²=0.55) and from physical capacity (CV R²=0.34), the gap between them, and the top SHAP-explained mechanical drivers
2. **Mechanic scores** — five 0–100 percentile scores (Arm Action, Block, Posture, Rotation, Momentum) vs. the OBP cohort, plus a ranked improvement table
3. **Injury risk** — elbow varus moment (UCL stress) and shoulder IR moment (rotator cuff stress) flagged against clinical thresholds with normalized Nm/kg values
4. **Peer comps** — top 5 most mechanically similar pitchers in the OBP dataset via cosine similarity, with velocity range and playing level breakdown
5. **Kinematic profile** — hip-shoulder separation and torso rotational velocity percentiles, stress efficiency ratio (Nm/mph), and a scatter plot of all 411 OBP pitchers colored by velocity
6. **Kinetic chain efficiency** — energy transfer, generation, and absorption per segment (rear leg, lead leg, trunk, thorax, shoulder, elbow) vs. cohort average, with per-segment efficiency table
7. **Joint moment time-series** — per-frame joint moments across the delivery window (FP to MIR) from OBP ground-truth inverse dynamics, with event markers and peak moment cards

## Project Structure

```
pitchlens/
├── data/
│   ├── poi_metrics.py          Load & join OBP poi_metrics + metadata + hp_obp datasets
│   ├── obp_fullsig.py          Load full-signal landmark data (used by simulation)
│   └── full_sig_moments.py     Load per-frame joint moments from forces_moments.zip
│
├── analytics/
│   ├── velo_model.py           Biomechanics + strength velo models (HistGBR + SHAP)
│   ├── scoring.py              Composite mechanic scores, percentile ranking, injury flags
│   └── peer_match.py           Cosine-similarity peer matching
│
├── simulation/                 MuJoCo full-body pitcher model (3D visualization layer)
│   ├── build_pitcher_fullbody.py
│   ├── replay_pitcher_fullbody.py
│   ├── ball_flight.py
│   ├── site_ik.py
│   └── trajectory.py
│
├── dashboard/
│   └── app.py                  Streamlit dashboard (7 sections)
│
├── cv/                         Phase 3 — computer vision pipeline (not yet built)
│
├── models/                     Generated MuJoCo MJCF files per session_pitch
└── logs/                       Inverse-dynamics torque CSVs
```

## Data Sources

| Dataset | Path | Rows | Key Contents |
|---------|------|------|--------------|
| `poi_metrics.csv` | `openbiomechanics/baseball_pitching/data/poi/` | 411 | ~75 biomechanical POI variables per pitch |
| `metadata.csv` | `openbiomechanics/baseball_pitching/data/` | 411 | Age, height, weight, playing level |
| `hp_obp.csv` | `openbiomechanics/high_performance/data/` | 1934 | CMJ, IMTP, hop test, relative strength, shoulder ROM |
| `forces_moments.zip` | `openbiomechanics/baseball_pitching/data/full_sig/` | 247,709 frames | Per-frame joint moments and forces, all sessions |

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the dashboard

```bash
streamlit run pitchlens/dashboard/app.py
```

First launch fits models (~20s). Subsequent pitcher selections are instant via Streamlit's cache.

### Smoke test the analytics pipeline

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

The core output mirrors Driveline's most important insight:

```
Biomechanics model  → expected velo from mechanics = 90.2 mph
Actual velo         = 90.4 mph
Gap                 = -0.2 mph → mechanics and physical capacity are well balanced

Top SHAP drivers:
  elbow_transfer_fp_br             +0.58 mph
  thorax_distal_transfer_fp_br     +0.06 mph
  max_shoulder_internal_rotational_velo  -0.05 mph
```

## Model Details

### Biomechanics Velo Model
- **Features**: 78 — kinematics, energy flow, GRF, anthropometrics (mass, height, BMI, playing level)
- **Algorithm**: `HistGradientBoostingRegressor` — handles NaN natively, outperforms Ridge on this dataset
- **CV**: GroupKFold 5-fold grouped by session (no data leakage across pitches from the same athlete)
- **Explainability**: SHAP `TreeExplainer` — signed per-feature contributions per pitcher
- **Top predictors**: `elbow_transfer_fp_br` (r=0.69), `thorax_distal_transfer_fp_br` (r=0.65), `shoulder_transfer_fp_br` (r=0.65) — energy transfer features dominate over kinematics

### Strength Velo Model
- **Features**: 20 — CMJ, IMTP, hop test RSI, relative strength, body weight, shoulder/thoracic ROM, playing level
- **Algorithm**: `HistGradientBoostingRegressor`
- **CV**: 5-fold KFold; filtered to `pitch_speed_mph >= 60` to remove non-pitcher assessments
- **Top predictors**: `peak_power_[w]_mean_cmj`, hop test RSI, shoulder IR ROM, jump height

### Mechanics Scorer
- Empirical CDFs from 411 OBP pitches → 0–100 percentile per variable
- 5 composite scores: Arm Action, Block, Posture, Rotation, Momentum (mean ~50, well-calibrated)
- Injury flags from `elbow_varus_moment` and `shoulder_internal_rotation_moment`

### Peer Matcher
- Cosine similarity on 62 normalized kinematic + energy flow features
- `find_comps()`, `velo_range_for_mechanics()`, `level_breakdown()`

### Kinetic Chain Efficiency (Section 6)
- Energy transfer columns are the strongest velo predictors in the dataset (r=0.65–0.69)
- Per-segment efficiency = transfer / (generation + transfer)
- Grouped bar chart: pitcher vs cohort average per segment

### Joint Moment Time-Series (Section 7)
- Source: `forces_moments.zip` — 247,709 rows of ground-truth OBP inverse dynamics
- 90 frames per delivery window, all joints fully populated
- Event markers: PKH, FP, MER, BR, MIR
- Multiselect joints, optional cohort average overlay, peak moment cards
- Loaded via `lru_cache` — 247k-row CSV read once per session

## Key Findings from Data Exploration

- Energy transfer features (elbow, shoulder, thorax) outperform all kinematic features as velo predictors (r=0.65–0.69 vs r=0.29–0.33 for kinematics)
- Hip-shoulder separation correlates positively with both velocity (r=0.29) and elbow stress (r=0.16) — more separation generates more power but also more arm load
- Hip-to-shoulder timing (r=0.036 with velo, r=-0.008 with elbow stress) is not a useful predictor in this dataset despite its prominence in coaching literature
- Ridge regression fails catastrophically on hp_obp (R²=-51) due to HS/College/Pro population heterogeneity — XGBoost handles it correctly

## Injury Risk Flags

| Joint | Normal | Elevated | High Risk |
|-------|--------|----------|-----------|
| Elbow varus (UCL proxy) | < 50 Nm | 50–80 Nm | > 80 Nm |
| Shoulder IR (rotator cuff proxy) | < 40 Nm | 40–60 Nm | > 60 Nm |

Note: OBP college cohort mean elbow varus is ~130–140 Nm — most elite pitchers exceed these thresholds. Use for relative comparison and trend tracking, not absolute injury prediction.

## Simulation Layer

`simulation/` contains a full-body MuJoCo pitcher model built from OBP landmark data with damped least-squares IK, inverse-dynamics torque computation via `mj_inverse`, and ball release speed estimation validated to within 2% of radar gun readings. Originally an RL training environment; retained as a 3D visualization layer.

## Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Analytics pipeline (models, scoring, peer matching) | Done |
| 2 | Streamlit dashboard (7 sections) | Done |
| 3a | MediaPipe pose extraction from iPhone video | Planned |
| 3b | 2D→3D POI metric calibration | Planned |
| 3c | End-to-end phone pipeline for practice use | Planned |
s
See `IDEAS.md` for the full feature backlog and `TO_DO.txt` for the detailed build log.

## CV Layer (Phase 3)

Film a bullpen session with an iPhone, extract POI-equivalent metrics automatically, run through the Phase 1 models. Makes the full Launchpad-style diagnostic accessible without a $50k motion capture setup.

Two-camera setup (side + front/back) covers the full biomechanical picture. Side view captures arm slot, lead knee extension, torso tilt, stride length. Front/back view captures pelvis rotation, thorax rotation, hip-shoulder separation.

Reference: Bright et al. (2023, arXiv:2309.01010) and PitcherNet (2024, CVPR) solve the broadcast video problem for MLB organizations. PitchLens solves the accessible diagnostics problem for college teams — controlled-environment bullpen video is a simpler input than broadcast.

## Dependencies

```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
scipy>=1.11
shap>=0.45
mujoco>=3.0
streamlit>=1.32
plotly>=5.18
mediapipe>=0.10      # Phase 3
opencv-python>=4.9   # Phase 3
```