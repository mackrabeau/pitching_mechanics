# PitchLens — Ideas Backlog

Running list of feature ideas and future directions discussed during development.
Not prioritized — everything here is a candidate for a future sprint.

---

## Analytics & Modeling

### More training data
- 403 pitches (~100 athletes) is the current dataset. College dominates (314/411).
- More data would most improve: generalization across playing levels, reliability
  of lower-ranked SHAP features, strength model (currently needs playing_level_enc
  as a crutch due to HS/College/Pro population split).
- Potential sources: collect own team data once CV layer is built, partner with
  other programs, augment with synthetic data via physics simulation.

### Torque time-series visualization
- Already have torques_1623_3.csv (440 frames x 21 joints) from MuJoCo inverse dynamics.
- Plot per-joint torque across the delivery window with event markers (PKH, FP, MER, BR, MIR).
- Clinically meaningful: show exactly when peak elbow varus torque occurs relative to BR.
- Launchpad does not show this publicly — genuine differentiator.

### Kinetic chain efficiency view
- poi_metrics has full energy flow columns: generation, transfer, absorption per segment.
- Visualize as stacked bar: energy contribution from rear leg, lead leg, pelvis,
  trunk, shoulder, elbow — compared to OBP cohort average.
- Shows where in the chain energy is leaking vs. being efficiently transferred.

### Hip-to-shoulder timing callout
- timing_peak_torso_to_peak_pelvis_rot_velo is one of the most coaching-relevant
  metrics in the dataset but currently buried in the improvements table.
- Deserves its own dedicated display with distribution plot showing where the
  pitcher falls and what the elite window looks like.
- Frame as: "your hip-to-shoulder separation timing is Xms — elite range is Y-Zms."
- NOTE: correlation with velo is r=0.036 (essentially zero) and correlation with
  elbow stress is r=-0.008 — not worth featuring as a standalone metric.
  Superseded by Section 5 (separation + torso velo + stress tradeoff).

### Video frame annotation (Phase 3 — after CV layer)
- Once CV pipeline is running, every biomechanical event has a timestamp
  that maps back to a specific frame in the original video.
- Show the corresponding video frame alongside each diagnostic finding:
    - Frame at max hip-shoulder separation
    - Frame at max elbow varus moment (peak UCL stress)
    - Frame at foot plant, MER, ball release
    - Frame where energy leak is worst (e.g. lead knee absorption peak)
- This turns abstract numbers into something a pitcher can actually see —
  "here is the exact moment your lead knee is leaking energy" is far more
  actionable than a bar chart value.
- Implementation: store frame index alongside each POI metric during CV
  extraction, then display with st.image() or a simple video scrubber.
- Could also do side-by-side: pitcher's frame at event X vs closest OBP
  comp's frame at the same event — visual mechanical comparison.
- This is the feature that makes the CV layer genuinely compelling to
  coaches and players vs just producing numbers.
- Current injury flags use fixed Nm thresholds (elbow >80 Nm = high risk).
  These are conservative and most elite pitchers flag as high risk, which
  reduces clinical usefulness.
- Better approach: build a percentile-based risk model — flag when a pitcher
  is in the top 10% of elbow stress FOR THEIR VELOCITY TIER. A pitcher
  throwing 90 mph with 120 Nm is less concerning than one throwing 78 mph
  with 120 Nm — the stress is disproportionate to their output.
- Could also model stress efficiency: elbow_varus_moment / pitch_speed_mph.
  Lower ratio = getting more velo per unit of arm stress = more efficient.
- The separation/stress tradeoff (r=0.162 between hip-shoulder sep and elbow
  varus) suggests that pitchers in Q3 of separation (not Q4) have the best
  velo-to-stress profile. Worth investigating as a coaching target range
  rather than "maximize separation."

### Session-level averaging
- Right now each pitch is treated independently. Averaging across all pitches in
  a session would give more stable estimates and reduce noise from individual throws.
- Especially important for injury risk flags — one high-stress pitch matters less
  than consistently high stress across a session.

---

## Dashboard & UI

### LLM-powered drill recommendations (RAG pipeline)
- Take top 3 areas for improvement from the scorer.
- Map them to specific interventions via a RAG pipeline over:
    - Driveline's public research blog and OBP documentation
    - ASMI (American Sports Medicine Institute) publications
    - Published pitching mechanics literature
- Output: "Your rotation score is 11th percentile. Based on current research,
  focus on: [specific drill], [specific warm-up], [specific strength exercise]."
- Model: Claude API or GPT-4 with a curated knowledge base as context.
- This is the bridge between "here's what's wrong" and "here's what to do about it."
- Discussed but NOT yet implemented — save for after CV layer.

### Manual input mode for real pitchers
- Current dashboard: OBP dataset dropdown only.
- Future: add "Manual input" tab where a pitcher or coach enters their own values.
- Required for real-world use before CV layer is complete.
- Could pre-populate with OBP averages as defaults to make it easier to use.

### Longitudinal tracking
- Same pitcher across multiple sessions — show mechanical drift over time.
- Requires a session database (could be simple CSV or SQLite).
- Key question: are mechanics improving after a training intervention?

### Fatigue detection
- Mechanical signature of a tired arm across a start or bullpen session.
- Compare first 20 pitches vs. last 20 pitches of a session.
- Hypothesis: hip-shoulder separation degrades, elbow varus moment increases.

### Multi-pitch session analysis
- How mechanics change across a full bullpen (pitch 1 vs pitch 50).
- Requires collecting sequential pitch data, not just best-effort snapshots.

---

## Computer Vision (Phase 3)

### iPhone bullpen pipeline
- Core thesis target: film a bullpen from the side with an iPhone on a tripod,
  receive full PitchLens diagnostic output automatically.
- Pipeline: MediaPipe Pose → event detection (FP, MER, BR, MIR) →
  POI metric extraction → calibration model → report layer.
- Latency target: < 5 seconds on a 2022 MacBook.

### Two-camera setup for full 3D coverage
- Side view: good coverage of X (forward) and Z (vertical) planes.
  Captures arm slot, lead knee extension, torso tilt, stride length, cog velo.
- Front/back view: needed for Y-plane metrics — pelvis rotation, thorax rotation,
  hip-shoulder separation in the transverse plane.
- Two iPhones covers the full biomechanical picture.
- Identified as the right approach before trying to solve monocular depth.

### Jerin Bright / PitcherNet reference
- Bright et al. (2023, arXiv:2309.01010): synthetic data augmentation for motion
  blur in broadcast video, 54% reduction in 2D pose loss.
- PitcherNet (2024, CVPR workshop): broadcast video → pitch velo, release point,
  pitcher ID (96.82% accuracy). Owned by Baltimore Orioles, not public.
- 2025 paper: monocular video → 3D pose → injury risk prediction. Directly relevant.
- Key differentiation: Bright solves broadcast video for MLB orgs. PitchLens
  solves accessible diagnostics for college teams without MLB budgets.
- PGLM (Pose Global Lifting Model): lightweight transformer lifting pelvis-rooted
  3D pose into global space — directly relevant to our coordinate system needs.
- Code not public. Use as methodological reference, not implementation base.

### MLB pitcher database (longer-term)
- Build biomechanical profiles of named MLB pitchers from broadcast video.
- Pipeline: Baseball Savant free video clips → CV pipeline → OBP-equivalent features.
- Enables "your mechanics are most similar to [MLB pitcher]" — high coaching value.
- Requires calibration model to be validated first (CV → OBP equivalence).
- Nobody has published an open version of this. Genuine research contribution.
- Not a priority until Phase 3 CV pipeline is validated on controlled video.

---

## Data

### Hitting module
- OBP repo has full hitting dataset (baseball_hitting/) with the same structure.
- poi_metrics.csv, metadata.csv, hittrax.csv for exit velocity outcomes.
- Could build a parallel "HitLens" diagnostic for batters.
- Low priority — pitching first.

### Statcast integration
- Baseball Savant has free pitcher-level Statcast summaries via CSV download.
- Join to OBP comp pitchers where possible (lossy — no direct ID link).
- Would add: avg velo, spin rate, pitch mix, whiff%, xERA to comp cards.
- Adds "what do pitchers with your mechanics achieve in pro ball" layer.

### Strength model improvement
- Current R²=0.34 — lower than the biomechanics model.
- Would improve with: own team strength data collected alongside video sessions,
  more athletes across playing levels, wrist/forearm specific strength metrics
  not currently in hp_obp.
- The playing_level_enc feature is a crutch — real fix is more data per level.