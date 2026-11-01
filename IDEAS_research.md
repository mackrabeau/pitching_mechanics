# PitchLens — Research Paper Ideas

Ideas and findings relevant to the research paper. To be combined with IDEAS.md
when transitioning from thesis to publication.

---

## Literature Context: What's Known vs. What's Novel

### What the literature already knows
- Lead leg GRF strongly predicts velocity — confirmed across multiple papers
  (MacWilliams 1998, McNally 2015, Kageyama 2014). Not novel on its own.
- Sequential kinetic chain model (pelvis → trunk → shoulder → arm) well
  established since Putnam (1993). Not novel.
- Pelvis rotation velocity predicts velocity in some studies (Stodden & Fleisig
  2001) but results are inconsistent across datasets. Partially contested.
- Hip-shoulder separation predicts velocity — well established. Not novel.
- Timing of kinematic sequence matters — established (multiple ASMI papers).
  Earlier pelvis, later shoulder/elbow = more velocity. Known.
- GRF and rotational variables both predict velocity — known but always studied
  separately, never framed as parallel independent chains.
- Wang et al. (2025): pelvic rotation variability in transverse plane (r=0.78)
  is strongest predictor — novel recent finding, different framing than peak vel.
- Orishimo et al. (2023 JSCR): pelvis and trunk biomechanics and velocity —
  found trunk rotation more predictive than pelvis rotation. Partially aligns.
- Bullock et al. (2022 J Biomechanics): GBM velocity prediction, found lead leg
  GRF resultant 9.1% relative influence alongside rotational variables — treated
  as one model, not two independent chains.

### What appears to be novel in your analysis

1. **Two-chain independence demonstrated empirically on OBP:**
   GRF and rotational chain are statistically orthogonal on n=403 pitches.
   peak_grf_z vs torso rotation r=-0.031, vs shoulder IR r=-0.041.
   Literature treats them as parts of one sequential chain; your data shows
   they're parallel independent pathways that additively contribute to velocity.
   This is a conceptual reframing backed by data.

2. **Pelvis rotation uncorrelated with velocity (r=0.033) on OBP:**
   Contradicts Stodden & Fleisig (2001). Possible explanations worth testing:
   (a) their sample was elite-only; OBP is mixed levels causing confounding,
   (b) pelvis velocity matters only above a threshold (nonlinear relationship),
   (c) pelvis velocity is fully mediated by trunk rotation (no direct effect).
   Any of these is a publishable finding that advances the literature.

3. **Novel inter-chain timing variable (timing_grf_to_torso_ms):**
   Computed from full signal force plate + joint velocity data. No published
   paper has computed or analyzed this specific variable. Key findings:
   - Mean lag = -33ms: torso fires before GRF peaks in 78% of pitches
   - Correlates with GRF magnitude (r=0.411) but NOT rotational chain
     (torso rotation r=0.076, shoulder IR r=0.036)
   - Mechanistic interpretation: later torso = more time to build GRF,
     not more efficient rotational transfer
   - This is the bridge between the two chains

4. **Mechanistic explanation for hip-shoulder separation:**
   Separation (r=+0.158 with timing) delays torso rotation, allowing GRF to
   build before the rotational chain fires. This is why separation predicts
   velocity — not because it creates more rotational torque, but because it
   synchronizes the two chains. This specific mechanism has not been stated
   explicitly in the literature.

5. **ML causal chain on OBP with full signal derived features:**
   No published paper has combined POI metrics + full signal derived features
   (timing_grf_to_torso_ms, peak_grf_z) + backwards induction + SHAP on OBP.

### Key papers to cite
- Putnam (1993): sequential kinetic chain theory — foundational
- MacWilliams et al. (1998): characteristic GRF patterns in pitching
- Stodden & Fleisig (2001 J Applied Biomechanics): pelvis/trunk and velocity
- McNally et al. (2015): stride leg GRF and velocity
- Werner et al. (2008 AJSM): lead leg GRF and arm kinetics
- Orishimo et al. (2023 JSCR): pelvis and trunk biomechanics and velocity
- Bullock et al. (2022 J Biomechanics): GBM velocity prediction — closest prior
- Wang et al. (2025): pelvic rotation variability as velocity predictor
- Wasserberger et al. (2022): OBP dataset paper

---

## Timing Analysis Results (from full signal data, n=398 pitches)

```
timing_grf_to_torso_ms distribution (after outlier removal ±200ms):
  mean = -33ms, std = 59ms
  25th pct = -65ms, median = -49ms, 75th pct = -31ms
  78% of pitches: torso fires BEFORE GRF peak (negative lag)
  22% of pitches: torso fires AFTER GRF peak (positive lag)

Correlations with timing_grf_to_torso_ms:
  lead_grf_mag_max              r=+0.411 ***   ← timing predicts GRF magnitude
  peak_grf_z                    r=+0.316 ***
  hip_shoulder_separation       r=+0.158 **    ← separation delays rotation
  pitch_speed_mph               r=+0.098       (borderline p=0.051)
  torso_rotation                r=+0.076       (not significant)
  shoulder_IR                   r=+0.036       (not significant)
  pelvis_rotation               r=+0.031       (not significant)

Correlations with peak_grf_z:
  pitch_speed_mph               r=+0.403 ***
  torso_rotation                r=-0.031       (not significant)
  shoulder_IR                   r=-0.041       (not significant)
  timing_grf_to_torso_ms        r=+0.316 ***
```

Interpretation: The timing variable is a GRF chain variable, not a rotational
chain variable. Later torso rotation = more GRF built up = more velocity. The
rotational chain runs independently and contributes additively. Hip-shoulder
separation is the coordination mechanism that allows later torso timing.

---

## Core Paper: Two-Chain Causal Decomposition of Pitching Velocity

### One-sentence contribution
We demonstrate that pitching velocity is produced by two largely independent
causal chains — a ground force chain and a rotational chain — whose coordination
via timing is the primary efficiency multiplier, and we quantify the per-link
efficiency of each chain using backwards-induction ML models with SHAP attribution
on the full OBP dataset (n=403).

---

### Key empirical findings (from exploratory analysis)

**Two independent chains predict velocity:**
- Ground force chain: lead_grf_mag_max (r=+0.420) is the single strongest
  predictor of ball speed — stronger than any rotational variable.
- Rotational chain: torso rotation (r=+0.320), shoulder IR (r=+0.325),
  hip-shoulder separation (r=+0.291).
- Critically: GRF variables are nearly orthogonal to rotational variables
  (lead_grf_mag_max vs shoulder IR r=-0.006, vs torso rotation r=-0.022).
  The two chains are independent pathways to velocity.

**Pelvis rotation is a dead end (r=+0.033 with velo):**
- Pelvis rotation is strongly correlated with torso rotation (r=0.459) but
  has essentially zero direct relationship with ball speed.
- Implication: pelvis speed only matters insofar as it loads the trunk.
  A fast pelvis that doesn't decelerate properly hurts rather than helps.
  Conventional coaching focus on "hip rotation" is misplaced — the target
  should be trunk rotation, not pelvis rotation.

**Lead knee absorption is negatively correlated with velocity (r=-0.127):**
- More energy absorbed at the lead knee = less velocity. The lead leg should
  be a stiff wall, not a shock absorber. Quantifies the "lead leg block" cue.
- lead_knee_extension_from_fp_to_br (r=+0.203): how much the knee extends
  matters more than how fast it extends (angular velo r=+0.098).

**Timing mediates the two chains:**
- The existing POI timing variable (timing_peak_torso_to_peak_pelvis_rot_velo)
  only captures intra-rotational-chain timing (r=+0.036 with velo directly).
- The critical missing variable is inter-chain timing: lag between peak GRF
  and peak torso rotation. This is a research gap in the OBP literature.
- Theory: optimal velocity requires the rotational chain to fire when GRF is
  at peak output — too early (before lead foot is planted) wastes GRF;
  too late (after GRF has decayed) loses the bracing effect.
- This explains why hip-shoulder separation matters: it delays the rotational
  chain firing until GRF is established.

**Chain structure (data-driven):**
```
GROUND FORCE CHAIN:
stride_length / max_cog_velo_x
    → lead_grf_mag_max          (r=+0.420 with velo)
        → lead_hip_absorption   (r=+0.190)
        → lead_knee_extension   (r=+0.203)

ROTATIONAL CHAIN:
max_rotation_hip_shoulder_separation (r=+0.291)
    → max_torso_rotational_velo      (r=+0.320)
        → max_shoulder_internal_rotational_velo (r=+0.325)
            → max_elbow_extension_velo (r=+0.563 inter-link, r=+0.125 velo)
                → pitch_speed_mph

COORDINATION:
timing_grf_peak_to_torso_peak_ms [TO BE COMPUTED from full signal]
    → mediates efficiency of both chains simultaneously
```

---

### Novel feature: inter-chain timing variable

**The gap:** OBP POI metrics have only one timing variable
(timing_peak_torso_to_peak_pelvis_rot_velo), which captures intra-rotational
timing only. No existing variable captures when the rotational chain fires
relative to GRF peak — the inter-chain coordination.

**The computation:** from the full signal data (forces_moments.zip, 247k frames):
1. For each pitch, load per-frame lead GRF (z-axis vertical component)
2. Find frame of peak lead GRF → t_grf_peak
3. Find frame of peak torso rotational velocity → t_torso_peak
4. Compute lag: timing_grf_peak_to_torso_peak_ms = (t_torso_peak - t_grf_peak) * (1000/fps)
5. Positive = torso fires after GRF peak (correct sequencing)
   Negative = torso fires before GRF peak (early rotation, energy leak)

**Why this is a contribution:** this variable has not been computed or analyzed
in the OBP literature. It directly tests the two-chain coordination hypothesis
and provides a mechanistic explanation for why hip-shoulder separation timing
predicts velocity.

---

### Proposed paper structure

**Title:** "Two Independent Causal Chains in Baseball Pitching Velocity:
Ground Force and Rotational Mechanics as Parallel Pathways"

*(Alternative: "Sequential Causal Decomposition of Pitching Velocity Using
Open Biomechanics Data and Gradient Boosting")*

1. **Introduction**
   - Velocity as primary performance metric; current diagnostic tools stop at
     correlations; no published model decomposes the full causal chain
   - Propose two-chain framework: GRF chain + rotational chain, coordinated
     by inter-chain timing

2. **Related Work**
   - Putnam (1993): sequential kinetic chain theory
   - Hirashima et al. (2008): per-joint inverse dynamics contribution
   - Fleisig / ASMI: isolated correlation studies
   - Werner et al. (2008): lead leg GRF and its relationship to arm stress
   - 2022 gradient boosting velocity paper (Journal of Biomechanics): closest
     prior work — replicated and extended here
   - OBP dataset paper (Wasserberger et al., 2022)

3. **Data**
   - OBP dataset: 403 pitches, ~100 athletes, college-dominant
   - POI metrics + full signal data
   - Novel feature engineering: inter-chain timing from full signal

4. **Methods**
   - Exploratory correlation analysis (chain structure discovery)
   - Novel timing feature computation from full signal
   - Backwards induction chain models (HistGradientBoosting + SHAP)
   - Residual chaining to mitigate error compounding
   - DAG specification from biomechanics theory + causal inference
     (backdoor adjustment via dowhy/causalml) as methodological comparison

5. **Results**
   - Two-chain structure validation (correlation + independence)
   - Inter-chain timing variable: distribution, correlation with velo,
     relationship to hip-shoulder separation
   - Per-link model performance (R², RMSE, SHAP)
   - Residual analysis: which links lose most energy across the cohort
   - DAG causal estimates vs correlation baselines
   - Case study: trace a specific pitcher's chain breakdown

6. **Discussion**
   - Pelvis rotation finding: challenges conventional "hip-first" coaching
   - Lead leg finding: quantifies the block as energy conservation not just bracing
   - Timing as coordination variable: bridges GRF and rotational literature
   - Limitations: college-dominant sample, no LHP analysis, POI vs full signal
   - Future work: CV pipeline validation on new data; LHP chain comparison

7. **Conclusion**

---

### Methodological options for error compounding

The backwards induction architecture risks error compounding across sequential
models and ignores simultaneous inter-segment interactions. Mitigations:

1. **RESIDUAL CHAINING (primary):** at each step model the residual (deviation
   from expected given upstream inputs) rather than raw value. Decouples models,
   reduces compounding, residuals are coaching-interpretable.

2. **MULTI-OUTPUT MODEL (comparison):** one model predicting all intermediate
   targets simultaneously. Handles simultaneity directly. SHAP still works.
   Include as robustness check.

3. **DAG + CAUSAL INFERENCE (methodological contribution):** specify causal
   graph from biomechanics theory, use backdoor adjustment for per-link causal
   effects. Libraries: dowhy, causalml. Separates direct from indirect effects.

4. **CONFIDENCE INTERVALS:** bootstrapped prediction intervals propagated
   down the chain. Makes uncertainty visible, low implementation cost.

5. **STRUCTURAL EQUATION MODELING (comparison):** semopy library. Handles
   simultaneous relationships natively. Include as traditional statistics baseline.

---

### What's already done vs. needs building

| Component | Status |
|---|---|
| Baseline velo model (BiomechanicsVeloModel) | ✅ Done (R²=0.55, RMSE=3.06) |
| OBP data pipeline | ✅ Done |
| SHAP framework | ✅ Done |
| Exploratory correlation analysis | ✅ Done |
| Two-chain structure identified | ✅ Done |
| Inter-chain timing feature from full signal | ❌ In progress |
| Backwards induction chain models | ❌ To build |
| Residual chaining | ❌ To build |
| DAG + causal inference | ❌ To build |
| Multi-output model | ❌ To build |
| Figures / visualizations | ❌ To build |
| Writing | ❌ To build |

---

### Thesis angle
Full arc: OBP exploratory analysis → two-chain hypothesis → novel timing feature
from full signal → causal chain models → CV pipeline validation on own team data.
Each component is a complete contribution. Realistic for undergrad thesis with
publication potential.