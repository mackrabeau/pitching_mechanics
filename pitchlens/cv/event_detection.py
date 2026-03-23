"""Detect pitch delivery events from MediaPipe landmark trajectories.

Events detected (mirrors OBP force_plate.csv naming):
    fp_10_time  — foot plant: lead ankle velocity drops near zero
    MER_time    — max external rotation: peak right elbow-shoulder distance
    BR_time     — ball release: peak right wrist velocity
    MIR_time    — max internal rotation: wrist velocity local min after BR

All times are in seconds from video start.

Usage:
    from pitchlens.cv.event_detection import detect_events
    events = detect_events("pitchlens/logs/landmarks_4609.csv")
    print(events)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelmin


# ── Config ────────────────────────────────────────────────────────────────────

SMOOTH_SIGMA = 3          # Gaussian smoothing sigma in frames (~100ms at 30fps)
MIR_SEARCH_WINDOW_S = 0.5 # seconds after BR to search for MIR


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class PitchEvents:
    fp_10_time: float
    MER_time: float
    BR_time: float
    MIR_time: float

    def to_dict(self) -> dict[str, float]:
        return {
            "fp_10_time": self.fp_10_time,
            "MER_time": self.MER_time,
            "BR_time": self.BR_time,
            "MIR_time": self.MIR_time,
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _interp_nan(arr: np.ndarray) -> np.ndarray:
    """Linear interpolation over NaN gaps."""
    arr = arr.copy().astype(float)
    nans = np.isnan(arr)
    if nans.any():
        idx = np.arange(len(arr))
        arr[nans] = np.interp(idx[nans], idx[~nans], arr[~nans])
    return arr


def _smooth(arr: np.ndarray, sigma: float = SMOOTH_SIGMA) -> np.ndarray:
    return gaussian_filter1d(_interp_nan(arr), sigma=sigma)


def _speed_2d(x: np.ndarray, y: np.ndarray, times: np.ndarray) -> np.ndarray:
    vx = np.gradient(x, times)
    vy = np.gradient(y, times)
    return np.sqrt(vx**2 + vy**2)


def _distance(x1, y1, x2, y2) -> np.ndarray:
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


# ── Core detection ────────────────────────────────────────────────────────────

def detect_events(
    landmarks_csv: str | Path,
    *,
    smooth_sigma: float = SMOOTH_SIGMA,
    verbose: bool = True,
) -> PitchEvents:
    """Detect pitch events from a MediaPipe landmark CSV."""
    df = pd.read_csv(landmarks_csv)
    times = df["time_s"].values
    fps = 1.0 / np.median(np.diff(times))

    # ── Extract, interpolate NaNs, and smooth ─────────────────────────────────
    rw_x = _smooth(df["RIGHT_WRIST_x"].values, smooth_sigma)
    rw_y = _smooth(df["RIGHT_WRIST_y"].values, smooth_sigma)
    re_x = _smooth(df["RIGHT_ELBOW_x"].values, smooth_sigma)
    re_y = _smooth(df["RIGHT_ELBOW_y"].values, smooth_sigma)
    rs_x = _smooth(df["RIGHT_SHOULDER_x"].values, smooth_sigma)
    rs_y = _smooth(df["RIGHT_SHOULDER_y"].values, smooth_sigma)
    la_x = _smooth(df["LEFT_ANKLE_x"].values, smooth_sigma)
    la_y = _smooth(df["LEFT_ANKLE_y"].values, smooth_sigma)

    # ── 1. Ball Release (BR) — peak right wrist speed ─────────────────────────
    wrist_speed = _smooth(_speed_2d(rw_x, rw_y, times), smooth_sigma)
    br_frame = int(np.argmax(wrist_speed))
    br_time = float(times[br_frame])

    # ── 2. MER — onset of rapid wrist acceleration before BR ─────────────────
    # From side view, elbow-shoulder distance doesn't reliably capture MER.
    # Instead: MER is the frame where wrist speed crosses 20% of its peak value
    # on the way up to BR — i.e. the start of the arm acceleration phase.
    wrist_peak = wrist_speed[br_frame]
    threshold_mer = 0.20 * wrist_peak
    # Search only between halfway point and BR
    search_start_mer = br_frame // 2
    rising = np.where(wrist_speed[search_start_mer:br_frame] > threshold_mer)[0]
    if len(rising) > 0:
        mer_frame = int(search_start_mer + rising[0])
    else:
        mer_frame = br_frame - int(0.15 * fps)  # fallback: 150ms before BR
    mer_time = float(times[mer_frame])

    # ── 3. Foot Plant (FP) — last ankle speed minimum 0.8s–0.1s before BR ────
    ankle_speed = _smooth(_speed_2d(la_x, la_y, times), smooth_sigma)
    fp_search_start = max(0, br_frame - int(0.8 * fps))
    fp_search_end = max(0, br_frame - int(0.1 * fps))
    ankle_mins = argrelmin(ankle_speed[fp_search_start:fp_search_end], order=3)[0]

    if len(ankle_mins) > 0:
        fp_frame = int(fp_search_start + ankle_mins[-1])
    else:
        # fallback: frame of minimum ankle speed in window
        fp_frame = int(fp_search_start + np.argmin(ankle_speed[fp_search_start:fp_search_end]))
    fp_time = float(times[fp_frame])
    
    # ── 4. MIR — first wrist speed local min after BR ─────────────────────────
    search_end = min(br_frame + int(MIR_SEARCH_WINDOW_S * fps), len(times) - 1)
    mir_segment = wrist_speed[br_frame + 1 : search_end]  # skip BR frame itself

    if len(mir_segment) == 0:
        mir_frame = min(br_frame + 5, len(times) - 1)
    else:
        mir_mins = argrelmin(mir_segment, order=3)[0]
        if len(mir_mins) > 0:
            mir_frame = int(br_frame + 1 + mir_mins[0])
        else:
            # No local min found — use the frame of minimum speed in the window
            # but enforce at least 3 frames after BR
            offset = max(3, int(np.argmin(mir_segment)))
            mir_frame = int(br_frame + 1 + offset)
    mir_time = float(times[mir_frame])

    # ── Sanity check ──────────────────────────────────────────────────────────
    ordered = fp_time < mer_time < br_time < mir_time
    if verbose:
        print(f"\nDetected pitch events (fps={fps:.1f}):")
        print(f"  Foot plant (FP):          {fp_time:.3f}s  (frame {fp_frame})")
        print(f"  Max ext. rotation (MER):  {mer_time:.3f}s  (frame {mer_frame})")
        print(f"  Ball release (BR):        {br_time:.3f}s  (frame {br_frame})")
        print(f"  Max int. rotation (MIR):  {mir_time:.3f}s  (frame {mir_frame})")
        print(f"  FP→BR window:             {br_time - fp_time:.3f}s")
        print(f"  Peak wrist speed:         {wrist_speed[br_frame]:.4f} norm/s")
        if not ordered:
            print(f"  WARNING: Event ordering violated (FP<MER<BR<MIR expected)")
            print(f"  Check that pitcher faces RIGHT on screen (RHP side view).")

    return PitchEvents(
        fp_10_time=fp_time,
        MER_time=mer_time,
        BR_time=br_time,
        MIR_time=mir_time,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("landmarks_csv", type=Path)
    p.add_argument("--sigma", type=float, default=SMOOTH_SIGMA,
                   help="Gaussian smoothing sigma in frames (default: 3)")
    args = p.parse_args()

    detect_events(args.landmarks_csv, smooth_sigma=args.sigma, verbose=True)