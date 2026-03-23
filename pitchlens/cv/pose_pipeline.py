"""MediaPipe pose extraction pipeline for pitching video.

Processes a video file frame by frame, extracts 33 body landmarks
per frame, and returns a structured DataFrame ready for downstream
POI metric extraction.

Usage:
    from pitchlens.cv.pose_pipeline import extract_pose, visualize_pose

    df = extract_pose('pitchlens/videos/IMG_4609.MOV')
    print(df.shape)   # (n_frames, 33*4 + 2)  x/y/z/vis per landmark + frame + time
"""
from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# ── MediaPipe landmark names we care about for pitching ──────────────────

LANDMARK_NAMES = [
    "NOSE",
    "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_ELBOW",    "RIGHT_ELBOW",
    "LEFT_WRIST",    "RIGHT_WRIST",
    "LEFT_HIP",      "RIGHT_HIP",
    "LEFT_KNEE",     "RIGHT_KNEE",
    "LEFT_ANKLE",    "RIGHT_ANKLE",
    "LEFT_HEEL",     "RIGHT_HEEL",
    "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
]

# Map from MediaPipe landmark name to index
_MP_POSE = mp.solutions.pose
LANDMARK_IDX = {name: _MP_POSE.PoseLandmark[name].value for name in LANDMARK_NAMES}

# Minimum visibility threshold — landmarks below this are set to NaN
VIS_THRESHOLD = 0.3


# ── Core extraction function ──────────────────────────────────────────────

def extract_pose(
    video_path: str | Path,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    visibility_threshold: float = VIS_THRESHOLD,
) -> pd.DataFrame:
    """Extract pose landmarks from every frame of a pitching video.

    Args:
        video_path:               Path to video file (.MOV, .mp4, etc.)
        min_detection_confidence: MediaPipe detection confidence threshold.
        min_tracking_confidence:  MediaPipe tracking confidence threshold.
        visibility_threshold:     Landmarks below this visibility are NaN.

    Returns:
        DataFrame with columns:
            frame       — frame index (0-based)
            time_s      — timestamp in seconds
            {LANDMARK}_x, {LANDMARK}_y, {LANDMARK}_z, {LANDMARK}_vis
            for each landmark in LANDMARK_NAMES

        Coordinates are normalized 0-1 (x=horizontal, y=vertical, z=depth).
        x=0 is left edge, y=0 is top edge of frame.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {video_path.name}")
    print(f"  FPS: {fps:.1f}  Frames: {total_frames}  "
          f"Duration: {total_frames/fps:.2f}s")

    pose = _MP_POSE.Pose(
        static_image_mode=False,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    records = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        row: dict[str, float | int] = {
            "frame":  frame_idx,
            "time_s": frame_idx / fps,
        }

        if result.pose_landmarks:
            lms = result.pose_landmarks.landmark
            for name in LANDMARK_NAMES:
                idx = LANDMARK_IDX[name]
                lm = lms[idx]
                vis = float(lm.visibility)
                if vis >= visibility_threshold:
                    row[f"{name}_x"]   = float(lm.x)
                    row[f"{name}_y"]   = float(lm.y)
                    row[f"{name}_z"]   = float(lm.z)
                    row[f"{name}_vis"] = vis
                else:
                    row[f"{name}_x"]   = np.nan
                    row[f"{name}_y"]   = np.nan
                    row[f"{name}_z"]   = np.nan
                    row[f"{name}_vis"] = vis
        else:
            # No detection this frame — all NaN
            for name in LANDMARK_NAMES:
                row[f"{name}_x"]   = np.nan
                row[f"{name}_y"]   = np.nan
                row[f"{name}_z"]   = np.nan
                row[f"{name}_vis"] = 0.0

        records.append(row)
        frame_idx += 1

        if frame_idx % 30 == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames...", end="\r")

    cap.release()
    pose.close()
    print(f"  Done. {frame_idx} frames processed.          ")

    df = pd.DataFrame(records)
    return df


# ── Visualization helper ──────────────────────────────────────────────────

def visualize_pose(
    video_path: str | Path,
    output_path: str | Path | None = None,
    max_frames: int | None = None,
) -> None:
    """Write a new video with MediaPipe pose overlay drawn on each frame.

    Args:
        video_path:   Input video path.
        output_path:  Output video path. Defaults to input_pose_overlay.mp4.
        max_frames:   Stop after this many frames (None = full video).
    """
    video_path = Path(video_path)
    if output_path is None:
        output_path = video_path.parent / f"{video_path.stem}_pose_overlay.mp4"
    output_path = Path(output_path)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
    pose = _MP_POSE.Pose(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and frame_idx >= max_frames:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if result.pose_landmarks:
            mp_draw.draw_landmarks(
                frame,
                result.pose_landmarks,
                _MP_POSE.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
            )

        # Frame number overlay
        cv2.putText(
            frame, f"frame {frame_idx}  t={frame_idx/fps:.2f}s",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
        )

        out.write(frame)
        frame_idx += 1

        if frame_idx % 30 == 0:
            print(f"  Writing frame {frame_idx}...", end="\r")

    cap.release()
    out.release()
    pose.close()
    print(f"  Overlay video saved: {output_path}")


# ── Summary stats ─────────────────────────────────────────────────────────

def landmark_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Report visibility and NaN rate per landmark across all frames."""
    rows = []
    for name in LANDMARK_NAMES:
        vis_col = f"{name}_vis"
        x_col   = f"{name}_x"
        if vis_col not in df.columns:
            continue
        rows.append({
            "landmark":    name,
            "mean_vis":    df[vis_col].mean(),
            "pct_detected": df[x_col].notna().mean() * 100,
            "n_frames":    len(df),
        })
    return pd.DataFrame(rows).sort_values("pct_detected", ascending=False)


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, sys
    sys.path.insert(0, str(Path(__file__).parents[2]))

    p = argparse.ArgumentParser(description="Extract MediaPipe pose from pitching video.")
    p.add_argument("video", type=Path, help="Path to video file")
    p.add_argument("--overlay", action="store_true",
                   help="Write pose overlay video")
    p.add_argument("--out-csv", type=Path, default=None,
                   help="Save landmark DataFrame to CSV")
    p.add_argument("--max-frames", type=int, default=None)
    args = p.parse_args()

    df = extract_pose(args.video, visibility_threshold=VIS_THRESHOLD)

    print()
    print("Landmark quality report:")
    print(landmark_quality_report(df).to_string(index=False))

    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
        print(f"Saved: {args.out_csv}")

    if args.overlay:
        visualize_pose(args.video, max_frames=args.max_frames)
