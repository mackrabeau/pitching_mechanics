from __future__ import annotations

import argparse
from pathlib import Path

from pitching_mechanics.obp.full_sig import ObpFullSigDataset


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Inspect a single OBP processed pitch (full_sig zips): event times and basic ranges."
    )
    p.add_argument(
        "--root",
        type=Path,
        default=Path("/Users/mackrabeau/Documents/Code/pitching_mechanics"),
        help="Repo root containing openbiomechanics/",
    )
    p.add_argument("--session-pitch", required=True, help="e.g. 1623_3")
    p.add_argument(
        "--table",
        default="joint_angles",
        choices=["joint_angles", "joint_velos", "landmarks", "forces_moments", "force_plate", "energy_flow"],
        help="Which full_sig table to summarize.",
    )
    p.add_argument(
        "--cols",
        nargs="*",
        default=[],
        help="Optional list of columns to print min/max for (defaults to a small kinematics subset).",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()
    ds = ObpFullSigDataset(args.root)

    events = ds.load_events(args.session_pitch)
    print("session_pitch:", args.session_pitch)
    print("events:", events)

    if args.cols:
        cols = args.cols
    else:
        if args.table == "joint_angles":
            cols = [
                "torso_pelvis_angle_z",
                "torso_angle_z",
                "pelvis_angle_z",
                "shoulder_angle_z",
                "elbow_angle_x",
                "wrist_angle_x",
            ]
        elif args.table == "joint_velos":
            cols = [
                "torso_pelvis_velo_z",
                "torso_velo_z",
                "pelvis_velo_z",
                "shoulder_velo_z",
                "elbow_velo_x",
                "wrist_velo_x",
            ]
        elif args.table == "landmarks":
            cols = ["hand_jc_x", "hand_jc_y", "hand_jc_z"]
        else:
            cols = []

    tbl = ds.load_table(args.table, args.session_pitch, columns=cols if cols else None)
    t = tbl["time"]
    print(f"{args.table}: N={t.shape[0]}, t=[{t[0]:.4f}, {t[-1]:.4f}]")

    if cols:
        for c in cols:
            if c not in tbl:
                print(f"  {c}: (missing)")
                continue
            v = tbl[c]
            finite = v[~(v != v)]  # drop NaNs without numpy dependency on isnan semantics
            if finite.size == 0:
                print(f"  {c}: all NaN")
                continue
            print(f"  {c}: min={finite.min():.3f} max={finite.max():.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

