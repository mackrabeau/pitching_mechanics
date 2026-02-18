from __future__ import annotations

import csv
import dataclasses
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np


_DEFAULT_EVENT_TIME_COLS: Tuple[str, ...] = (
    "pkh_time",
    "fp_10_time",
    "fp_100_time",
    "MER_time",
    "BR_time",
    "MIR_time",
)


@dataclasses.dataclass(frozen=True)
class ObpFullSigPaths:
    """Locations of OBP processed full-signal tables (zipped CSVs)."""

    root: Path

    @property
    def full_sig_dir(self) -> Path:
        return (
            self.root
            / "openbiomechanics"
            / "baseball_pitching"
            / "data"
            / "full_sig"
        )

    def zip_path(self, table: str) -> Path:
        # OBP naming in this repo:
        # - joint_angles.zip, joint_velos.zip, landmarks.zip
        # - forces_moments.zip, force_plate.zip, energy_flow.zip
        return self.full_sig_dir / f"{table}.zip"


class ObpFullSigDataset:
    """Stream-reader for OBP processed full-signal zipped CSV tables.

    This is intentionally lightweight (csv+zipfile) so it works without pandas
    and without loading the full dataset into memory.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        event_time_cols: Sequence[str] = _DEFAULT_EVENT_TIME_COLS,
    ):
        self._paths = ObpFullSigPaths(Path(root))
        self._event_time_cols = tuple(event_time_cols)

    @property
    def paths(self) -> ObpFullSigPaths:
        return self._paths

    @property
    def event_time_cols(self) -> Tuple[str, ...]:
        return self._event_time_cols

    def _open_single_csv_member(self, zip_path: Path) -> tuple[zipfile.ZipFile, str]:
        zf = zipfile.ZipFile(zip_path)
        members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if len(members) != 1:
            zf.close()
            raise ValueError(f"Expected 1 CSV in {zip_path}, found {len(members)}: {members[:10]}")
        return zf, members[0]

    def header(self, table: str) -> List[str]:
        """Return the CSV header for a table (list of column names)."""
        zp = self._paths.zip_path(table)
        zf, member = self._open_single_csv_member(zp)
        try:
            with zf.open(member) as f:
                line = f.readline().decode("utf-8", "replace").rstrip("\n")
            return next(csv.reader([line]))
        finally:
            zf.close()

    def iter_rows(self, table: str, session_pitch: str) -> Iterator[Dict[str, str]]:
        """Yield dict rows for a specific `session_pitch`."""
        zp = self._paths.zip_path(table)
        zf, member = self._open_single_csv_member(zp)
        try:
            with zf.open(member) as f:
                # TextIOWrapper can be slow; decode line-by-line ourselves.
                reader = csv.DictReader(
                    (line.decode("utf-8", "replace") for line in f),
                )
                for row in reader:
                    if row.get("session_pitch") == session_pitch:
                        yield row
        finally:
            zf.close()

    def load_table(
        self,
        table: str,
        session_pitch: str,
        *,
        columns: Optional[Sequence[str]] = None,
        include_events: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Load a single table for one pitch into columnar numpy arrays.

        Returns a dict mapping column name -> numpy array.
        Always includes `time` if present.
        If include_events=True, returns event times in scalar arrays of shape (N,)
        (duplicated per row, matching the source format), and additionally returns
        a separate entry `_events` with the first non-empty values found.
        """
        # Discover header to validate requested columns.
        hdr = self.header(table)
        if "time" not in hdr:
            raise ValueError(f"{table} missing required 'time' column")

        want_cols: List[str]
        if columns is None:
            want_cols = list(hdr)
        else:
            want_cols = ["session_pitch", "time"] + [c for c in columns if c not in ("session_pitch", "time")]
            missing = [c for c in want_cols if c not in hdr]
            if missing:
                raise ValueError(f"{table} missing requested columns: {missing}")

        # Ensure event columns included if requested and present in this table.
        event_cols = [c for c in self._event_time_cols if c in hdr]
        if include_events:
            for c in event_cols:
                if c not in want_cols:
                    want_cols.append(c)

        # Allocate Python lists first (unknown row count).
        out_lists: Dict[str, List[float]] = {c: [] for c in want_cols if c != "session_pitch"}
        events_first: Dict[str, float] = {}

        for row in self.iter_rows(table, session_pitch):
            # time
            for c in out_lists.keys():
                v = row.get(c, "")
                if v == "" or v is None:
                    out_lists[c].append(np.nan)
                else:
                    try:
                        out_lists[c].append(float(v))
                    except ValueError:
                        out_lists[c].append(np.nan)
            if include_events:
                for c in event_cols:
                    if c not in events_first:
                        v = row.get(c, "")
                        if v not in ("", None):
                            try:
                                events_first[c] = float(v)
                            except ValueError:
                                pass

        # Convert to numpy arrays.
        out: Dict[str, np.ndarray] = {c: np.asarray(v, dtype=np.float64) for c, v in out_lists.items()}
        # Convenience structured events output.
        if include_events:
            out["_events"] = np.asarray(
                [events_first.get(c, np.nan) for c in event_cols],
                dtype=np.float64,
            )
            out["_event_names"] = np.asarray(event_cols, dtype=object)
        return out

    def load_kinematics(
        self,
        session_pitch: str,
        *,
        include_landmarks: bool = True,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Load the core kinematics tables for a pitch."""
        data: Dict[str, Dict[str, np.ndarray]] = {}
        data["joint_angles"] = self.load_table("joint_angles", session_pitch)
        data["joint_velos"] = self.load_table("joint_velos", session_pitch)
        if include_landmarks:
            data["landmarks"] = self.load_table("landmarks", session_pitch)
        return data

    def load_events(self, session_pitch: str) -> Dict[str, float]:
        """Load canonical event times for a pitch.

        For MVP we read from `force_plate` because it is compact and explicitly
        contains event timestamps as columns (duplicated per sample).
        """
        force_plate = self.load_table(
            "force_plate",
            session_pitch,
            columns=[],
            include_events=True,
        )
        events = self.extract_events(force_plate)
        if not events:
            # Fallback: try joint_angles
            angles = self.load_table("joint_angles", session_pitch, columns=[], include_events=True)
            events = self.extract_events(angles)
        return events

    @staticmethod
    def extract_events(table_dict: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Extract event times (scalars) from the `_events` / `_event_names` fields."""
        names = table_dict.get("_event_names")
        vals = table_dict.get("_events")
        if names is None or vals is None:
            return {}
        events: Dict[str, float] = {}
        for n, v in zip(names.tolist(), vals.tolist()):
            if n and np.isfinite(v):
                events[str(n)] = float(v)
        return events

