"""
Attendance Logger — marks, stores, and reports daily attendance via CSV (Pandas).
"""

import os
import time
from datetime import datetime
import pandas as pd
from src.utils.config import Config


class AttendanceLogger:
    """
    Manages daily attendance records.
    - Prevents duplicate entries using a cooldown period.
    - Persists records to dated CSV files.
    - Provides report generation.
    """

    def __init__(self):
        self._today = datetime.now().strftime(Config.DATE_FORMAT)
        self._csv_path = os.path.join(Config.ATTENDANCE_DIR, f"attendance_{self._today}.csv")
        self._cooldown: dict[str, float] = {}   # name → last marked timestamp
        self._df = self._load_or_create()

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _load_or_create(self) -> pd.DataFrame:
        if os.path.exists(self._csv_path):
            df = pd.read_csv(self._csv_path)
            # Rebuild cooldown from existing records
            for _, row in df.iterrows():
                self._cooldown[row[Config.CSV_NAME_COLUMN]] = time.time()
            print(f"[INFO] Loaded existing attendance: {self._csv_path}")
            return df
        cols = [Config.CSV_DATE_COLUMN, Config.CSV_NAME_COLUMN,
                Config.CSV_TIME_COLUMN, Config.CSV_STATUS_COLUMN]
        return pd.DataFrame(columns=cols)

    def _is_on_cooldown(self, name: str) -> bool:
        last = self._cooldown.get(name, 0)
        return (time.time() - last) < Config.ATTENDANCE_COOLDOWN_SECONDS

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def mark_attendance(self, name: str, status: str = "Present") -> bool:
        """
        Mark attendance for a student.
        Returns True if newly marked, False if on cooldown.
        """
        if self._is_on_cooldown(name):
            return False

        now = datetime.now()
        record = {
            Config.CSV_DATE_COLUMN: now.strftime(Config.DATE_FORMAT),
            Config.CSV_NAME_COLUMN: name,
            Config.CSV_TIME_COLUMN: now.strftime(Config.TIME_FORMAT),
            Config.CSV_STATUS_COLUMN: status,
        }
        self._df = pd.concat([self._df, pd.DataFrame([record])], ignore_index=True)
        self._cooldown[name] = time.time()
        self.save()
        print(f"[ATTENDANCE] ✓ {name} — {status} at {record[Config.CSV_TIME_COLUMN]}")
        return True

    def save(self):
        """Write current attendance DataFrame to CSV."""
        self._df.to_csv(self._csv_path, index=False)

    def count_today(self) -> int:
        return len(self._df[Config.CSV_NAME_COLUMN].unique()) if not self._df.empty else 0

    def get_today_records(self) -> pd.DataFrame:
        return self._df.copy()

    def print_report(self, date: str | None = None):
        """Print a formatted attendance report for a given date (default: today)."""
        target = date or self._today
        path = os.path.join(Config.ATTENDANCE_DIR, f"attendance_{target}.csv")
        if not os.path.exists(path):
            print(f"[INFO] No attendance record for {target}.")
            return
        df = pd.read_csv(path)
        print(f"\n{'=' * 45}")
        print(f"  ATTENDANCE REPORT — {target}")
        print(f"{'=' * 45}")
        print(df.to_string(index=False))
        print(f"\n  Total present: {len(df[Config.CSV_NAME_COLUMN].unique())}")
        print(f"{'=' * 45}\n")

    def get_summary(self, days: int = 7) -> pd.DataFrame:
        """Return a pivot-table summary of the last N days of attendance."""
        frames = []
        for fname in sorted(os.listdir(Config.ATTENDANCE_DIR)):
            if fname.startswith("attendance_") and fname.endswith(".csv"):
                path = os.path.join(Config.ATTENDANCE_DIR, fname)
                frames.append(pd.read_csv(path))
        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames, ignore_index=True)
        pivot = combined.pivot_table(
            index=Config.CSV_NAME_COLUMN,
            columns=Config.CSV_DATE_COLUMN,
            values=Config.CSV_STATUS_COLUMN,
            aggfunc="first",
        ).fillna("Absent")
        return pivot
