"""
Generate attendance reports — daily summary or weekly pivot table.
Usage:
    python report.py                   # today's report
    python report.py --date 2025-10-15 # specific date
    python report.py --summary         # 7-day pivot table
    python report.py --export          # export summary to Excel
"""

import argparse
from src.attendance.attendance_logger import AttendanceLogger


def main():
    parser = argparse.ArgumentParser(description="Attendance Report Generator")
    parser.add_argument("--date", type=str, default=None, help="Date in YYYY-MM-DD format")
    parser.add_argument("--summary", action="store_true", help="Show 7-day attendance summary")
    parser.add_argument("--export", action="store_true", help="Export summary to Excel")
    args = parser.parse_args()

    logger = AttendanceLogger()

    if args.summary or args.export:
        df = logger.get_summary(days=7)
        if df.empty:
            print("[INFO] No attendance data available.")
            return
        print("\n7-Day Attendance Summary")
        print("=" * 60)
        print(df.to_string())
        if args.export:
            out_path = "data/attendance_logs/summary_report.xlsx"
            df.to_excel(out_path)
            print(f"\n[OK] Summary exported → {out_path}")
    else:
        logger.print_report(args.date)


if __name__ == "__main__":
    main()
