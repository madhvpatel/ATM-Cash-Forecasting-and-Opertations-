"""Measure how often each ATM is refilled, and the highest balance it holds, from the raw 15-minute files.

A refill is a rise in End_Cash of more than REFILL_JUMP between consecutive snapshots (withdrawals only
lower it). Window: the WINDOW_DAYS days up to the portal cutoff.

Run: python3 scripts/measure_refill_cadence.py  (~5 min) -> data/processed/refill_cadence.csv
"""
import tarfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "refill_cadence.csv"
CUTOFF = pd.Timestamp("2026-08-24")
WINDOW_DAYS = 28
REFILL_JUMP = 100_000.0
MAX_CYCLE_DAYS = 7


def snapshots(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    days = {f"{d:%Y%m%d}" for d in pd.date_range(start, end)}
    frames = []
    for month in sorted({f"{d:%Y%m}" for d in pd.date_range(start, end)}):
        archive = next(RAW.glob(f"*/ATM_CashFiles_15Min_XLSX_{month}01_*.tar.gz"))
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                stamp = member.name.rsplit("_", 1)[-1][:8]
                if member.name.endswith(".xlsx") and stamp in days:
                    frames.append(pd.read_excel(tar.extractfile(member), usecols=["ATM_ID", "End_Cash", "TIMESTAMP"]))
    snap = pd.concat(frames)
    snap["terminal_id"] = snap["ATM_ID"].str[-4:].astype(int)
    return snap.sort_values(["terminal_id", "TIMESTAMP"])


def cadence(snap: pd.DataFrame) -> pd.DataFrame:
    snap = snap.assign(jump=snap.groupby("terminal_id")["End_Cash"].diff())
    refills = snap[snap["jump"] > REFILL_JUMP]
    out = pd.DataFrame({
        "refills": refills.groupby("terminal_id").size(),
        "median_post_refill": refills.groupby("terminal_id")["End_Cash"].median(),
        "max_balance": snap.groupby("terminal_id")["End_Cash"].max(),
    }).fillna({"refills": 0}).reset_index()
    out["refills_per_week"] = out["refills"] * 7 / WINDOW_DAYS
    # Days between visits, rounded; at least 1, at most a week (also when no refill was seen)
    out["cycle_days"] = (7 / out["refills_per_week"].where(out["refills_per_week"] > 0)).round()
    out["cycle_days"] = out["cycle_days"].fillna(MAX_CYCLE_DAYS).clip(1, MAX_CYCLE_DAYS).astype(int)
    return out


def main():
    start = CUTOFF - pd.Timedelta(days=WINDOW_DAYS - 1)
    out = cadence(snapshots(start, CUTOFF)).assign(window_start=str(start.date()), window_end=str(CUTOFF.date()))
    assert len(out) == 100, f"expected 100 ATMs, got {len(out)}"
    out.to_csv(OUT, index=False)
    print(out.head(8).to_string(index=False))
    print(f"cycle_days counts: {out['cycle_days'].value_counts().sort_index().to_dict()} -> {OUT}")


if __name__ == "__main__":
    main()
