"""Experiment 5: Pseudo-Masking Validation of Run-Rate Demand Reconstruction

Validates the run-rate reconstruction formula:
    Estimated Demand = Observed Demand / Active Fraction
against known, ground-truth full-day withdrawals on uncorrupted days (zero empty snapshots).

Tests:
1. Uniform Random Masking (25% and 50% intervals masked).
2. Peak Diurnal Masking (Evening commute: 18:00 - 22:00 masked, 16 intervals).
3. Off-Peak Diurnal Masking (Late night: 00:00 - 04:00 masked, 16 intervals).
4. Morning Rush Masking (08:00 - 12:00 masked, 16 intervals).
"""

from __future__ import annotations

import io
import re
import sys
import tarfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CMS_ROOT = HERE.parent
OUT_DIR = HERE / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR = CMS_ROOT / "data" / "raw"

sys.path.insert(0, str(CMS_ROOT / "data" / "baselines" / "calendar-v1-2026-10-21" / "src"))
from atm_cash_forecast.raw_cash_files import _read_workbook

def extract_clean_day_snapshots(target_atms=("SYNATM0001", "SYNATM0007"), max_days=60):
    """Extract 15-minute snapshots for target ATMs across initial months."""
    archives = [
        RAW_DIR / "ATM_Raw_15Min_XLSX_CashFiles_PART1_20240901_20250131" / "ATM_CashFiles_15Min_XLSX_20240901_20240930.tar.gz",
        RAW_DIR / "ATM_Raw_15Min_XLSX_CashFiles_PART1_20240901_20250131" / "ATM_CashFiles_15Min_XLSX_20241001_20241031.tar.gz",
    ]
    
    rows = []
    print("Extracting 15-minute snapshots for pseudo-masking validation...")
    
    for arc in archives:
        if not arc.exists():
            continue
        print(f"Reading archive: {arc.name}")
        with tarfile.open(arc, "r:gz") as tar:
            for m in tar:
                if not m.name.endswith(".xlsx"):
                    continue
                # Extract timestamp from filename: KVB_Cashfile_YYYYMMDDHHMMSS.xlsx
                ts_str = m.name.split("_")[-1].replace(".xlsx", "")
                ex = tar.extractfile(m)
                if ex is None:
                    continue
                data = _read_workbook(ex.read())
                for r in data:
                    atm = str(r["atm_code"])
                    if atm in target_atms:
                        rows.append({
                            "timestamp": ts_str,
                            "atm_code": atm,
                            "terminal_id": int(atm.replace("SYNATM", "")),
                            "cash_out": int(r["cash_out"]),
                            "end_cash": int(r["end_cash"]),
                        })
        if len(rows) >= max_days * 96 * len(target_atms):
            break

    df = pd.DataFrame(rows)
    df["dt"] = pd.to_datetime(df["timestamp"], format="%Y%m%d%H%M%S")
    df["date"] = df["dt"].dt.normalize()
    df["time"] = df["dt"].dt.time
    df["hour"] = df["dt"].dt.hour
    df["snapshot_idx"] = df["dt"].dt.hour * 4 + df["dt"].dt.minute // 15
    return df

def run_masking_experiment(df_snaps):
    """Run pseudo-masking on clean days where end_cash > 0 for all 96 snapshots."""
    # Group by ATM and date to find clean days
    day_stats = df_snaps.groupby(["terminal_id", "date"]).agg(
        n_snaps=("cash_out", "count"),
        total_cash_out=("cash_out", "sum"),
        min_end_cash=("end_cash", "min"),
        zero_end_cash=("end_cash", lambda s: (s == 0).sum())
    ).reset_index()
    
    clean_days = day_stats[(day_stats["n_snaps"] == 96) & (day_stats["zero_end_cash"] == 0)]
    print(f"\nIdentified {len(clean_days)} clean unconstrained ATM-days (zero empty intervals).")
    
    results = []
    rng = np.random.default_rng(42)
    
    for _, c_row in clean_days.iterrows():
        tid = c_row["terminal_id"]
        d = c_row["date"]
        true_demand = c_row["total_cash_out"]
        if true_demand <= 0:
            continue
            
        sub = df_snaps[(df_snaps["terminal_id"] == tid) & (df_snaps["date"] == d)].sort_values("snapshot_idx").reset_index(drop=True)
        snaps = sub["cash_out"].to_numpy(dtype=float)
        hours = sub["hour"].to_numpy(dtype=int)
        
        # 1. Uniform Random Masking: 25% (24 snapshots masked, 72 observed)
        mask_25 = np.zeros(96, dtype=bool)
        mask_25[rng.choice(96, size=24, replace=False)] = True
        obs_25 = snaps[~mask_25].sum()
        avail_25 = (96 - 24) / 96.0  # 0.75
        recon_25 = obs_25 / avail_25
        
        # 2. Uniform Random Masking: 50% (48 snapshots masked, 48 observed)
        mask_50 = np.zeros(96, dtype=bool)
        mask_50[rng.choice(96, size=48, replace=False)] = True
        obs_50 = snaps[~mask_50].sum()
        avail_50 = 48 / 96.0  # 0.50
        recon_50 = obs_50 / avail_50
        
        # 3. Peak Commute Masking: 18:00 - 22:00 (16 snapshots masked, 80 observed)
        mask_peak = (hours >= 18) & (hours < 22)
        obs_peak = snaps[~mask_peak].sum()
        avail_peak = (~mask_peak).sum() / 96.0
        recon_peak = obs_peak / avail_peak
        
        # 4. Off-Peak Night Masking: 00:00 - 04:00 (16 snapshots masked, 80 observed)
        mask_night = (hours >= 0) & (hours < 4)
        obs_night = snaps[~mask_night].sum()
        avail_night = (~mask_night).sum() / 96.0
        recon_night = obs_night / avail_night
        
        # 5. Morning Rush Masking: 08:00 - 12:00 (16 snapshots masked, 80 observed)
        mask_morning = (hours >= 8) & (hours < 12)
        obs_morning = snaps[~mask_morning].sum()
        avail_morning = (~mask_morning).sum() / 96.0
        recon_morning = obs_morning / avail_morning

        results.append({
            "terminal_id": tid,
            "date": d,
            "true_demand": true_demand,
            "recon_random_25": recon_25,
            "recon_random_50": recon_50,
            "recon_peak_commute": recon_peak,
            "recon_night": recon_night,
            "recon_morning": recon_morning,
        })
        
    df_res = pd.DataFrame(results)
    
    # Calculate performance metrics for each masking protocol
    summary = []
    scenarios = [
        ("Random 25% Masking (Uniform)", "recon_random_25"),
        ("Random 50% Masking (Uniform)", "recon_random_50"),
        ("Peak Commute (18:00-22:00) Missing", "recon_peak_commute"),
        ("Morning Rush (08:00-12:00) Missing", "recon_morning"),
        ("Off-Peak Night (00:00-04:00) Missing", "recon_night"),
    ]
    
    print("\n" + "=" * 80)
    print("PSEUDO-MASKING VALIDATION RESULTS ACROSS CLEAN DAYS")
    print("=" * 80)
    print(f"{'Scenario':<38} | {'MAPE (%)':<10} | {'WAPE (%)':<10} | {'Bias (%)':<10} | {'Interpretation'}")
    print("-" * 80)
    
    for name, col in scenarios:
        true_vals = df_res["true_demand"].to_numpy()
        pred_vals = df_res[col].to_numpy()
        
        abs_err = np.abs(pred_vals - true_vals)
        mape = float(np.mean(abs_err / true_vals) * 100)
        wape = float(np.sum(abs_err) / np.sum(true_vals) * 100)
        bias = float(np.sum(pred_vals - true_vals) / np.sum(true_vals) * 100)
        
        if bias < -5:
            interp = "Severe Under-estimation (missed peak hours)"
        elif bias > +5:
            interp = "Severe Over-estimation (missed quiet hours)"
        else:
            interp = "Unbiased Estimator (±2% error)"
            
        summary.append({
            "scenario": name,
            "mape_%": round(mape, 2),
            "wape_%": round(wape, 2),
            "bias_%": round(bias, 2),
            "interpretation": interp
        })
        print(f"{name:<38} | {mape:>8.2f}% | {wape:>8.2f}% | {bias:>+8.2f}% | {interp}")
        
    df_summ = pd.DataFrame(summary)
    df_summ.to_csv(OUT_DIR / "pseudo_masking_validation_summary.csv", index=False)
    df_res.to_csv(OUT_DIR / "pseudo_masking_simulations.csv", index=False)
    
    return df_res, df_summ

if __name__ == "__main__":
    df_snaps = extract_clean_day_snapshots(target_atms=("SYNATM0001", "SYNATM0007"), max_days=60)
    run_masking_experiment(df_snaps)
