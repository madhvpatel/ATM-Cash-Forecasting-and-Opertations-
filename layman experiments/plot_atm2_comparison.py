"""Generate visual comparison between ATM 1 (Railway) and ATM 2 (Airport).

Outputs:
- layman experiments/output/fig_atm1_vs_atm2_comparison.png
"""

from __future__ import annotations

import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "output"
ARTIFACT_DIR = Path("/Users/madhavpatel/.gemini/antigravity-ide/brain/9cf32330-589c-44d5-b25f-e0c20f75e706")

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "figure.titlesize": 14,
    "figure.titleweight": "bold",
})

def plot_comparison():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

    # Subplot 1: Model Ladder Comparison (7-Day WAPE)
    models = ["Hist Only (A)", "+Calendar (B)", "+Holidays (C)", "+Weather (D)", "Full Stack (F)", "Exp 3 Frozen"]
    atm1_7d = [12.5, 9.8, 9.6, 9.8, 9.5, 12.3]
    atm2_7d = [12.6, 8.53, 8.39, 9.18, 8.96, 10.7]

    x = np.arange(len(models))
    w = 0.35

    r1 = ax1.bar(x - w/2, atm1_7d, w, label="ATM 1: Railway Concourse", color="#3498db")
    r2 = ax1.bar(x + w/2, atm2_7d, w, label="ATM 2: Airport Terminal", color="#2ecc71")

    for r in r1:
        h = r.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3), textcoords="offset points",
                     ha="center", fontsize=8.5, color="#1c5ca8", fontweight="bold")
    for r in r2:
        h = r.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3), textcoords="offset points",
                     ha="center", fontsize=8.5, color="#1e8449", fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=15, ha="right", fontsize=9)
    ax1.set_ylabel("7-Day Cycle WAPE (%)")
    ax1.set_title("A. 7-Day Replenishment Error: ATM 1 vs ATM 2", pad=10)
    ax1.set_ylim(0, 16)
    ax1.axhline(10.0, color="#e67e22", linestyle=":", label="10% Buffer Target")
    ax1.legend(loc="upper right", fontsize=8.5)

    # Subplot 2: Day of Month Salary Wave (Y1 vs Y2 Multipliers)
    panel = pd.read_csv(HERE.parent / "simulated" / "maharashtra" / "data" / "atm_daily.csv", parse_dates=["date"])
    atm2 = panel[panel["terminal_id"] == 2]
    y1_2 = atm2[(atm2["date"] >= "2024-09-01") & (atm2["date"] <= "2025-08-31")]
    y2_2 = atm2[(atm2["date"] >= "2025-09-01") & (atm2["date"] <= "2026-08-31")]

    dom1 = y1_2.groupby("dom")["demand"].mean() / y1_2["demand"].mean()
    dom2 = y2_2.groupby("dom")["demand"].mean() / y2_2["demand"].mean()

    days = np.arange(1, 32)
    ax2.plot(days, dom1, marker="o", markersize=4, color="#3498db", label="Year 1 Profile (2024-25)")
    ax2.plot(days, dom2, marker="s", markersize=4, color="#2ecc71", linestyle="--", label="Year 2 Actual (2025-26)")
    ax2.axhline(1.0, color="#7f8c8d", linestyle=":", alpha=0.7)
    ax2.set_xlabel("Day of Month (1 to 31)")
    ax2.set_ylabel("Demand Multiplier (Relative to Mean)")
    ax2.set_title("B. ATM 2 (Airport) Salary Cycle Signature (r = 0.939)", pad=10)
    ax2.legend(loc="upper right", fontsize=8.5)
    ax2.annotate("Month-End Rush\n(+20% to +35%)", xy=(30, 1.25), xytext=(22, 1.35),
                 arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.2), fontsize=8.5, fontweight="bold", color="#27ae60")
    ax2.annotate("Salary Payout Wave\n(+15% to +30%)", xy=(4, 1.25), xytext=(7, 1.35),
                 arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.2), fontsize=8.5, fontweight="bold", color="#27ae60")

    # Subplot 3: Day of Week Rhythm (Railway vs Airport)
    atm1 = panel[panel["terminal_id"] == 1]
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_atm1 = atm1.groupby("dow")["demand"].mean() / atm1["demand"].mean()
    dow_atm2 = atm2.groupby("dow")["demand"].mean() / atm2["demand"].mean()

    ax3.plot(dow_names, dow_atm1, marker="o", linewidth=2, color="#3498db", label="ATM 1: Railway (DOW Corr: 0.58)")
    ax3.plot(dow_names, dow_atm2, marker="D", linewidth=2.2, color="#e67e22", label="ATM 2: Airport (DOW Corr: 0.84)")
    ax3.axhline(1.0, color="#7f8c8d", linestyle=":", alpha=0.7)
    ax3.set_ylabel("Demand Multiplier")
    ax3.set_title("C. Weekly Travel Patterns: Airport vs Railway Concourse", pad=10)
    ax3.legend(loc="lower left", fontsize=8.5)
    ax3.annotate("Airport Weekend Flight Surge\n(Sat/Sun ~1.10x)", xy=(6, 1.10), xytext=(4.2, 1.13),
                 arrowprops=dict(arrowstyle="->", color="#d35400", lw=1.2), fontsize=8.5, fontweight="bold", color="#d35400")

    # Subplot 4: Feature Importance Comparison (Gain Share)
    imp_df = pd.read_csv(OUT_DIR / "atm2_feature_importance.csv").head(6)
    y_pos = np.arange(len(imp_df))[::-1]
    ax4.barh(y_pos, imp_df["gain_pct"], color="#9b59b6", edgecolor="#8e44ad")
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(imp_df["feature"], fontsize=9, family="monospace")
    ax4.set_xlabel("Split Gain Share (%)")
    ax4.set_title("D. Top Feature Drivers in Model F (ATM 2 Airport)", pad=10)
    for i, row in imp_df.iterrows():
        ax4.text(row["gain_pct"] + 0.3, len(imp_df) - 1 - i, f"{row['gain_pct']:.1f}%", va="center", fontsize=8.5, fontweight="bold")
    ax4.set_xlim(0, 25)

    fig.suptitle("Reproducibility Study: ATM 2 (Mumbai Airport) vs ATM 1 (Mumbai Railway Concourse)\nEvaluating Exogenous Drivers Across Two Distinct High-Transit Sites in the Same City", y=0.98)

    plt.tight_layout()
    out_path = OUT_DIR / "fig_atm1_vs_atm2_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Copy to artifacts
    shutil.copy(out_path, ARTIFACT_DIR / "fig_atm1_vs_atm2_comparison.png")
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    plot_comparison()
