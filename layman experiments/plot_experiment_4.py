"""Generate publication-quality charts for Experiment 4: Incremental Value of Exogenous Variables.

Produces:
1. fig1_controlled_comparisons_forest.png: Forest plot with 95% paired bootstrap CIs
2. fig2_model_ladder_wape.png: Model ladder across Daily and 7-Day replenishment horizons
3. fig3_segmented_slices.png: Event days slice and Lead-time horizon curves
4. fig4_atms_site_types.png: Site types and individual ATM breakdown
5. experiment_4_executive_dashboard.png: Comprehensive 4-panel executive graphic
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "output"
ARTIFACT_DIR = Path("/Users/madhavpatel/.gemini/antigravity-ide/brain/9cf32330-589c-44d5-b25f-e0c20f75e706")

# Set consistent, elegant plot styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "semibold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
    "figure.titleweight": "bold",
    "figure.autolayout": False,
})

COLORS = {
    "primary": "#1f77b4",     # Blue
    "secondary": "#2ca02c",   # Green
    "accent": "#ff7f0e",      # Orange
    "neutral": "#7f7f7f",     # Gray
    "danger": "#d62728",      # Red
    "purple": "#9467bd",      # Purple
    "teal": "#17becf",        # Teal
    "navy": "#0f2b48",
    "gold": "#d4af37",
    "card_bg": "#f8f9fa",
}

def plot_fig1_forest_comparisons():
    """Forest plot of controlled comparisons and 95% block bootstrap CIs."""
    df_comp = pd.read_csv(OUT_DIR / "experiment_4_controlled_comparisons.csv")
    
    # Parse point estimates and CIs
    comparisons = [
        "Calendar vs History Only",
        "Events/Holidays vs History+Calendar",
        "Weather vs History+Calendar",
        "All External (Full Stack) vs History+Calendar",
        "Value of Weather in Full Stack (F vs F-Weather)",
        "Value of Holidays in Full Stack (F vs F-Holidays)",
        "Value of Calendar in Full Stack (F vs F-Calendar)",
    ]
    
    # Filter and reorder
    df_plot = df_comp.set_index("Comparison").loc[[
        "Calendar vs History Only",
        "Events/Holidays vs History+Calendar",
        "Weather vs History+Calendar",
        "All External (Full Stack) vs History+Calendar",
        "Value of Weather in Full Stack (F vs F-minus-Weather)",
        "Value of Holidays in Full Stack (F vs F-minus-Holidays)",
        "Value of Calendar in Full Stack (F vs F-minus-Calendar)",
    ]].reset_index()

    labels = [
        "Calendar vs. History Alone\n(+2.77 pp, +22.8% rel)",
        "Events/Holidays vs. Hist+Cal\n(+0.19 pp, +2.0% rel)",
        "Weather vs. Hist+Cal\n(+0.08 pp, +0.9% rel)",
        "All External (Full Stack) vs. Hist+Cal\n(+0.29 pp, +3.1% rel)",
        "Ablation: Weather Value in Stack\n(+0.10 pp, +1.0% rel)",
        "Ablation: Holiday Value in Stack\n(+0.20 pp, +2.2% rel)",
        "Ablation: Calendar Value in Stack\n(+2.63 pp, +22.5% rel)",
    ]

    points_7d = [2.77, 0.19, 0.08, 0.29, 0.10, 0.20, 2.63]
    ci_7d_low = [1.58, 0.04, -0.18, -0.05, -0.21, -0.01, 1.58]
    ci_7d_high = [3.96, 0.35, 0.35, 0.63, 0.45, 0.45, 3.60]

    points_daily = [3.01, 0.18, 0.03, 0.24, 0.06, 0.20, 3.05]
    ci_d_low = [2.31, 0.01, -0.13, 0.02, -0.12, 0.04, 2.42]
    ci_d_high = [3.60, 0.36, 0.23, 0.47, 0.25, 0.36, 3.60]

    y_pos = np.arange(len(labels))[::-1]

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)

    # Shaded region for zero crossing (insignificant)
    ax.axvspan(-0.5, 0, color="#fde8e8", alpha=0.5, label="Negative / No Improvement")
    ax.axvline(0, color="#d62728", linestyle="--", linewidth=1.5, alpha=0.8)

    # Plot 7-Day Cycle CIs (Primary Replenishment metric)
    offset = 0.15
    for i, y in enumerate(y_pos):
        pt = points_7d[i]
        lo = ci_7d_low[i]
        hi = ci_7d_high[i]
        color = "#2ca02c" if lo > 0 else ("#ff7f0e" if pt > 0 else "#7f7f7f")
        ax.errorbar(pt, y + offset, xerr=[[pt - lo], [hi - pt]], fmt="o", color=color,
                    ecolor=color, elinewidth=2.5, capsize=5, capthick=2, markersize=8)
        ax.text(hi + 0.12, y + offset, f"{pt:+.2f} pp", va="center", fontsize=9, fontweight="bold", color=color)

    # Plot Daily CIs
    for i, y in enumerate(y_pos):
        pt = points_daily[i]
        lo = ci_d_low[i]
        hi = ci_d_high[i]
        color = "#1f77b4" if lo > 0 else ("#17becf" if pt > 0 else "#7f7f7f")
        ax.errorbar(pt, y - offset, xerr=[[pt - lo], [hi - pt]], fmt="s", color=color,
                    ecolor=color, elinewidth=1.8, capsize=4, capthick=1.5, markersize=6, alpha=0.85)

    # Legend entries dummy
    ax.plot([], [], "o", color="#2ca02c", markersize=8, label="7-Day Cycle WAPE Reduction (Operational Target)")
    ax.plot([], [], "s", color="#1f77b4", markersize=6, label="Daily WAPE Reduction (Single Day)")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontweight="medium")
    ax.set_xlabel("WAPE Error Reduction (Percentage Points, higher = greater improvement)")
    ax.set_title("Experiment 4: Incremental Value of Exogenous Information Groups\n(Paired Block-Bootstrap 95% Confidence Intervals across 55 Weekly Origins)", pad=15)
    ax.set_xlim(-0.6, 4.5)
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.95)
    
    # Annotation on significance
    ax.annotate("Calendar is decisive\n(+2.77 pp, 95% CI strictly > 0)", xy=(2.77, y_pos[0] + offset),
                xytext=(3.1, y_pos[0] + 0.4), arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.2),
                fontsize=9, fontweight="bold", color="#1b5e20", bbox=dict(boxstyle="round,pad=0.3", fc="#e8f5e9", ec="#a5d6a7"))

    ax.annotate("Weather crosses zero\n(not statistically significant)", xy=(0.08, y_pos[2] + offset),
                xytext=(0.8, y_pos[2] + 0.4), arrowprops=dict(arrowstyle="->", color="#c62828", lw=1.2),
                fontsize=9, fontweight="bold", color="#c62828", bbox=dict(boxstyle="round,pad=0.3", fc="#ffebee", ec="#ef9a9a"))

    plt.tight_layout()
    path = OUT_DIR / "fig1_controlled_comparisons_forest.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")

def plot_fig2_model_ladder():
    """Bar chart comparing all models on Daily vs 7-Day Cycle WAPE."""
    df_overall = pd.read_csv(OUT_DIR / "experiment_4_overall_models.csv")
    
    # Pick core models
    core_models = [
        ("Model_A_History_Only", "A: History Only\n(Lags + Rolling)"),
        ("Model_B_History_Calendar", "B: Hist + Calendar\n(+DOM, DOW, Month)"),
        ("Model_C_History_Cal_Holidays", "C: Hist + Cal + Holidays\n(+RBI Bank Holidays)"),
        ("Model_D_History_Cal_Weather", "D: Hist + Cal + Weather\n(+ERA5 Rain/Temp)"),
        ("Model_F_Full_Stack", "F: Full Stack\n(All Exogenous Groups)"),
        ("Seasonal_Exp3", "Exp 3 Frozen Baseline\n(ATM Mean × DOM)"),
    ]
    
    model_keys = [m[0] for m in core_models]
    model_labels = [m[1] for m in core_models]
    
    sub = df_overall.set_index("Model").loc[model_keys].reset_index()
    
    x = np.arange(len(core_models))
    width = 0.36
    
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    
    rects1 = ax.bar(x - width/2, sub["Daily_WAPE_%"], width, label="Daily WAPE (%)", color="#4a90e2", edgecolor="#1c5ca8", linewidth=1.2)
    rects2 = ax.bar(x + width/2, sub["7Day_WAPE_%"], width, label="7-Day Cycle WAPE (%) [Operational Replenishment]", color="#2ecc71", edgecolor="#1e8449", linewidth=1.2)
    
    # Annotate bar values
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#1c5ca8")
                    
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#1e8449")

    # Add relative improvement arrows
    # A to B
    ax.annotate("22.8% Error Cut\n(12.1% → 9.4%)", xy=(1 + width/2, 9.37), xytext=(0.5, 13.5),
                arrowprops=dict(arrowstyle="->", color="#1e8449", lw=1.5), fontsize=9, fontweight="bold", color="#1e8449",
                bbox=dict(boxstyle="round,pad=0.3", fc="#e8f8f5", ec="#a3e4d7"))

    # A to F
    ax.annotate("25.1% Total Gain\n(12.1% → 9.08%)", xy=(4 + width/2, 9.08), xytext=(3.5, 13.5),
                arrowprops=dict(arrowstyle="->", color="#1e8449", lw=1.5), fontsize=9, fontweight="bold", color="#1e8449",
                bbox=dict(boxstyle="round,pad=0.3", fc="#e8f8f5", ec="#a3e4d7"))

    ax.set_ylabel("WAPE Error Rate (%) - Lower is Better")
    ax.set_title("Forecast Accuracy Progression: Moving Beyond Past Withdrawals\n(Scored on 3,080 Unseen ATM-Days across 55 Consecutive Weekly Origins)", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels)
    ax.set_ylim(0, 26)
    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.95)
    
    # Horizontal line at 10% WAPE (90% accuracy target)
    ax.axhline(10.0, color="#e67e22", linestyle=":", linewidth=1.5, label="10% WAPE Threshold (~90% Accuracy)")
    ax.legend(loc="upper right")

    plt.tight_layout()
    path = OUT_DIR / "fig2_model_ladder_wape.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")

def plot_fig3_segmented_slices():
    """2-panel figure: Event days performance and Lead-time horizon curves."""
    df_events = pd.read_csv(OUT_DIR / "experiment_4_slice_events.csv")
    df_lead = pd.read_csv(OUT_DIR / "experiment_4_slice_horizons.csv")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    # Panel 1: Event Days vs Ordinary Days
    categories = ["Regional Holiday Days", "Payday Days", "Ordinary Days"]
    ev_sub = df_events.set_index("Slice").loc[categories].reset_index()
    
    x = np.arange(len(categories))
    w = 0.22
    
    r1 = ax1.bar(x - 1.5*w, ev_sub["Hist_Only (A)"], w, label="A: History Only", color="#7f8c8d")
    r2 = ax1.bar(x - 0.5*w, ev_sub["Hist+Cal (B)"], w, label="B: History + Calendar", color="#3498db")
    r3 = ax1.bar(x + 0.5*w, ev_sub["Hist+Cal+Holidays (C)"], w, label="C: Hist + Cal + Holidays", color="#e74c3c")
    r4 = ax1.bar(x + 1.5*w, ev_sub["Full Stack (F)"], w, label="F: Full Stack", color="#2ecc71")
    
    for r in r3:
        h = r.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3), textcoords="offset points",
                     ha="center", fontsize=8.5, fontweight="bold", color="#c0392b")

    # Holiday lift callout
    ax1.annotate("Holiday Lift: -1.61 pp\n(24.17% → 22.56%)", xy=(0 + 0.5*w, 22.56), xytext=(-0.25, 25.5),
                 arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.3), fontsize=8.5, fontweight="bold", color="#c0392b",
                 bbox=dict(boxstyle="round,pad=0.25", fc="#fadbd8", ec="#f1948a"))

    ax1.set_ylabel("Daily WAPE (%)")
    ax1.set_title("A. Error on Event Days vs. Ordinary Days", pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Regional Holidays\n(6.5% of days)", "Payday Dates\n(9.5% of days)", "Ordinary Days\n(83.9% of days)"])
    ax1.set_ylim(14, 27)
    ax1.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)
    
    # Panel 2: Forecast Lead Time Horizon (h = 1 to 7)
    days = np.arange(1, 8)
    ax2.plot(days, df_lead["Hist_Only (A)"], marker="o", linewidth=2, color="#7f8c8d", label="A: History Only")
    ax2.plot(days, df_lead["Hist+Cal (B)"], marker="s", linewidth=2, color="#3498db", label="B: Hist + Calendar")
    ax2.plot(days, df_lead["Full Stack (F)"], marker="D", linewidth=2.5, color="#2ecc71", label="F: Full Stack")
    ax2.plot(days, df_lead["Exp3 Seasonal"], marker="^", linewidth=1.8, linestyle="--", color="#e67e22", label="Exp 3 Frozen Seasonal")
    
    # Highlight day 7 drift of seasonal model
    ax2.annotate("Exp 3 Static Drift\n(19.2% → 25.1%)", xy=(7, df_lead["Exp3 Seasonal"].iloc[-1]), xytext=(5.2, 23.8),
                 arrowprops=dict(arrowstyle="->", color="#e67e22", lw=1.2), fontsize=8.5, fontweight="bold", color="#d35400",
                 bbox=dict(boxstyle="round,pad=0.25", fc="#fdebd0", ec="#f8c471"))

    ax2.annotate("Full Stack Stays Flat\n(~18.0% - 18.8%)", xy=(4, df_lead["Full Stack (F)"].iloc[3]), xytext=(2.5, 17.0),
                 arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.2), fontsize=8.5, fontweight="bold", color="#27ae60",
                 bbox=dict(boxstyle="round,pad=0.25", fc="#e8f8f5", ec="#a3e4d7"))

    ax2.set_xlabel("Forecast Lead Time (Day of 7-Day Cycle)")
    ax2.set_ylabel("Daily WAPE (%)")
    ax2.set_title("B. Error Progression Across the 7-Day Horizon", pad=12)
    ax2.set_xticks(days)
    ax2.set_xticklabels([f"Day {h}\n(h={h})" for h in days])
    ax2.set_ylim(16, 26.5)
    ax2.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)

    plt.tight_layout()
    path = OUT_DIR / "fig3_segmented_slices.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")

def plot_fig4_atms_site_types():
    """Horizontal bar chart comparing site types and individual ATMs."""
    df_atm = pd.read_csv(OUT_DIR / "experiment_4_slice_atms.csv")
    
    # Sort by Full Stack 7Day WAPE
    df_atm = df_atm.sort_values("7Day_Full (F)", ascending=True).reset_index(drop=True)
    
    labels = [f"ATM {row.ATM}: {row.City} ({row['Site Type']})" for _, row in df_atm.iterrows()]
    y_pos = np.arange(len(labels))
    height = 0.35
    
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    
    r1 = ax.barh(y_pos + height/2, df_atm["7Day_Hist (A)"], height, label="History Only (A)", color="#bdc3c7", edgecolor="#7f8c8d")
    r2 = ax.barh(y_pos - height/2, df_atm["7Day_Full (F)"], height, label="Full Stack Exogenous (F)", color="#27ae60", edgecolor="#1e8449")
    
    for i, row in df_atm.iterrows():
        # Annotate Full Stack
        ax.text(row["7Day_Full (F)"] + 0.2, i - height/2, f"{row['7Day_Full (F)']:.1f}%", va="center", fontsize=9, fontweight="bold", color="#1e8449")
        # Relative gain
        gain = (row["7Day_Hist (A)"] - row["7Day_Full (F)"]) / row["7Day_Hist (A)"] * 100
        ax.text(row["7Day_Hist (A)"] + 0.2, i + height/2, f"{row['7Day_Hist (A)']:.1f}% (-{gain:.0f}%)", va="center", fontsize=8.5, color="#555")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontweight="medium")
    ax.set_xlabel("7-Day Cycle Replenishment WAPE (%) - Lower is Better")
    ax.set_title("7-Day Replenishment Accuracy by ATM Location & Site Type\n(High-Transit Concourses and Airports Reach 7.4% - 8.8% WAPE)", pad=15)
    ax.set_xlim(0, 16.5)
    ax.axvline(10.0, color="#e67e22", linestyle=":", linewidth=1.5, label="10% WAPE Buffer Target")
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.95)
    
    plt.tight_layout()
    path = OUT_DIR / "fig4_atms_site_types.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")

def plot_fig5_executive_dashboard():
    """Composite 4-panel executive dashboard."""
    fig = plt.figure(figsize=(16, 11), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)

    # Subplot 1: Model Ladder (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    df_overall = pd.read_csv(OUT_DIR / "experiment_4_overall_models.csv")
    core_models = [
        ("Model_A_History_Only", "A: Hist Only"),
        ("Model_B_History_Calendar", "B: +Calendar"),
        ("Model_C_History_Cal_Holidays", "C: +Holidays"),
        ("Model_D_History_Cal_Weather", "D: +Weather"),
        ("Model_F_Full_Stack", "F: Full Stack"),
        ("Seasonal_Exp3", "Exp 3 Frozen"),
    ]
    sub = df_overall.set_index("Model").loc[[m[0] for m in core_models]].reset_index()
    x = np.arange(len(core_models))
    w = 0.36
    ax1.bar(x - w/2, sub["Daily_WAPE_%"], w, label="Daily WAPE", color="#4a90e2")
    ax1.bar(x + w/2, sub["7Day_WAPE_%"], w, label="7-Day Cycle WAPE", color="#2ecc71")
    ax1.set_xticks(x)
    ax1.set_xticklabels([m[1] for m in core_models], rotation=15, ha="right", fontsize=9)
    ax1.set_ylabel("WAPE (%)")
    ax1.set_title("1. Accuracy by Information Tier", pad=10)
    ax1.set_ylim(0, 25)
    ax1.axhline(10.0, color="#e67e22", linestyle=":", label="10% WAPE Target")
    ax1.legend(loc="upper right", fontsize=8.5)

    # Subplot 2: Forest Plot CIs (Top Right)
    ax2 = fig.add_subplot(gs[0, 1])
    labels_short = [
        "Calendar vs History Alone",
        "Events/Holidays vs Hist+Cal",
        "Weather vs Hist+Cal",
        "Full Stack vs Hist+Cal",
        "Ablation: Weather in Stack",
        "Ablation: Holidays in Stack",
    ]
    pts_7d = [2.77, 0.19, 0.08, 0.29, 0.10, 0.20]
    lo_7d = [1.58, 0.04, -0.18, -0.05, -0.21, -0.01]
    hi_7d = [3.96, 0.35, 0.35, 0.63, 0.45, 0.45]
    y_pos = np.arange(len(labels_short))[::-1]
    ax2.axvline(0, color="#d62728", linestyle="--", linewidth=1.2)
    for i, y in enumerate(y_pos):
        pt, lo, hi = pts_7d[i], lo_7d[i], hi_7d[i]
        col = "#2ecc71" if lo > 0 else ("#e67e22" if pt > 0 else "#7f7f7f")
        ax2.errorbar(pt, y, xerr=[[pt - lo], [hi - pt]], fmt="o", color=col, elinewidth=2, capsize=4, markersize=7)
        ax2.text(hi + 0.1, y, f"{pt:+.2f} pp", va="center", fontsize=8.5, fontweight="bold", color=col)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels_short, fontsize=9)
    ax2.set_xlabel("7-Day WAPE Reduction (pp, 95% Bootstrap CI)")
    ax2.set_title("2. Exogenous Gain & Statistical Significance", pad=10)
    ax2.set_xlim(-0.5, 4.5)

    # Subplot 3: Event Days vs Normal Days (Bottom Left)
    ax3 = fig.add_subplot(gs[1, 0])
    df_events = pd.read_csv(OUT_DIR / "experiment_4_slice_events.csv")
    ev_cats = ["Regional Holiday Days", "Payday Days", "Ordinary Days"]
    ev_sub = df_events.set_index("Slice").loc[ev_cats].reset_index()
    x_ev = np.arange(len(ev_cats))
    w_ev = 0.25
    ax3.bar(x_ev - w_ev, ev_sub["Hist_Only (A)"], w_ev, label="History Only (A)", color="#7f8c8d")
    ax3.bar(x_ev, ev_sub["Hist+Cal (B)"], w_ev, label="Hist + Calendar (B)", color="#3498db")
    ax3.bar(x_ev + w_ev, ev_sub["Hist+Cal+Holidays (C)"], w_ev, label="+Holidays (C)", color="#e74c3c")
    ax3.set_xticks(x_ev)
    ax3.set_xticklabels(["Regional Holidays\n(176 ATM-days)", "Payday Dates\n(272 ATM-days)", "Ordinary Days\n(2,632 ATM-days)"], fontsize=9)
    ax3.set_ylabel("Daily WAPE (%)")
    ax3.set_title("3. Event Day Specialization (-1.61 pp on Holidays)", pad=10)
    ax3.set_ylim(15, 26)
    ax3.legend(loc="upper right", fontsize=8.5)

    # Subplot 4: Lead Time Horizon Curve (Bottom Right)
    ax4 = fig.add_subplot(gs[1, 1])
    df_lead = pd.read_csv(OUT_DIR / "experiment_4_slice_horizons.csv")
    days = np.arange(1, 8)
    ax4.plot(days, df_lead["Hist_Only (A)"], marker="o", linewidth=2, color="#7f8c8d", label="History Only (A)")
    ax4.plot(days, df_lead["Hist+Cal (B)"], marker="s", linewidth=2, color="#3498db", label="Hist + Calendar (B)")
    ax4.plot(days, df_lead["Full Stack (F)"], marker="D", linewidth=2.5, color="#2ecc71", label="Full Stack (F)")
    ax4.plot(days, df_lead["Exp3 Seasonal"], marker="^", linewidth=1.5, linestyle="--", color="#e67e22", label="Exp 3 Frozen Seasonal")
    ax4.set_xticks(days)
    ax4.set_xticklabels([f"Day {h}" for h in days], fontsize=9)
    ax4.set_xlabel("Forecast Horizon Lead Day")
    ax4.set_ylabel("Daily WAPE (%)")
    ax4.set_title("4. Error Stability Across 7-Day Cycle", pad=10)
    ax4.set_ylim(16, 26)
    ax4.legend(loc="upper left", fontsize=8.5)

    fig.suptitle("Experiment 4 Executive Summary: Incremental Value of Exogenous Information\n(Controlled Rolling-Origin Backtest Across 55 Origins, 8 Maharashtra ATMs, 3,080 Scored Dates)", y=0.98)
    
    path = OUT_DIR / "experiment_4_executive_dashboard.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

def copy_to_artifacts():
    """Copy all generated figures to artifact directory for embedding."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    for p in OUT_DIR.glob("*.png"):
        shutil.copy(p, ARTIFACT_DIR / p.name)
        print(f"Copied to artifact: {ARTIFACT_DIR / p.name}")

if __name__ == "__main__":
    plot_fig1_forest_comparisons()
    plot_fig2_model_ladder()
    plot_fig3_segmented_slices()
    plot_fig4_atms_site_types()
    plot_fig5_executive_dashboard()
    copy_to_artifacts()
