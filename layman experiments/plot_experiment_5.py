"""Plot Experiment 5: Pseudo-Masking Validation & Replenishment Pareto Frontier

Generates:
- output/fig_experiment_5_cashout_reconstruction.png
- Copies to brain artifact directory
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shutil

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR = Path("/Users/madhavpatel/.gemini/antigravity-ide/brain/9cf32330-589c-44d5-b25f-e0c20f75e706")

def generate_experiment_5_plot():
    # Load required data
    mask_summ = pd.read_csv(OUT_DIR / "pseudo_masking_validation_summary.csv")
    tradeoff = pd.read_csv(OUT_DIR / "experiment_5_replenishment_tradeoff.csv")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # -------------------------------------------------------------
    # Plot 1: Pseudo-Masking Validation on Complete Days (Estimator Bias)
    # -------------------------------------------------------------
    scenarios = mask_summ["scenario"].tolist()
    biases = mask_summ["bias_%"].tolist()
    mapes = mask_summ["mape_%"].tolist()
    
    clean_labels = [
        "Uniform Random\n(25% Missing)",
        "Uniform Random\n(50% Missing)",
        "Peak Commute\n(18-22h Missing)",
        "Morning Rush\n(08-12h Missing)",
        "Off-Peak Night\n(00-04h Missing)",
    ]
    
    x = np.arange(len(clean_labels))
    colors = ["#2ecc71" if abs(b) <= 2 else "#e74c3c" if b < 0 else "#f39c12" for b in biases]
    
    bars = ax1.bar(x, biases, color=colors, edgecolor="#2c3e50", width=0.55)
    ax1.axhline(0, color="#34495e", linestyle="--", lw=1.2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(clean_labels, fontsize=8.5)
    ax1.set_ylabel("Reconstruction Bias (% vs True Demand)", fontweight="bold")
    ax1.set_title("A. Pseudo-Masking Validation: Run-Rate Estimator Bias on Clean Days", pad=10, fontweight="bold")
    ax1.set_ylim(-20, 26)
    
    for b, bias_val, mape_val in zip(bars, biases, mapes):
        y_pos = bias_val + (1.2 if bias_val >= 0 else -2.5)
        ax1.annotate(f"Bias: {bias_val:+.1f}%\n(MAPE: {mape_val:.1f}%)", xy=(b.get_x() + b.get_width()/2, y_pos),
                     ha="center", fontsize=8, fontweight="bold",
                     color="#1e8449" if abs(bias_val) <= 2 else "#c0392b" if bias_val < 0 else "#d35400")
        
    ax1.annotate("Diurnal Rush Loss:\nEvening cashout under-\nestimates by -12.8%", xy=(2, -12.8), xytext=(2.2, -17.5),
                 arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2), fontsize=8, fontweight="bold", color="#c0392b",
                 bbox=dict(boxstyle="round,pad=0.2", fc="#fadbd8", ec="#f1948a"))

    # -------------------------------------------------------------
    # Plot 2: Demand Truncation Spectrum (ATM 7: Nashik Metro)
    # -------------------------------------------------------------
    categories = ["Normal Days\n(100% Full)", "Cashout Days\n(Observed Dispensed)", "Cashout Days\n(Run-Rate Reconstructed)"]
    demands = [10.58, 12.31, 14.66]
    cat_colors = ["#95a5a6", "#e67e22", "#27ae60"]
    
    b_cat = ax2.bar(categories, demands, color=cat_colors, edgecolor="#2c3e50", width=0.5)
    ax2.set_ylabel("Daily Demand (INR Lakh)", fontweight="bold")
    ax2.set_title("B. Demand Truncation on Cashout Days (ATM 7: Nashik Metro)", pad=10, fontweight="bold")
    ax2.set_ylim(0, 18)
    
    for b, val in zip(b_cat, demands):
        ax2.annotate(f"INR {val:.2f}L", xy=(b.get_x() + b.get_width()/2, val), xytext=(0, 4),
                     textcoords="offset points", ha="center", fontsize=9, fontweight="bold")
        
    ax2.annotate("+19.1% Estimated Demand Gap\n(+INR 2.35L/day unserved)", xy=(2, 14.66), xytext=(1.05, 16.2),
                 arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.2), fontsize=8.5, fontweight="bold", color="#27ae60",
                 bbox=dict(boxstyle="round,pad=0.25", fc="#e8f8f5", ec="#a3e4d7"))

    # -------------------------------------------------------------
    # Plot 3: Replenishment Pareto Frontier (Stockouts vs Ending Idle Cash)
    # -------------------------------------------------------------
    # ATM 7: Nashik Metro
    sub7 = tradeoff[tradeoff["terminal_id"] == 7]
    cens7 = sub7[sub7["model"] == "Censored Baseline"].sort_values("cer_buffer_target_%")
    recon7 = sub7[sub7["model"] == "Reconstructed Model"].sort_values("cer_buffer_target_%")
    
    ax3.plot(cens7["mean_ending_idle_cash_lakh"], cens7["stockout_cycles"], "o--", color="#e74c3c", lw=2, markersize=7, label="Censored Model (Current Baseline)")
    ax3.plot(recon7["mean_ending_idle_cash_lakh"], recon7["stockout_cycles"], "s-", color="#27ae60", lw=2.2, markersize=8, label="Reconstructed Model (Experiment 5)")
    
    for _, r in cens7.iterrows():
        ax3.annotate(f"{r['cer_buffer_target_%']}%", (r["mean_ending_idle_cash_lakh"], r["stockout_cycles"]), xytext=(-8, 6), textcoords="offset points", fontsize=8, color="#c0392b")
    for _, r in recon7.iterrows():
        ax3.annotate(f"{r['cer_buffer_target_%']}%", (r["mean_ending_idle_cash_lakh"], r["stockout_cycles"]), xytext=(6, -8), textcoords="offset points", fontsize=8, color="#1e8449", fontweight="bold")
        
    ax3.set_xlabel("Average Ending Idle Cash (INR Lakh in Vault)", fontweight="bold")
    ax3.set_ylabel("Stockout Replenishment Cycles (Out of 55)", fontweight="bold")
    ax3.set_title("C. Operational Trade-Off: Stockout Risk vs. Ending Idle Cash (ATM 7)", pad=10, fontweight="bold")
    ax3.legend(loc="upper right", fontsize=9)
    ax3.set_ylim(-0.5, 15)
    ax3.grid(True, linestyle=":", alpha=0.6)

    # -------------------------------------------------------------
    # Plot 4: Total Cost of Cash Logistics (Holding Cost + Shortfall Penalty)
    # -------------------------------------------------------------
    # Assume CIT emergency emergency cash run / stockout penalty = INR 5,000 per empty cycle
    # Annual holding cost = Mean Inventory * 6% p.a.
    for m_label, df_m, col, mark in [("Censored Baseline", cens7, "#e74c3c", "o--"), ("Reconstructed Model", recon7, "#27ae60", "s-")]:
        tot_cost = df_m["annual_idle_cash_holding_cost_inr"] + (df_m["stockout_cycles"] * 5000)
        ax4.plot(df_m["cer_buffer_target_%"], tot_cost / 1e3, mark, color=col, lw=2, markersize=7, label=m_label)
        
    ax4.set_xlabel("Target Cash Efficiency Ratio / Buffer (%)", fontweight="bold")
    ax4.set_ylabel("Annualized Logistics & Capital Cost (INR Thousand)", fontweight="bold")
    ax4.set_title("D. Total Logistics Cost: Idle Cash Cost + Emergency Stockout Risk", pad=10, fontweight="bold")
    ax4.legend(loc="upper right", fontsize=9)
    ax4.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle("Experiment 5: Cashout-Aware Demand Reconstruction & Operational Replenishment\nPseudo-Masking Validation, Diurnal Peak Bias, and the Stockout vs. Idle Cash Frontier",
                 fontsize=14, fontweight="bold", y=0.98)
    
    plt.tight_layout()
    out_file = OUT_DIR / "fig_experiment_5_cashout_reconstruction.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    
    shutil.copy(out_file, ARTIFACT_DIR / "fig_experiment_5_cashout_reconstruction.png")
    print(f"Saved Experiment 5 plot to {out_file} and copied to {ARTIFACT_DIR / 'fig_experiment_5_cashout_reconstruction.png'}")

if __name__ == "__main__":
    generate_experiment_5_plot()
