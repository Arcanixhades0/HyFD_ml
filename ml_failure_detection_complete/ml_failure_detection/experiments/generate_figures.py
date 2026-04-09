"""
Figure Generator for HyFD Paper
=================================
Generates all publication-quality figures for the research paper.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import os

RESULTS_DIR = "../results"
FIGURES_DIR = "../figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# Publication style
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

COLORS = {
    "HyFD": "#1a73e8",
    "DriftOnly": "#e67e22",
    "UncertaintyOnly": "#27ae60",
    "QualityOnly": "#8e44ad",
    "OODOnly": "#e74c3c",
}

SCENARIO_LABELS = {
    "no_failure": "No Failure",
    "data_drift": "Data Drift",
    "noisy_features": "Noisy Features",
    "missing_data": "Missing Data",
    "ood_samples": "OOD Samples",
    "corrupted_data": "Corrupted Data",
}


# ──────────────────────────────────────────────────────────────
# Figure 1: System Architecture Diagram
# ──────────────────────────────────────────────────────────────
def fig_system_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#f8f9fa")

    def box(x, y, w, h, text, color, fontsize=10, text_color="white"):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.1",
            linewidth=1.5, edgecolor="#2c3e50", facecolor=color, zorder=3
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=text_color,
                zorder=4, wrap=True, multialignment="center")

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#2c3e50",
                                   lw=1.5), zorder=5)

    # Production Data
    box(0.2, 3.0, 2.0, 1.2, "Production\nData Batch", "#2c3e50", 10)
    arrow(2.2, 3.6, 3.0, 3.6)

    # ML Model
    box(3.0, 3.0, 2.0, 1.2, "Production\nML Model", "#16213e", 10)
    arrow(5.0, 3.6, 5.8, 3.6)

    # Predictions output
    box(5.8, 3.0, 1.8, 1.2, "Predictions\n+ Probabilities", "#34495e", 10)

    # Signal Detectors
    arrow(1.2, 3.0, 1.2, 2.2)
    arrow(4.0, 3.0, 4.0, 2.2)
    arrow(6.7, 3.0, 6.7, 2.2)
    arrow(1.2, 3.0, 1.2, 1.2)

    box(0.1, 1.0, 1.8, 1.0, "Drift\nDetector", "#1a73e8", 9)
    box(2.1, 1.0, 1.8, 1.0, "Uncertainty\nEstimator", "#27ae60", 9)
    box(4.1, 1.0, 1.8, 1.0, "Slice\nAnalyzer", "#e67e22", 9)
    box(6.1, 1.0, 1.8, 1.0, "Quality\nMonitor", "#8e44ad", 9)
    box(8.1, 1.0, 1.8, 1.0, "OOD\nDetector", "#e74c3c", 9)

    # Arrows from signals to fusion
    for xi in [1.0, 3.0, 5.0, 7.0, 9.0]:
        arrow(xi, 1.0, xi, 0.5)
        # arrows converge
        ax.annotate("", xy=(5.8, 0.35), xytext=(xi, 0.5),
                    arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=1.0))

    # Weighted Fusion
    box(4.5, -0.1, 2.6, 0.8, "Weighted Signal Fusion (HyFD)", "#c0392b", 10)
    arrow(5.8, 0.7, 5.8, 1.5)

    # Alert
    box(9.8, 2.8, 1.8, 1.5, "Failure\nAlert\n⚠", "#e74c3c", 10)
    arrow(7.1, 0.3, 10.7, 0.3)
    ax.annotate("", xy=(10.7, 2.8), xytext=(10.7, 0.3),
                arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=2.0))

    # Reference database
    box(0.2, 5.3, 2.0, 1.0, "Reference\nData (D_ref)", "#7f8c8d", 9)
    for xi, yi in [(1.0, 6.3), (1.0, 6.3)]:
        ax.annotate("", xy=(1.0, 2.0), xytext=(1.0, 5.3),
                    arrowprops=dict(arrowstyle="->", color="#7f8c8d",
                                   lw=1.0, linestyle="dashed"))

    ax.text(6.0, 6.7, "HyFD System Architecture",
            ha="center", va="center", fontsize=15, fontweight="bold", color="#2c3e50")

    # Legend
    legend_patches = [
        mpatches.Patch(color="#1a73e8", label="Drift Detection"),
        mpatches.Patch(color="#27ae60", label="Uncertainty"),
        mpatches.Patch(color="#e67e22", label="Slice Analysis"),
        mpatches.Patch(color="#8e44ad", label="Data Quality"),
        mpatches.Patch(color="#e74c3c", label="OOD Detection"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9,
              framealpha=0.9, ncol=5)

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig1_system_architecture.png")
    plt.close()
    print("  ✓ Figure 1: System architecture")


# ──────────────────────────────────────────────────────────────
# Figure 2: Overall Detection Performance Comparison
# ──────────────────────────────────────────────────────────────
def fig_overall_performance():
    metrics = pd.read_csv(f"{RESULTS_DIR}/overall_metrics.csv")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    detectors = metrics["Detector"].tolist()
    colors = [COLORS.get(d, "#95a5a6") for d in detectors]

    metrics_to_plot = [
        ("Accuracy (%)", "Detection Accuracy (%)", "Higher is Better"),
        ("FPR (%)", "False Positive Rate (%)", "Lower is Better"),
        ("F1-Score", "F1-Score", "Higher is Better"),
    ]

    for ax, (col, ylabel, note) in zip(axes, metrics_to_plot):
        bars = ax.bar(range(len(detectors)), metrics[col], color=colors,
                      edgecolor="white", linewidth=1.2, zorder=3)
        ax.set_xticks(range(len(detectors)))
        ax.set_xticklabels(detectors, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel}\n({note})", fontsize=11)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.set_ylim(0, max(metrics[col].max() * 1.15, 1.0))

        for bar, val in zip(bars, metrics[col]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.1f}" if col != "F1-Score" else f"{val:.3f}",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold")

        # Highlight HyFD
        hyfd_idx = detectors.index("HyFD") if "HyFD" in detectors else None
        if hyfd_idx is not None:
            bars[hyfd_idx].set_edgecolor("#c0392b")
            bars[hyfd_idx].set_linewidth(2.5)

    plt.suptitle("Detection Performance: HyFD vs. Baseline Methods",
                  fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig2_overall_performance.png")
    plt.close()
    print("  ✓ Figure 2: Overall performance comparison")


# ──────────────────────────────────────────────────────────────
# Figure 3: Per-Scenario Heatmap
# ──────────────────────────────────────────────────────────────
def fig_per_scenario_heatmap():
    per_scen = pd.read_csv(f"{RESULTS_DIR}/per_scenario_accuracy.csv", index_col=0)
    per_scen.index.name = "Detector"

    # Rename columns
    per_scen.columns = [SCENARIO_LABELS.get(c, c) for c in per_scen.columns]

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.heatmap(per_scen, annot=True, fmt=".0f", cmap="RdYlGn",
                vmin=0, vmax=100, linewidths=0.5, linecolor="#cccccc",
                ax=ax, cbar_kws={"label": "Detection Accuracy (%)"})
    ax.set_title("Per-Scenario Detection Accuracy (%) by Method",
                  fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Failure Scenario")
    ax.set_ylabel("Detection Method")
    ax.tick_params(axis="x", rotation=25)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig3_per_scenario_heatmap.png")
    plt.close()
    print("  ✓ Figure 3: Per-scenario heatmap")


# ──────────────────────────────────────────────────────────────
# Figure 4: Signal Contribution Radar / Grouped Bar
# ──────────────────────────────────────────────────────────────
def fig_signal_contribution():
    sig = pd.read_csv(f"{RESULTS_DIR}/signal_contribution.csv")
    signals = ["Drift Signal", "Uncertainty Signal", "Slice Signal",
               "Quality Signal", "OOD Signal"]
    signal_colors = ["#1a73e8", "#27ae60", "#e67e22", "#8e44ad", "#e74c3c"]

    x = np.arange(len(sig["Scenario"]))
    width = 0.15

    fig, ax = plt.subplots(figsize=(13, 5.5))

    for i, (sig_name, color) in enumerate(zip(signals, signal_colors)):
        offset = (i - 2) * width
        bars = ax.bar(x + offset, sig[sig_name], width,
                      label=sig_name.replace(" Signal", ""), color=color,
                      alpha=0.85, edgecolor="white", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS.get(s, s) for s in sig["Scenario"]],
                        rotation=20, ha="right")
    ax.set_ylabel("Normalized Signal Score (0–1)")
    ax.set_title("HyFD Signal Contribution per Failure Scenario",
                  fontsize=13, fontweight="bold")
    ax.legend(title="Detection Signal", bbox_to_anchor=(1.01, 1),
               loc="upper left", framealpha=0.9)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0.35, color="#c0392b", linestyle="--", linewidth=1.2,
                label="Detection Threshold")
    ax.text(len(x) - 0.5, 0.37, "Detection Threshold (τ=0.35)",
            color="#c0392b", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig4_signal_contribution.png")
    plt.close()
    print("  ✓ Figure 4: Signal contribution")


# ──────────────────────────────────────────────────────────────
# Figure 5: Latency vs Batch Size
# ──────────────────────────────────────────────────────────────
def fig_latency():
    lat = pd.read_csv(f"{RESULTS_DIR}/latency_results.csv")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lat["batch_size"], lat["mean_latency_ms"], "o-",
            color=COLORS["HyFD"], linewidth=2.5, markersize=7,
            label="HyFD Mean Latency", zorder=3)
    ax.fill_between(
        lat["batch_size"],
        lat["mean_latency_ms"] - lat["std_latency_ms"],
        lat["mean_latency_ms"] + lat["std_latency_ms"],
        alpha=0.2, color=COLORS["HyFD"], label="±1 std"
    )
    ax.axhline(200, color="#e74c3c", linestyle="--", linewidth=1.5,
                label="200ms SLA Target", alpha=0.7)
    ax.set_xlabel("Production Batch Size (samples)")
    ax.set_ylabel("Detection Latency (ms)")
    ax.set_title("HyFD Detection Latency vs. Batch Size",
                  fontsize=13, fontweight="bold")
    ax.legend(framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.set_xscale("log")
    ax.set_xticks(lat["batch_size"])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig5_latency.png")
    plt.close()
    print("  ✓ Figure 5: Latency analysis")


# ──────────────────────────────────────────────────────────────
# Figure 6: Score Distribution Under Different Scenarios
# ──────────────────────────────────────────────────────────────
def fig_score_distributions():
    raw = pd.read_csv(f"{RESULTS_DIR}/raw_results.csv")
    hyfd_raw = raw[raw["detector"] == "HyFD"]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    axes = axes.flatten()

    scenarios = ["no_failure", "data_drift", "noisy_features",
                  "missing_data", "ood_samples", "corrupted_data"]
    scen_colors = {
        "no_failure": "#27ae60",
        "data_drift": "#1a73e8",
        "noisy_features": "#e67e22",
        "missing_data": "#8e44ad",
        "ood_samples": "#e74c3c",
        "corrupted_data": "#c0392b"
    }

    for ax, scen in zip(axes, scenarios):
        sub = hyfd_raw[hyfd_raw["scenario"] == scen]["score"]
        color = scen_colors.get(scen, "#555")
        ax.hist(sub, bins=15, color=color, alpha=0.75, edgecolor="white")
        ax.axvline(0.35, color="black", linestyle="--", linewidth=1.5,
                    label="Threshold")
        ax.set_title(SCENARIO_LABELS.get(scen, scen), fontsize=11, fontweight="bold")
        ax.set_xlabel("HyFD Composite Score")
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.3)
        mean_score = sub.mean()
        ax.axvline(mean_score, color=color, linestyle="-", linewidth=2, alpha=0.9)
        ax.text(0.95, 0.92, f"μ={mean_score:.2f}", transform=ax.transAxes,
                ha="right", fontsize=9, color=color, fontweight="bold")

    plt.suptitle("HyFD Composite Score Distributions per Scenario",
                  fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig6_score_distributions.png")
    plt.close()
    print("  ✓ Figure 6: Score distributions")


# ──────────────────────────────────────────────────────────────
# Figure 7: Drift Detection — KS Statistic Visualization
# ──────────────────────────────────────────────────────────────
def fig_drift_visualization():
    rng = np.random.RandomState(42)
    ref = rng.normal(0, 1, 2000)
    no_drift = rng.normal(0.05, 1.05, 500)
    high_drift = rng.normal(2.5, 1.2, 500)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, prod, title, color in zip(
        axes,
        [no_drift, high_drift],
        ["Normal Production (No Drift)", "Drifted Production (Covariate Shift)"],
        ["#27ae60", "#e74c3c"]
    ):
        ax.hist(ref, bins=40, alpha=0.6, color="#1a73e8", label="Reference", density=True)
        ax.hist(prod, bins=40, alpha=0.6, color=color, label="Production", density=True)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Feature Value")
        ax.set_ylabel("Density")
        ax.legend(framealpha=0.9)
        ax.grid(alpha=0.3)

        from scipy import stats
        ks_stat, p_val = stats.ks_2samp(ref, prod)
        ax.text(0.03, 0.93, f"KS={ks_stat:.3f}, p={p_val:.4f}",
                transform=ax.transAxes, fontsize=9.5,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.suptitle("Feature Distribution: Reference vs. Production",
                  fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig7_drift_visualization.png")
    plt.close()
    print("  ✓ Figure 7: Drift visualization")


if __name__ == "__main__":
    import matplotlib.ticker
    print("Generating all figures...")
    fig_system_architecture()
    fig_overall_performance()
    fig_per_scenario_heatmap()
    fig_signal_contribution()
    fig_latency()
    fig_score_distributions()
    fig_drift_visualization()
    print(f"\n✓ All figures saved to: {FIGURES_DIR}/")
