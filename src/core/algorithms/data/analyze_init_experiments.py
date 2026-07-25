"""
analyze_init_experiments.py — Statistical Analysis and Publication-Quality Plot Generator.

Reads:
  - src/core/algorithms/data/init_strategy_summary.csv
  - src/core/algorithms/data/init_strategy_convergence.csv

Outputs:
  - src/core/algorithms/data/plots/*.png & *.pdf
  - src/core/algorithms/data/plots/statistical_tests.csv
  - src/core/algorithms/data/plots/statistical_report.txt
"""

import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Paths
DATA_DIR = Path(__file__).parent
PLOTS_DIR = DATA_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Color Palette for Consistent Branding (Pure Matplotlib, No Seaborn)
STRATEGY_COLORS = {
    "crisp": "#2b5c8f",          # Deep Slate Blue
    "random_overlap": "#d95f02", # Terracotta Orange
    "boundary_seeded": "#1b9e77",# Emerald Green
}

STRATEGY_LABELS = {
    "crisp": "Crisp (Baseline)",
    "random_overlap": "Random Overlap",
    "boundary_seeded": "Boundary Seeded",
}

# Global Matplotlib Formatting for Research Papers
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "font.family": "sans-serif",
})


def compute_cliffs_delta(x, y):
    """Computes Cliff's Delta effect size between two sample vectors."""
    n_x = len(x)
    n_y = len(y)
    if n_x == 0 or n_y == 0:
        return 0.0

    greater = 0
    lesser = 0
    for xi in x:
        for yj in y:
            if xi > yj:
                greater += 1
            elif xi < yj:
                lesser += 1
    return (greater - lesser) / (n_x * n_y)


def compute_cohens_d(x, y):
    """Computes Cohen's d effect size between two sample vectors."""
    n_x, n_y = len(x), len(y)
    if n_x < 2 or n_y < 2:
        return 0.0
    mean_x, mean_y = np.mean(x), np.mean(y)
    var_x, var_y = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled_std = math.sqrt(((n_x - 1) * var_x + (n_y - 1) * var_y) / (n_x + n_y - 2))
    if pooled_std == 0:
        return 0.0
    return (mean_x - mean_y) / pooled_std


def save_figure(fig, name):
    """Saves figure in both PNG and PDF formats."""
    png_path = PLOTS_DIR / f"{name}.png"
    pdf_path = PLOTS_DIR / f"{name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot Saved] {png_path.name} & {pdf_path.name}")


def plot_convergence_curves(df_conv):
    """Figure 1: Convergence curves (mean +/- std ribbon) of best Q per generation for each dataset."""
    datasets = df_conv["dataset"].unique()
    for ds in datasets:
        df_ds = df_conv[df_conv["dataset"] == ds]
        fig, ax = plt.subplots(figsize=(7, 4.5))

        for strat in ["crisp", "random_overlap", "boundary_seeded"]:
            df_s = df_ds[df_ds["strategy"] == strat]
            if df_s.empty:
                continue

            grouped = df_s.groupby("generation")["best_Q"]
            gens = grouped.mean().index
            means = grouped.mean().values
            stds = grouped.std().values

            color = STRATEGY_COLORS[strat]
            label = STRATEGY_LABELS[strat]

            ax.plot(gens, means, label=label, color=color, linewidth=2)
            ax.fill_between(gens, means - stds, means + stds, color=color, alpha=0.15)

        ax.set_title(f"Convergence History: {ds}", fontweight="bold")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Modularity Q (1 - intra - inter)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower right")
        save_figure(fig, f"convergence_curves_{ds}")


def plot_final_q_boxplots(df_summary):
    """Figure 2: Boxplots of final Q by initialization strategy for each dataset."""
    datasets = df_summary["dataset"].unique()
    for ds in datasets:
        df_ds = df_summary[df_summary["dataset"] == ds]
        fig, ax = plt.subplots(figsize=(6.5, 4.5))

        strategies = ["crisp", "random_overlap", "boundary_seeded"]
        data = [df_ds[df_ds["strategy"] == s]["max_Q"].values for s in strategies]
        labels = [STRATEGY_LABELS[s] for s in strategies]
        colors = [STRATEGY_COLORS[s] for s in strategies]

        bp = ax.boxplot(data, patch_artist=True, labels=labels, tick_labels=labels, widths=0.45)

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        for median in bp["medians"]:
            median.set(color="black", linewidth=2)

        ax.set_title(f"Final Modularity (Q) Distribution: {ds}", fontweight="bold")
        ax.set_ylabel("Modularity Q")
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")
        save_figure(fig, f"boxplot_final_q_{ds}")


def plot_grouped_ground_truth_metrics(df_summary):
    """Figure 3: Grouped bar charts (mean +/- std) for ground-truth metrics (NMI, AMI, ARI)."""
    datasets = df_summary["dataset"].unique()
    metrics = ["nmi", "ami", "ari"]
    metric_labels = ["NMI", "AMI", "ARI"]
    strategies = ["crisp", "random_overlap", "boundary_seeded"]

    for ds in datasets:
        df_ds = df_summary[df_summary["dataset"] == ds]
        fig, ax = plt.subplots(figsize=(7.5, 4.5))

        x = np.arange(len(metrics))
        width = 0.25

        for idx, strat in enumerate(strategies):
            df_s = df_ds[df_ds["strategy"] == strat]
            means = [df_s[m].mean() for m in metrics]
            stds = [df_s[m].std() for m in metrics]

            offset = (idx - 1) * width
            ax.bar(
                x + offset,
                means,
                width,
                yerr=stds,
                capsize=4,
                label=STRATEGY_LABELS[strat],
                color=STRATEGY_COLORS[strat],
                edgecolor="black",
                linewidth=0.8,
            )

        ax.set_title(f"Ground-Truth Agreement Metrics: {ds}", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
        ax.set_ylabel("Score")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")
        ax.legend(loc="upper right")
        save_figure(fig, f"grouped_gt_metrics_{ds}")


def plot_runtime_boxplots(df_summary):
    """Figure 4: Boxplots of runtime (execution time in milliseconds)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    datasets = df_summary["dataset"].unique()
    strategies = ["crisp", "random_overlap", "boundary_seeded"]

    positions = []
    data = []
    colors = []

    pos = 1
    for ds in datasets:
        df_ds = df_summary[df_summary["dataset"] == ds]
        for strat in strategies:
            vals = df_ds[df_ds["strategy"] == strat]["runtime_ms"].values
            data.append(vals)
            positions.append(pos)
            colors.append(STRATEGY_COLORS[strat])
            pos += 1
        pos += 1

    bp = ax.boxplot(data, positions=positions, patch_artist=True, widths=0.6)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for median in bp["medians"]:
        median.set(color="black", linewidth=2)

    # Set x-ticks at center of dataset groups
    group_centers = [2, 6, 10]
    ax.set_xticks(group_centers[:len(datasets)])
    ax.set_xticklabels(datasets)
    ax.set_title("Runtime Comparison across Strategies & Datasets", fontweight="bold")
    ax.set_ylabel("Execution Time (ms)")
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    # Custom Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=STRATEGY_COLORS[s], label=STRATEGY_LABELS[s], alpha=0.7) for s in strategies]
    ax.legend(handles=legend_elements, loc="upper left")

    save_figure(fig, "boxplot_runtime_all")


def plot_overlap_stats_bars(df_summary):
    """Figure 5: Bar charts of average memberships per node and number of overlapping nodes."""
    datasets = df_summary["dataset"].unique()
    strategies = ["crisp", "random_overlap", "boundary_seeded"]

    for ds in datasets:
        df_ds = df_summary[df_summary["dataset"] == ds]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

        x = np.arange(len(strategies))
        strat_display = [STRATEGY_LABELS[s] for s in strategies]
        colors = [STRATEGY_COLORS[s] for s in strategies]

        avg_m_means = [df_ds[df_ds["strategy"] == s]["avg_memberships"].mean() for s in strategies]
        avg_m_stds = [df_ds[df_ds["strategy"] == s]["avg_memberships"].std() for s in strategies]

        num_o_means = [df_ds[df_ds["strategy"] == s]["num_overlapping_nodes"].mean() for s in strategies]
        num_o_stds = [df_ds[df_ds["strategy"] == s]["num_overlapping_nodes"].std() for s in strategies]

        # Ax1: Average Memberships per Node
        ax1.bar(x, avg_m_means, yerr=avg_m_stds, capsize=4, color=colors, edgecolor="black", linewidth=0.8)
        ax1.set_xticks(x)
        ax1.set_xticklabels(strat_display, rotation=15)
        ax1.set_ylabel("Average Memberships per Node")
        ax1.set_title(f"Average Memberships ({ds})", fontweight="bold")
        ax1.grid(True, linestyle="--", alpha=0.4, axis="y")

        # Ax2: Number of Overlapping Nodes
        ax2.bar(x, num_o_means, yerr=num_o_stds, capsize=4, color=colors, edgecolor="black", linewidth=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(strat_display, rotation=15)
        ax2.set_ylabel("Number of Overlapping Nodes")
        ax2.set_title(f"Overlapping Node Count ({ds})", fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.4, axis="y")

        save_figure(fig, f"overlap_stats_bars_{ds}")


def plot_memberships_vs_q_scatter(df_summary):
    """Figure 6: Scatter plot of average memberships vs final Q, colored by strategy."""
    fig, ax = plt.subplots(figsize=(7, 5))
    strategies = ["crisp", "random_overlap", "boundary_seeded"]

    for strat in strategies:
        df_s = df_summary[df_summary["strategy"] == strat]
        ax.scatter(
            df_s["avg_memberships"],
            df_s["max_Q"],
            label=STRATEGY_LABELS[strat],
            color=STRATEGY_COLORS[strat],
            alpha=0.7,
            edgecolors="none",
            s=40,
        )

    ax.set_title("Average Memberships vs Final Modularity (Q)", fontweight="bold")
    ax.set_xlabel("Average Memberships per Node")
    ax.set_ylabel("Final Modularity Q")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left")
    save_figure(fig, "scatter_memberships_vs_q")


def plot_runtime_vs_q_scatter(df_summary):
    """Figure 7: Scatter plot of runtime vs final Q, colored by strategy."""
    fig, ax = plt.subplots(figsize=(7, 5))
    strategies = ["crisp", "random_overlap", "boundary_seeded"]

    for strat in strategies:
        df_s = df_summary[df_summary["strategy"] == strat]
        ax.scatter(
            df_s["runtime_ms"],
            df_s["max_Q"],
            label=STRATEGY_LABELS[strat],
            color=STRATEGY_COLORS[strat],
            alpha=0.7,
            edgecolors="none",
            s=40,
        )

    ax.set_title("Execution Time vs Final Modularity (Q)", fontweight="bold")
    ax.set_xlabel("Execution Time (ms)")
    ax.set_ylabel("Final Modularity Q")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right")
    save_figure(fig, "scatter_runtime_vs_q")


def plot_pareto_objective_scatter(df_summary):
    """Figure 8: Pareto objective space scatter plot (intra vs inter)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    strategies = ["crisp", "random_overlap", "boundary_seeded"]

    for strat in strategies:
        df_s = df_summary[df_summary["strategy"] == strat]
        ax.scatter(
            df_s["intra"],
            df_s["inter"],
            label=STRATEGY_LABELS[strat],
            color=STRATEGY_COLORS[strat],
            alpha=0.7,
            edgecolors="none",
            s=40,
        )

    ax.set_title("Objective Space: Intra-Modularity vs Inter-Modularity", fontweight="bold")
    ax.set_xlabel("Intra-Modularity Objective (f1, lower is better)")
    ax.set_ylabel("Inter-Modularity Objective (f2, lower is better)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right")
    save_figure(fig, "scatter_pareto_objectives")


def plot_correlation_heatmap(df_summary):
    """Figure 9: Pure Matplotlib Correlation Heatmap for Numeric Metrics."""
    numeric_cols = ["max_Q", "runtime_ms", "nmi", "ami", "ari", "avg_memberships", "num_overlapping_nodes"]
    col_labels = ["Q", "Runtime", "NMI", "AMI", "ARI", "Avg Mem", "Overlap Count"]

    corr = df_summary[numeric_cols].corr().values
    n = len(numeric_cols)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticklabels(col_labels)

    # Annotate correlation values in cells
    for i in range(n):
        for j in range(n):
            val = corr[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9)

    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Metric Correlation Heatmap", fontweight="bold")
    save_figure(fig, "correlation_heatmap")


def perform_statistical_tests(df_summary):
    """Performs Friedman test, Wilcoxon signed-rank post-hoc tests with Holm-Bonferroni correction, and effect sizes."""
    datasets = df_summary["dataset"].unique()
    strategies = ["crisp", "random_overlap", "boundary_seeded"]
    pairs = [("crisp", "random_overlap"), ("crisp", "boundary_seeded"), ("random_overlap", "boundary_seeded")]

    results = []
    report_lines = []

    report_lines.append("==========================================================================")
    report_lines.append("OHP-MOCD INITIALIZATION STRATEGY STATISTICAL SIGNIFICANCE REPORT")
    report_lines.append("==========================================================================")
    report_lines.append("")

    for ds in datasets:
        df_ds = df_summary[df_summary["dataset"] == ds]
        report_lines.append(f"--------------------------------------------------------------------------")
        report_lines.append(f"DATASET: {ds}")
        report_lines.append(f"--------------------------------------------------------------------------")

        for metric in ["max_Q", "nmi", "ami", "ari", "runtime_ms"]:
            samples = [df_ds[df_ds["strategy"] == s][metric].values for s in strategies]
            
            # Friedman Test
            stat_f, p_f = stats.friedmanchisquare(*samples)
            report_lines.append(f"\n[Metric: {metric}]")
            report_lines.append(f"  Friedman Test: Statistic = {stat_f:.4f}, p-value = {p_f:.4e}")

            # Pairwise Wilcoxon tests
            p_vals = []
            pair_records = []
            for s1, s2 in pairs:
                v1 = df_ds[df_ds["strategy"] == s1][metric].values
                v2 = df_ds[df_ds["strategy"] == s2][metric].values
                res_w = stats.wilcoxon(v1, v2)
                p_val = res_w.pvalue
                p_vals.append(p_val)

                delta = compute_cliffs_delta(v1, v2)
                d = compute_cohens_d(v1, v2)

                pair_records.append({
                    "dataset": ds,
                    "metric": metric,
                    "comparison": f"{s1}_vs_{s2}",
                    "raw_p_value": p_val,
                    "cliffs_delta": delta,
                    "cohens_d": d,
                })

            # Holm-Bonferroni correction
            sorted_indices = np.argsort(p_vals)
            m_tests = len(p_vals)
            adjusted_p = [0.0] * m_tests

            for rank, idx in enumerate(sorted_indices):
                adjusted_p[idx] = min(1.0, p_vals[idx] * (m_tests - rank))

            for idx, ((s1, s2), rec) in enumerate(zip(pairs, pair_records)):
                adj_p = adjusted_p[idx]
                rec["adjusted_p_value"] = adj_p
                rec["significant_alpha_005"] = (adj_p < 0.05)
                results.append(rec)

                report_lines.append(
                    f"  Pair ({s1} vs {s2}): p_raw={rec['raw_p_value']:.4e}, "
                    f"p_adj={adj_p:.4e}, Sig={adj_p < 0.05}, "
                    f"Cliff's Delta={rec['cliffs_delta']:.4f}, Cohen's d={rec['cohens_d']:.4f}"
                )

    # Export CSV and Text Report
    df_res = pd.DataFrame(results)
    csv_out = PLOTS_DIR / "statistical_tests.csv"
    df_res.to_csv(csv_out, index=False)
    print(f"\n  [Stats CSV] Saved to: {csv_out}")

    txt_out = PLOTS_DIR / "statistical_report.txt"
    with open(txt_out, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"  [Stats Report] Saved to: {txt_out}")


def main():
    print("=" * 70)
    print("RUNNING STATISTICAL ANALYSIS AND FIGURE GENERATION")
    print("=" * 70)

    summary_file = DATA_DIR / "init_strategy_summary.csv"
    conv_file = DATA_DIR / "init_strategy_convergence.csv"

    if not summary_file.exists() or not conv_file.exists():
        print("Error: Input CSV files missing!")
        return

    df_summary = pd.read_csv(summary_file)
    df_conv = pd.read_csv(conv_file)

    print("\nGenerating Figures...")
    plot_convergence_curves(df_conv)
    plot_final_q_boxplots(df_summary)
    plot_grouped_ground_truth_metrics(df_summary)
    plot_runtime_boxplots(df_summary)
    plot_overlap_stats_bars(df_summary)
    plot_memberships_vs_q_scatter(df_summary)
    plot_runtime_vs_q_scatter(df_summary)
    plot_pareto_objective_scatter(df_summary)
    plot_correlation_heatmap(df_summary)

    print("\nExecuting Statistical Significance Tests...")
    perform_statistical_tests(df_summary)

    print("\nALL ANALYSIS & FIGURE GENERATION COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
