"""
tests/benchmarks/run_abcd_sweeps_and_plots.py

ABCD+o^2 Synthetic Benchmark Sweeps across background mixing parameter xi in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6].
Evaluates all 7 authentic overlapping community detection algorithms.
Generates publication-quality standalone 300 DPI comparative curves.

Reference:
    Barrett, J., DeWolfe, R., Kamiński, B., Prałat, P., Smith, A., & Théberge, F. (2026).
    The artificial benchmark for community detection with outliers and overlapping communities (ABCD+o^2).
    Journal of Complex Networks, 14(4), cnag023. https://doi.org/10.1093/comnet/cnag023
    Preprint: arXiv:2506.05486.
"""

import sys
import time
import argparse
import collections
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.metrics import onmi, pairwise_f1
from tests.benchmarks.utils.abcd import generate_abcd_benchmark
from tests.benchmarks.run_overlapping_publication_suite import (
    shen_modularity_eq, ALGORITHMS
)

PLOTS_DIR = REPO_ROOT / "tests" / "benchmarks" / "plots" / "publication_suite"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.titlesize": 16,
    "figure.dpi": 300,
    "font.family": "sans-serif",
    "figure.autolayout": True
})

ALGO_COLORS = {
    "OHP-MOCD (Proposed)": "#1b9e77",  # Bold Teal/Green
    "SLPA (2011)": "#d95f02",           # Deep Orange
    "EF-MOCD (2020)": "#7570b3",        # Purple
    "MO-EE (2018)": "#e7298a",          # Magenta/Pink
    "LPAM (2021)": "#e6ab02",           # Gold/Yellow
    "NOCD (2019)": "#386cb0",           # Steel Blue
    "MCMOEA (2016)": "#a6761d",         # Brown/Amber
}

ALGO_MARKERS = {
    "OHP-MOCD (Proposed)": "o",
    "SLPA (2011)": "s",
    "EF-MOCD (2020)": "^",
    "MO-EE (2018)": "D",
    "LPAM (2021)": "v",
    "NOCD (2019)": "P",
    "MCMOEA (2016)": "*",
}


def run_abcd_suite(
    runs: int = 5,
    n_nodes: int = 300,
    eta: float = 1.5,
    tau1: float = 2.5,
    tau2: float = 1.5,
    d_min: int = 5,
    d_max: int = 30,
    c_min: int = 20,
    c_max: int = 60,
    xi_values: list = None
):
    if xi_values is None:
        xi_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    print("=" * 100)
    print(f"STARTING ABCD+o^2 SYNTHETIC BENCHMARK SUITE ({runs} SEEDS PER XI)")
    print(f"Parameters: N={n_nodes}, eta={eta} (overlap multiplier), xi sweep: {xi_values}")
    print("=" * 100)

    abcd_results = []

    for xi in xi_values:
        print(f"\n>>> ABCD+o^2 MIXING PARAMETER xi = {xi:.1f} (eta = {eta:.2f})")
        for seed in range(runs):
            effective_seed = seed * 100 + int(round(xi * 10))
            G, ground_truth, node_to_comms = generate_abcd_benchmark(
                n=n_nodes,
                tau1=tau1,
                tau2=tau2,
                d_min=d_min,
                d_max=d_max,
                c_min=c_min,
                c_max=c_max,
                xi=xi,
                eta=eta,
                seed=effective_seed
            )

            # Node overlap statistics
            memberships = [len(node_to_comms[u]) for u in G.nodes()]
            n_overlap = sum(1 for m in memberships if m > 1)
            pct_overlap = (n_overlap / G.number_of_nodes()) * 100.0 if G.number_of_nodes() > 0 else 0.0

            print(
                f"  [Seed {seed+1}/{runs}] Generated ABCD+o^2: Nodes={G.number_of_nodes()}, "
                f"Edges={G.number_of_edges()}, Comms={len(ground_truth)}, Overlap={n_overlap} nodes ({pct_overlap:.1f}%)"
            )

            for algo_name, runner in ALGORITHMS:
                t0 = time.perf_counter()
                try:
                    comms = runner(G, seed=seed)
                except Exception as e:
                    print(f"    [Error in {algo_name}: {e}]")
                    comms = [frozenset([n]) for n in G.nodes()]
                elapsed = time.perf_counter() - t0

                score_onmi = onmi(ground_truth, comms)
                score_f1 = pairwise_f1(ground_truth, comms)
                score_eq = shen_modularity_eq(G, comms)

                print(f"    - {algo_name:22s} | ONMI: {score_onmi:.4f} | F1: {score_f1:.4f} | EQ: {score_eq:.4f} | Time: {elapsed:.2f}s")

                abcd_results.append({
                    "xi": xi,
                    "eta": eta,
                    "Algorithm": algo_name,
                    "Seed": seed,
                    "ONMI": score_onmi,
                    "F1": score_f1,
                    "Shen_EQ": score_eq,
                    "Runtime_Sec": elapsed
                })

    df_abcd = pd.DataFrame(abcd_results)
    out_csv = REPO_ROOT / "tests" / "benchmarks" / "abcd_synthetic_sweep_raw.csv"
    df_abcd.to_csv(out_csv, index=False)
    print(f"\nSaved raw ABCD+o^2 trial data to: {out_csv}")

    generate_standalone_abcd_plots(df_abcd)
    generate_summary_table(df_abcd)
    return df_abcd


def generate_summary_table(df_abcd: pd.DataFrame):
    if df_abcd.empty:
        return

    summary = df_abcd.groupby(["xi", "Algorithm"]).agg(
        ONMI_mean=("ONMI", "mean"),
        ONMI_std=("ONMI", "std"),
        F1_mean=("F1", "mean"),
        F1_std=("F1", "std"),
        EQ_mean=("Shen_EQ", "mean"),
        EQ_std=("Shen_EQ", "std"),
        Runtime_mean=("Runtime_Sec", "mean"),
        Runtime_std=("Runtime_Sec", "std")
    ).reset_index()

    out_summary_csv = REPO_ROOT / "tests" / "benchmarks" / "abcd_synthetic_benchmark_master_results.csv"
    summary.to_csv(out_summary_csv, index=False)
    print(f"Saved aggregated ABCD+o^2 summary to: {out_summary_csv}")


def generate_standalone_abcd_plots(df_abcd: pd.DataFrame):
    if df_abcd.empty:
        return

    summary = df_abcd.groupby(["xi", "Algorithm"]).agg(
        ONMI_mean=("ONMI", "mean"),
        ONMI_std=("ONMI", "std"),
        F1_mean=("F1", "mean"),
        F1_std=("F1", "std"),
        EQ_mean=("Shen_EQ", "mean"),
        EQ_std=("Shen_EQ", "std")
    ).reset_index()

    algos = [a for a in ALGO_COLORS.keys() if a in summary["Algorithm"].unique()]

    # 1. Figure: Standalone ONMI vs xi
    plt.figure(figsize=(9, 6))
    for algo in algos:
        sub = summary[summary["Algorithm"] == algo]
        plt.errorbar(
            sub["xi"], sub["ONMI_mean"], yerr=sub["ONMI_std"],
            label=algo, color=ALGO_COLORS.get(algo, "#333333"),
            marker=ALGO_MARKERS.get(algo, "o"), linewidth=2.5,
            markersize=8, capsize=5, elinewidth=1.5
        )
    plt.title("Synthetic ABCD+$o^2$ Benchmark: Overlapping NMI vs. Topological Noise ($\\xi$)", pad=15)
    plt.xlabel("Mixing Parameter $\\xi$ (Noise Fraction)")
    plt.ylabel("Overlapping NMI ($ONMI$)")
    plt.ylim(-0.02, 1.02)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right", framealpha=0.95, edgecolor="gray")
    out_onmi = PLOTS_DIR / "fig_abcd_onmi_vs_xi.png"
    plt.savefig(out_onmi, dpi=300)
    plt.close()
    print(f"Generated standalone figure: {out_onmi}")

    # 2. Figure: Standalone Pairwise F1 vs xi
    plt.figure(figsize=(9, 6))
    for algo in algos:
        sub = summary[summary["Algorithm"] == algo]
        plt.errorbar(
            sub["xi"], sub["F1_mean"], yerr=sub["F1_std"],
            label=algo, color=ALGO_COLORS.get(algo, "#333333"),
            marker=ALGO_MARKERS.get(algo, "s"), linewidth=2.5,
            markersize=8, capsize=5, elinewidth=1.5
        )
    plt.title("Synthetic ABCD+$o^2$ Benchmark: Pairwise $F_1$-Score vs. Topological Noise ($\\xi$)", pad=15)
    plt.xlabel("Mixing Parameter $\\xi$ (Noise Fraction)")
    plt.ylabel("Pairwise $F_1$-Score")
    plt.ylim(-0.02, 1.02)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right", framealpha=0.95, edgecolor="gray")
    out_f1 = PLOTS_DIR / "fig_abcd_f1_vs_xi.png"
    plt.savefig(out_f1, dpi=300)
    plt.close()
    print(f"Generated standalone figure: {out_f1}")

    # 3. Figure: Standalone Extended Modularity EQ vs xi
    plt.figure(figsize=(9, 6))
    for algo in algos:
        sub = summary[summary["Algorithm"] == algo]
        plt.errorbar(
            sub["xi"], sub["EQ_mean"], yerr=sub["EQ_std"],
            label=algo, color=ALGO_COLORS.get(algo, "#333333"),
            marker=ALGO_MARKERS.get(algo, "^"), linewidth=2.5,
            markersize=8, capsize=5, elinewidth=1.5
        )
    plt.title("Synthetic ABCD+$o^2$ Benchmark: Extended Modularity ($EQ$) vs. Topological Noise ($\\xi$)", pad=15)
    plt.xlabel("Mixing Parameter $\\xi$ (Noise Fraction)")
    plt.ylabel("Shen Extended Modularity ($EQ$)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right", framealpha=0.95, edgecolor="gray")
    out_eq = PLOTS_DIR / "fig_abcd_eq_modularity_vs_xi.png"
    plt.savefig(out_eq, dpi=300)
    plt.close()
    print(f"Generated standalone figure: {out_eq}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ABCD+o^2 Synthetic Benchmark Sweeps")
    parser.add_argument("--runs", type=int, default=5, help="Number of seeds per noise level (default: 5)")
    parser.add_argument("--nodes", type=int, default=300, help="Number of nodes (default: 300)")
    parser.add_argument("--eta", type=float, default=1.5, help="Overlap multiplier (default: 1.5)")
    args = parser.parse_args()

    run_abcd_suite(runs=args.runs, n_nodes=args.nodes, eta=args.eta)
