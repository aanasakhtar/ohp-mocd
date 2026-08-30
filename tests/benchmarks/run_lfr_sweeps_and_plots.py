"""
tests/benchmarks/run_lfr_sweeps_and_plots.py

LFR Synthetic Benchmark Sweeps across mixing parameter mu in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6].
Evaluates all 7 authentic overlapping community detection algorithms.
Generates publication-quality standalone 300 DPI comparative curves.
"""

import sys
import time
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
from tests.benchmarks.utils.lfr import generate_lfr_benchmark
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

def run_lfr_suite(runs: int = 5):
    print("=" * 100)
    print(f"STARTING LFR SYNTHETIC BENCHMARK SUITE ({runs} SEEDS PER MU)")
    print("Parameter Sweep: mu in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]")
    print("=" * 100)
    
    mu_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    lfr_results = []
    
    for mu in mu_values:
        print(f"\n>>> LFR MIXING PARAMETER mu = {mu:.1f}")
        for seed in range(runs):
            G, _ = generate_lfr_benchmark(
                n=300,
                tau1=2.5,
                tau2=1.5,
                mu=mu,
                average_degree=12,
                min_community=20,
                seed=seed * 100 + int(mu * 10)
            )
            ground_truth = list({frozenset(G.nodes[u]['community']) for u in G.nodes()})
            print(f"  [Seed {seed+1}/{runs}] Generated LFR graph: Nodes={G.number_of_nodes()}, Edges={G.number_of_edges()}, Ground-Truth Comms={len(ground_truth)}")
            
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
                
                lfr_results.append({
                    "mu": mu,
                    "Algorithm": algo_name,
                    "Seed": seed,
                    "ONMI": score_onmi,
                    "F1": score_f1,
                    "Shen_EQ": score_eq,
                    "Runtime_Sec": elapsed
                })
                
    df_lfr = pd.DataFrame(lfr_results)
    out_csv = REPO_ROOT / "tests" / "benchmarks" / "lfr_synthetic_sweep_raw.csv"
    df_lfr.to_csv(out_csv, index=False)
    print(f"\nSaved raw LFR trial data to: {out_csv}")
    
    generate_standalone_lfr_plots(df_lfr)
    return df_lfr

def generate_standalone_lfr_plots(df_lfr: pd.DataFrame):
    if df_lfr.empty:
        return
        
    summary = df_lfr.groupby(["mu", "Algorithm"]).agg(
        ONMI_mean=("ONMI", "mean"),
        ONMI_std=("ONMI", "std"),
        F1_mean=("F1", "mean"),
        F1_std=("F1", "std"),
        EQ_mean=("Shen_EQ", "mean"),
        EQ_std=("Shen_EQ", "std")
    ).reset_index()
    
    algos = [a for a in ALGO_COLORS.keys() if a in summary["Algorithm"].unique()]
    
    # 1. Figure 5: Standalone ONMI vs mu
    plt.figure(figsize=(9, 6))
    for algo in algos:
        sub = summary[summary["Algorithm"] == algo]
        plt.errorbar(
            sub["mu"], sub["ONMI_mean"], yerr=sub["ONMI_std"],
            label=algo, color=ALGO_COLORS.get(algo, "#333333"),
            marker=ALGO_MARKERS.get(algo, "o"), linewidth=2.5,
            markersize=8, capsize=5, elinewidth=1.5
        )
    plt.title("Synthetic LFR Benchmark: Overlapping NMI vs. Topological Noise ($\\mu$)", pad=15)
    plt.xlabel("Mixing Parameter $\\mu$ (Noise Fraction)")
    plt.ylabel("Overlapping NMI ($ONMI$)")
    plt.ylim(-0.02, 1.02)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right", framealpha=0.95, edgecolor="gray")
    out_onmi = PLOTS_DIR / "fig5_lfr_onmi_vs_mu.png"
    plt.savefig(out_onmi, dpi=300)
    plt.close()
    print(f"Generated standalone figure: {out_onmi}")

    # 2. Figure 6: Standalone Pairwise F1 vs mu
    plt.figure(figsize=(9, 6))
    for algo in algos:
        sub = summary[summary["Algorithm"] == algo]
        plt.errorbar(
            sub["mu"], sub["F1_mean"], yerr=sub["F1_std"],
            label=algo, color=ALGO_COLORS.get(algo, "#333333"),
            marker=ALGO_MARKERS.get(algo, "s"), linewidth=2.5,
            markersize=8, capsize=5, elinewidth=1.5
        )
    plt.title("Synthetic LFR Benchmark: Pairwise $F_1$-Score vs. Topological Noise ($\\mu$)", pad=15)
    plt.xlabel("Mixing Parameter $\\mu$ (Noise Fraction)")
    plt.ylabel("Pairwise $F_1$-Score")
    plt.ylim(-0.02, 1.02)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right", framealpha=0.95, edgecolor="gray")
    out_f1 = PLOTS_DIR / "fig6_lfr_f1_vs_mu.png"
    plt.savefig(out_f1, dpi=300)
    plt.close()
    print(f"Generated standalone figure: {out_f1}")

    # 3. Figure 7: Standalone Extended Modularity EQ vs mu
    plt.figure(figsize=(9, 6))
    for algo in algos:
        sub = summary[summary["Algorithm"] == algo]
        plt.errorbar(
            sub["mu"], sub["EQ_mean"], yerr=sub["EQ_std"],
            label=algo, color=ALGO_COLORS.get(algo, "#333333"),
            marker=ALGO_MARKERS.get(algo, "^"), linewidth=2.5,
            markersize=8, capsize=5, elinewidth=1.5
        )
    plt.title("Synthetic LFR Benchmark: Extended Modularity ($EQ$) vs. Topological Noise ($\\mu$)", pad=15)
    plt.xlabel("Mixing Parameter $\\mu$ (Noise Fraction)")
    plt.ylabel("Shen Extended Modularity ($EQ$)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right", framealpha=0.95, edgecolor="gray")
    out_eq = PLOTS_DIR / "fig7_lfr_eq_modularity_vs_mu.png"
    plt.savefig(out_eq, dpi=300)
    plt.close()
    print(f"Generated standalone figure: {out_eq}")

if __name__ == "__main__":
    run_lfr_suite(runs=5)
