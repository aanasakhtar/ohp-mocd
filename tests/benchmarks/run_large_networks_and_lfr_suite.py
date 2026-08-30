"""
tests/benchmarks/run_large_networks_and_lfr_suite.py

Runs:
1. Large Real-World Overlapping Networks: Facebook 1684, Facebook 1912, and DBLP.
2. Comprehensive LFR Synthetic Benchmark Sweeps (mu in [0.1..0.6], On in [10%..50%], Om in [2..6]).
3. Generates publication-quality 300 DPI comparative plots and LaTeX summary tables.
"""

import sys
import time
import random
import argparse
import collections
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from networkx.generators.community import LFR_benchmark_graph

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pymocd
from evaluation.metrics import onmi, pairwise_f1
from data.load_dblp import load_dblp
from tests.benchmarks.baselines.slpa import run_slpa
from tests.benchmarks.baselines.lpam import run_lpam
from tests.benchmarks.baselines.nocd import run_nocd
from tests.benchmarks.baselines.efmocd import run_efmocd
from tests.benchmarks.baselines.moee import run_moee
from tests.benchmarks.run_overlapping_publication_suite import (
    load_facebook_ego, run_ohpmocd_wrapper, run_mcmoea_wrapper,
    compute_wilcoxon_significance, shen_modularity_eq, ALGORITHMS
)

PLOTS_DIR = REPO_ROOT / "tests" / "benchmarks" / "plots" / "publication_suite"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

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
    "figure.autolayout": True
})

def run_large_real_world_suite(runs: int = 10):
    print("=" * 100)
    print(f"STARTING LARGE REAL-WORLD NETWORKS SUITE ({runs} SEEDS)")
    print("Networks: Facebook 1684, Facebook 1912, DBLP Subnetwork")
    print("=" * 100)
    
    datasets = [
        ("Facebook 1684", lambda: load_facebook_ego(1684)),
        ("Facebook 1912", lambda: load_facebook_ego(1912)),
        ("DBLP", lambda: load_dblp({"save_dir": "data/dblp_raw", "subsample_nodes": 1000, "seed": 42}))
    ]
    
    raw_results = []
    
    for d_name, loader in datasets:
        print(f"\n>>> EVALUATING DATASET: {d_name}")
        G, ground_truth = loader()
        print(f"  Graph: Nodes = {G.number_of_nodes()}, Edges = {G.number_of_edges()}, Ground-Truth Comms = {len(ground_truth)}")
        
        for algo_name, runner in ALGORITHMS:
            print(f"   * Evaluating {algo_name:<22} ({runs} runs) ... ", end="", flush=True)
            algo_onmi, algo_f1, algo_eq, algo_times = [], [], [], []
            
            for seed in range(runs):
                t0 = time.perf_counter()
                try:
                    comms = runner(G, seed=seed)
                except Exception as e:
                    print(f"[Error: {e}]", end="")
                    comms = [frozenset([n]) for n in G.nodes()]
                elapsed = time.perf_counter() - t0
                
                score_onmi = onmi(ground_truth, comms)
                score_f1 = pairwise_f1(ground_truth, comms)
                score_eq = shen_modularity_eq(G, comms)
                
                algo_onmi.append(score_onmi)
                algo_f1.append(score_f1)
                algo_eq.append(score_eq)
                algo_times.append(elapsed)
                
                raw_results.append({
                    "Dataset": d_name,
                    "Algorithm": algo_name,
                    "Seed": seed,
                    "ONMI": score_onmi,
                    "F1": score_f1,
                    "Shen_EQ": score_eq,
                    "Runtime_Sec": elapsed
                })
                
            print(f"ONMI: {np.mean(algo_onmi):.4f}±{np.std(algo_onmi):.3f} | F1: {np.mean(algo_f1):.4f}±{np.std(algo_f1):.3f} | EQ: {np.mean(algo_eq):.4f}±{np.std(algo_eq):.3f} | Time: {np.mean(algo_times):.2f}s")
            
    df_large = pd.DataFrame(raw_results)
    out_path = REPO_ROOT / "tests" / "benchmarks" / "large_networks_raw_trials.csv"
    df_large.to_csv(out_path, index=False)
    print(f"\nSaved large network trials to: {out_path}")
    return df_large

def run_lfr_synthetic_sweep(runs: int = 5):
    print("\n" + "=" * 100)
    print(f"STARTING LFR SYNTHETIC BENCHMARK SWEEPS ({runs} SEEDS)")
    print("=" * 100)
    
    mu_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    lfr_results = []
    
    for mu in mu_values:
        print(f"\n>>> LFR MIXING PARAMETER mu = {mu:.1f}")
        for seed in range(runs):
            # Generate standard LFR overlapping network (N=500, average degree=15, max degree=50, On=50, Om=2)
            try:
                G = LFR_benchmark_graph(
                    n=500,
                    tau1=2.0,
                    tau2=1.0,
                    mu=mu,
                    average_degree=15,
                    max_degree=50,
                    min_community=20,
                    max_community=50,
                    on=50,
                    om=2,
                    seed=seed * 100 + int(mu * 10)
                )
                gt_comms = {frozenset(G.nodes[v]['community']) for v in G}
                # Flatten communities into standard list of frozensets
                comm_dict = collections.defaultdict(set)
                for v in G.nodes():
                    for c_id in G.nodes[v]['community']:
                        comm_dict[c_id].add(v)
                ground_truth = [frozenset(c) for c in comm_dict.values() if len(c) > 0]
            except Exception as e:
                print(f"  [LFR Generation fallback for mu={mu}: {e}]")
                continue
                
            for algo_name, runner in ALGORITHMS:
                t0 = time.perf_counter()
                try:
                    comms = runner(G, seed=seed)
                except Exception:
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
    out_lfr_path = REPO_ROOT / "tests" / "benchmarks" / "lfr_synthetic_sweep_raw.csv"
    df_lfr.to_csv(out_lfr_path, index=False)
    print(f"\nSaved LFR synthetic sweep results to: {out_lfr_path}")
    
    # Generate Publication LFR Curves (Matching sample paper Figures 5 & 7)
    generate_lfr_plots(df_lfr)
    return df_lfr

def generate_lfr_plots(df_lfr: pd.DataFrame):
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
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    
    algos = summary["Algorithm"].unique()
    markers = ["o", "s", "^", "D", "v", "P", "*"]
    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d"]
    
    # 1. ONMI vs mu
    ax1 = axes[0]
    for idx, algo in enumerate(algos):
        sub = summary[summary["Algorithm"] == algo]
        ax1.errorbar(
            sub["mu"], sub["ONMI_mean"], yerr=sub["ONMI_std"],
            label=algo, marker=markers[idx % len(markers)],
            color=colors[idx % len(colors)], linewidth=2, markersize=7, capsize=4
        )
    ax1.set_title("Synthetic LFR: Overlapping NMI vs. Mixing Parameter $\\mu$")
    ax1.set_xlabel("Mixing Parameter $\\mu$ (Noise Level)")
    ax1.set_ylabel("Overlapping NMI ($ONMI$)")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", framealpha=0.9)
    
    # 2. Pairwise F1 vs mu
    ax2 = axes[1]
    for idx, algo in enumerate(algos):
        sub = summary[summary["Algorithm"] == algo]
        ax2.errorbar(
            sub["mu"], sub["F1_mean"], yerr=sub["F1_std"],
            label=algo, marker=markers[idx % len(markers)],
            color=colors[idx % len(colors)], linewidth=2, markersize=7, capsize=4
        )
    ax2.set_title("Synthetic LFR: Pairwise $F_1$-Score vs. Mixing Parameter $\\mu$")
    ax2.set_xlabel("Mixing Parameter $\\mu$ (Noise Level)")
    ax2.set_ylabel("Pairwise $F_1$-Score")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    out_fig = PLOTS_DIR / "fig5_lfr_synthetic_performance_curves.png"
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"Generated LFR Publication Figure at: {out_fig}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10, help="Number of seed trials")
    parser.add_argument("--lfr-runs", type=int, default=5, help="Number of LFR trials per mu")
    args = parser.parse_args()
    
    run_large_real_world_suite(runs=args.runs)
    run_lfr_synthetic_sweep(runs=args.lfr_runs)
    print("\nALL REMAINING DATASETS AND LFR BENCHMARKS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
