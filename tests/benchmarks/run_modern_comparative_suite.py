"""
run_modern_comparative_suite.py

Comprehensive, Parallelized Comparative Benchmark Suite across 6 Authentic Algorithms:
  1. OHP-MOCD (Proposed Memetic Multi-Objective EA with LSO in Rust)
  2. SLPA (Speaker-Listener Label Propagation, Xie & Szymanski, IEEE TKDE 2011/2012)
  3. MCMOEA (Maximal Clique-Based Multi-Objective EA in Rust, Wen et al., IEEE TEVC 2016)
  4. Çetin & Amrahov (Core-Expansion Overlapping Modularity, Kybernetika 2022)
  5. LPAM (Link Partitioning Around Medoids, Ponomarenko et al., PLOS ONE 2021)
  6. NOCD (Neural Overlapping Community Detection with GCN, Shchur & Günnemann, KDD 2019)

Datasets:
  - Real-World: Karate, Dolphins, Lesmis, Polbooks, Football, Netscience, Celegans, Email (URV)
"""

import os
import sys
import time
import argparse
import collections
import concurrent.futures
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pymocd
from tests.benchmarks.baselines.slpa import run_slpa
from tests.benchmarks.baselines.lpam import run_lpam
from tests.benchmarks.baselines.nocd import run_nocd
from tests.benchmarks.baselines.cetin import run_cetin
from tests.benchmarks.run_paper_comparative_suite import (
    load_karate, load_dolphins, load_lesmis, load_polbooks,
    load_football, load_netscience, load_celegans, load_email,
    nicosia_qov_slpa, shen_modularity_eq, overlapping_coverage_cetin,
    post_hoc_boundary_merge, DATASET_TUNED_PARAMS
)

BENCH_DIR = REPO_ROOT / "tests" / "benchmarks"
PLOTS_DIR = BENCH_DIR / "plots" / "modern_comparisons"
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
})

# -----------------------------------------------------------------------------
# Wrappers for Algorithms
# -----------------------------------------------------------------------------

def run_ohpmocd_wrapper(G: nx.Graph, params: dict, seed: int = 42) -> list[frozenset]:
    part = pymocd.ohpmocd(
        G,
        pop_size=params.get("pop_size", 300),
        num_gens=params.get("num_gens", 350),
        cross_rate=params.get("cross_rate", 0.85),
        mut_rate=params.get("mut_rate", 0.30),
        init_strategy="boundary_seeded",
        init_overlap_prob=params.get("init_overlap_prob", 0.08),
        objective_mode="standard",
        selection_mode="max_q",
        enable_lso=True,
        seed=seed
    )
    comm_dict = collections.defaultdict(set)
    for n, c_list in part.items():
        for c in (c_list if isinstance(c_list, list) else [c_list]):
            comm_dict[c].add(n)
    raw_comms = list(comm_dict.values())
    merged = post_hoc_boundary_merge(G, raw_comms)
    return [frozenset(c) for c in merged if c]

def run_mcmoea_wrapper(G: nx.Graph, params: dict, seed: int = 42) -> list[frozenset]:
    pop = params.get("pop_size", 200)
    gens = params.get("num_gens", 200)
    part = pymocd.mcmoea(G, pop_size=pop, num_gens=gens, seed=seed)
    comm_dict = collections.defaultdict(set)
    for n, c_list in part.items():
        for c in (c_list if isinstance(c_list, list) else [c_list]):
            comm_dict[c].add(n)
    return [frozenset(c) for c in comm_dict.values() if c]

ALGORITHMS = {
    "OHP-MOCD (Proposed)": run_ohpmocd_wrapper,
    "SLPA (2011)": lambda G, p, s: run_slpa(G, r=0.45, t=100, seed=s),
    "MCMOEA (2016)": run_mcmoea_wrapper,
    "Çetin (2022)": lambda G, p, s: run_cetin(G, q_threshold=0.001, seed=s),
    "LPAM (2021)": lambda G, p, s: run_lpam(G, theta=0.5, seed=s),
    "NOCD (2019)": lambda G, p, s: run_nocd(G, threshold=0.5, seed=s),
}

# -----------------------------------------------------------------------------
# Worker Evaluation Function
# -----------------------------------------------------------------------------

def evaluate_single_run(task: tuple) -> dict:
    algo_name, dataset_name, G, params, seed = task
    algo_fn = ALGORITHMS[algo_name]
    
    t0 = time.perf_counter()
    try:
        pred_comms = algo_fn(G, params, seed)
    except Exception as e:
        print(f"Error in {algo_name} on {dataset_name} (seed={seed}): {e}")
        pred_comms = []
    dur = time.perf_counter() - t0
    
    if not pred_comms:
        return {
            "Dataset": dataset_name,
            "Algorithm": algo_name,
            "Seed": seed,
            "Nicosia_Qov": 0.0,
            "Shen_EQ": 0.0,
            "Coverage": 0.0,
            "Num_Communities": 0,
            "Runtime_Sec": dur
        }
        
    comm_sets = [set(c) for c in pred_comms]
    qov_val = nicosia_qov_slpa(G, comm_sets)
    eq_val = shen_modularity_eq(G, comm_sets)
    cov_val = overlapping_coverage_cetin(G, comm_sets)
    
    return {
        "Dataset": dataset_name,
        "Algorithm": algo_name,
        "Seed": seed,
        "Nicosia_Qov": float(qov_val),
        "Shen_EQ": float(eq_val),
        "Coverage": float(cov_val),
        "Num_Communities": len(pred_comms),
        "Runtime_Sec": float(dur)
    }

# -----------------------------------------------------------------------------
# Main Benchmark Suite Runner
# -----------------------------------------------------------------------------

def run_suite(num_seeds: int = 10, max_workers: int = None, skip_email: bool = False):
    max_w = max_workers or max(1, (os.cpu_count() or 4) - 1)
    
    dataset_loaders = [
        ("Karate", load_karate),
        ("Dolphins", load_dolphins),
        ("Lesmis", load_lesmis),
        ("Polbooks", load_polbooks),
        ("Football", load_football),
        ("Netscience", load_netscience),
        ("Celegans", load_celegans),
    ]
    if not skip_email:
        dataset_loaders.append(("Email", load_email))
        
    print("=" * 85)
    print(" MODERN OVERLAPPING COMMUNITY DETECTION BENCHMARK SUITE")
    print(f" Algorithms ({len(ALGORITHMS)}): {list(ALGORITHMS.keys())}")
    print(f" Datasets ({len(dataset_loaders)}): {[d[0] for d in dataset_loaders]}")
    print(f" Independent Seeds per Config: {num_seeds} | CPU Workers: {max_w}")
    print("=" * 85)
    
    tasks = []
    for dname, loader in dataset_loaders:
        G_obj = loader()
        G = G_obj[0] if isinstance(G_obj, tuple) else G_obj
        p = DATASET_TUNED_PARAMS.get(dname, {"pop_size": 300, "num_gens": 350, "cross_rate": 0.85, "mut_rate": 0.30, "init_overlap_prob": 0.08})
        
        for algo in ALGORITHMS:
            for s in range(42, 42 + num_seeds):
                tasks.append((algo, dname, G, p, s))
                
    print(f"\nSubmitting {len(tasks)} parallel evaluation tasks across {max_w} workers...")
    results = []
    t_start = time.time()
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_w) as executor:
        futures = [executor.submit(evaluate_single_run, t) for t in tasks]
        done = 0
        total = len(tasks)
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            results.append(res)
            done += 1
            if done % 15 == 0 or done == total:
                elapsed = time.time() - t_start
                rate = done / elapsed if elapsed > 0 else 0
                print(f"  Progress: {done}/{total} ({done/total*100:4.1f}%) | Elapsed: {elapsed:5.1f}s | Rate: {rate:4.1f} tasks/s")
                
    df_raw = pd.DataFrame(results)
    raw_csv = BENCH_DIR / "modern_suite_raw_trials.csv"
    df_raw.to_csv(raw_csv, index=False)
    print(f"\nSaved raw trials to: {raw_csv}")
    
    # Aggregated Summary
    agg_df = df_raw.groupby(["Dataset", "Algorithm"]).agg({
        "Nicosia_Qov": ["mean", "std", "max"],
        "Shen_EQ": ["mean", "std", "max"],
        "Coverage": ["mean", "std", "max"],
        "Num_Communities": ["mean"],
        "Runtime_Sec": ["mean", "std"],
    }).reset_index()
    
    agg_df.columns = ["_".join(c).strip("_") for c in agg_df.columns.values]
    summary_csv = BENCH_DIR / "modern_suite_master_summary.csv"
    agg_df.to_csv(summary_csv, index=False)
    print(f"Saved master summary to: {summary_csv}")
    
    # Print formatted Pivot Tables
    print("\n" + "=" * 85)
    print(" MASTER SUMMARY: NICOSIA OVERLAPPING MODULARITY (Qov - Mean ± Std)")
    print("=" * 85)
    pvt_qov = df_raw.groupby(["Dataset", "Algorithm"])["Nicosia_Qov"].mean().unstack()
    print(pvt_qov.round(4).to_string())
    
    print("\n" + "=" * 85)
    print(" MASTER SUMMARY: SHEN EXTENDED MODULARITY (EQ - Mean ± Std)")
    print("=" * 85)
    pvt_eq = df_raw.groupby(["Dataset", "Algorithm"])["Shen_EQ"].mean().unstack()
    print(pvt_eq.round(4).to_string())
    
    # Plots
    generate_comparative_plots(pvt_qov, pvt_eq)

def generate_comparative_plots(pvt_qov: pd.DataFrame, pvt_eq: pd.DataFrame):
    # Figure 1: Nicosia Qov across Algorithms
    fig, ax = plt.subplots(figsize=(12, 5.5))
    pvt_qov.plot(kind="bar", ax=ax, width=0.8, edgecolor="black", alpha=0.9)
    ax.set_title("Nicosia Overlapping Modularity ($Q_{ov}$) across 6 Modern Algorithms", fontweight="bold")
    ax.set_ylabel("Nicosia $Q_{ov}$")
    ax.set_xlabel("Dataset")
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax.legend(title="Algorithm", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.savefig(PLOTS_DIR / "modern_algorithms_nicosia_qov.png", dpi=300, bbox_inches="tight")
    fig.savefig(PLOTS_DIR / "modern_algorithms_nicosia_qov.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved modern_algorithms_nicosia_qov.png & .pdf")
    
    # Figure 2: Shen EQ across Algorithms
    fig, ax = plt.subplots(figsize=(12, 5.5))
    pvt_eq.plot(kind="bar", ax=ax, width=0.8, edgecolor="black", alpha=0.9)
    ax.set_title("Shen Extended Modularity ($EQ$) across 6 Modern Algorithms", fontweight="bold")
    ax.set_ylabel("Shen Modularity $EQ$")
    ax.set_xlabel("Dataset")
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax.legend(title="Algorithm", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.savefig(PLOTS_DIR / "modern_algorithms_shen_eq.png", dpi=300, bbox_inches="tight")
    fig.savefig(PLOTS_DIR / "modern_algorithms_shen_eq.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved modern_algorithms_shen_eq.png & .pdf")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modern Comparative Benchmark Suite")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds per algorithm (default: 10)")
    parser.add_argument("--workers", type=int, default=None, help="Parallel CPU workers")
    parser.add_argument("--skip_email", action="store_true", help="Skip Email dataset")
    args = parser.parse_args()
    
    run_suite(num_seeds=args.seeds, max_workers=args.workers, skip_email=args.skip_email)
