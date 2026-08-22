"""
run_lfr_comprehensive_suite.py

Parallelized, Rigorous Comparative Benchmark Suite on Synthetic LFR Overlapping Networks:
Evaluates 3 Fully-Implemented Published Algorithms:
  1. OHP-MOCD (Proposed Memetic Multi-Objective Evolutionary Algorithm with LSO in Rust)
  2. SLPA (Speaker-Listener Label Propagation with Exact Uniform Tie-Breaking, Xie & Szymanski, IEEE TKDE 2011/2012)
  3. MCMOEA (Maximal Clique-Based Multi-Objective Evolutionary Algorithm in Rust, Wen et al., IEEE TEVC 2016)

Experimental Benchmarks:
  - Experiment 1: Mixing Parameter mu Sweep (mu in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
  - Experiment 2: Overlapping Nodes On Sweep (On in [50, 100, 200, 300, 400, 500])
  - Experiment 3: Number of Memberships Om Sweep (Om in [2, 3, 4, 5, 6])
  - Experiment 4: Standard Published Configs (LFR0, LFR1, LFR2, LFR3 from Literature)

Metrics:
  - Ground-Truth Recovery: gNMI (Overlapping NMI) & Pairwise F1
  - Modularity Quality: Shen Extended Modularity (EQ) & Nicosia Overlapping Modularity (Qov)
  - Computational Efficiency: Execution Time (seconds)
"""

import os
import sys
import time
import json
import random
import argparse
import collections
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from networkx.generators.community import LFR_benchmark_graph

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pymocd
from evaluation.metrics import onmi, pairwise_f1
from tests.benchmarks.run_paper_comparative_suite import (
    nicosia_qov_slpa, shen_modularity_eq, post_hoc_boundary_merge
)

BENCH_DIR = REPO_ROOT / "tests" / "benchmarks"
PLOTS_DIR = BENCH_DIR / "plots" / "lfr_comparisons"
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
# LFR Benchmark Network Generator (Lancichinetti-Fortunato-Radicchi Overlapping)
# -----------------------------------------------------------------------------

def generate_lfr_overlapping(
    n: int = 1000,
    mu: float = 0.2,
    overlap_n: int = 100,
    overlap_m: int = 2,
    avg_degree: int = 15,
    max_degree: int = 50,
    min_community: int = 20,
    max_community: int = 50,
    seed: int = 42
) -> tuple[nx.Graph, list[frozenset]]:
    """Generates synthetic LFR network with controlled mixing parameter mu,
    overlapping nodes On, and memberships per overlapping node Om.
    """
    G_raw = LFR_benchmark_graph(
        n=n,
        tau1=3.0,
        tau2=1.5,
        mu=mu,
        average_degree=avg_degree,
        max_degree=max_degree,
        min_community=min_community,
        max_community=max_community,
        seed=seed
    )
    
    # Extract disjoint base communities
    base_comms = list({frozenset(G_raw.nodes[v]["community"]) for v in G_raw})
    community_sets = [set(c) for c in base_comms]
    
    node_memberships = {
        v: [cid] for cid, c in enumerate(community_sets) for v in c
    }
    
    nodes_by_degree = sorted(G_raw.nodes(), key=lambda v: G_raw.degree(v), reverse=True)
    overlap_count = min(overlap_n, len(nodes_by_degree))
    rng = random.Random(seed)
    
    for node in nodes_by_degree[:overlap_count]:
        current = set(node_memberships[node])
        extra_needed = overlap_m - len(current)
        if extra_needed <= 0:
            continue
            
        nbr_support = collections.Counter()
        for nbr in G_raw.neighbors(node):
            nbr_support.update(cid for cid in node_memberships[nbr] if cid not in current)
            
        while extra_needed > 0:
            target = None
            for cid, _ in nbr_support.most_common():
                if cid not in current:
                    target = cid
                    break
            if target is None:
                available = [cid for cid in range(len(community_sets)) if cid not in current]
                if not available:
                    break
                target = rng.choice(available)
                
            current.add(target)
            node_memberships[node].append(target)
            community_sets[target].add(node)
            extra_needed -= 1
            
    G = nx.Graph(G_raw)
    gt_communities = [frozenset(members) for members in community_sets if members]
    return G, gt_communities

# -----------------------------------------------------------------------------
# Published Baseline Algorithms (Strictly Faithful Implementations)
# -----------------------------------------------------------------------------

def run_slpa(G: nx.Graph, r: float = 0.45, t: int = 100, seed: int = 42) -> list[frozenset]:
    """SLPA: Speaker-Listener Label Propagation Algorithm (Xie & Szymanski, IEEE TKDE 2011/2012).
    Exact multi-agent memory buffer propagation with uniform random tie-breaking.
    """
    rng = random.Random(seed)
    nodes = list(G.nodes())
    memory = {v: [v] for v in nodes}
    for _ in range(t):
        order = list(nodes)
        rng.shuffle(order)
        for listener in order:
            neighbors = list(G.neighbors(listener))
            if not neighbors:
                continue
            speakers_labels = [rng.choice(memory[speaker]) for speaker in neighbors]
            counts = collections.Counter(speakers_labels)
            max_c = max(counts.values())
            candidates = [l for l, c in counts.items() if c == max_c]
            chosen_label = rng.choice(candidates)
            memory[listener].append(chosen_label)
            
    communities = collections.defaultdict(set)
    for v in nodes:
        total = len(memory[v])
        counts = collections.Counter(memory[v])
        for l, cnt in counts.items():
            if (cnt / total) >= r:
                communities[l].add(v)
    return [frozenset(c) for c in communities.values() if c]

def run_mcmoea(G: nx.Graph, pop_size: int = 200, num_gens: int = 200, seed: int = 42) -> list[frozenset]:
    """MCMOEA: Maximal Clique-Based Multi-Objective Evolutionary Algorithm (Wen et al., IEEE TEVC 2016).
    Native Rust implementation via Bron-Kerbosch maximal clique graph transformation.
    """
    part = pymocd.mcmoea(G, pop_size=pop_size, num_gens=num_gens, seed=seed)
    comm_dict = collections.defaultdict(set)
    for n, c_list in part.items():
        for c in (c_list if isinstance(c_list, list) else [c_list]):
            comm_dict[c].add(n)
    return [frozenset(c) for c in comm_dict.values() if c]

def run_ohpmocd_proposed(G: nx.Graph, pop_size: int = 300, num_gens: int = 350, seed: int = 42) -> list[frozenset]:
    """OHP-MOCD (Proposed): Memetic Multi-Objective Overlapping Evolutionary Algorithm with Parameter-Free LSO."""
    part = pymocd.ohpmocd(
        G,
        pop_size=pop_size,
        num_gens=num_gens,
        cross_rate=0.85,
        mut_rate=0.30,
        init_strategy="boundary_seeded",
        init_overlap_prob=0.08,
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

ALGORITHMS = {
    "OHP-MOCD (Proposed)": run_ohpmocd_proposed,
    "SLPA (2011)": run_slpa,
    "MCMOEA (2016)": run_mcmoea,
}

# -----------------------------------------------------------------------------
# Worker Evaluation Function
# -----------------------------------------------------------------------------

def evaluate_single_trial(task_tuple: tuple) -> dict:
    algo_name, exp_name, param_val, G, gt, seed = task_tuple
    algo_fn = ALGORITHMS[algo_name]
    
    t0 = time.perf_counter()
    try:
        pred_comms = algo_fn(G, seed=seed)
    except Exception as e:
        print(f"Error in {algo_name} on {exp_name} (seed={seed}): {e}")
        pred_comms = []
    dur = time.perf_counter() - t0
    
    if not pred_comms:
        return {
            "Experiment": exp_name,
            "Param_Value": param_val,
            "Algorithm": algo_name,
            "Seed": seed,
            "gNMI": 0.0,
            "Pairwise_F1": 0.0,
            "Shen_EQ": 0.0,
            "Nicosia_Qov": 0.0,
            "Time_Sec": dur
        }
        
    gnmi_val = onmi(pred_comms, gt)
    f1_val = pairwise_f1(pred_comms, gt)
    eq_val = shen_modularity_eq(G, [set(c) for c in pred_comms])
    qov_val = nicosia_qov_slpa(G, [set(c) for c in pred_comms])
    
    return {
        "Experiment": exp_name,
        "Param_Value": param_val,
        "Algorithm": algo_name,
        "Seed": seed,
        "gNMI": float(gnmi_val),
        "Pairwise_F1": float(f1_val),
        "Shen_EQ": float(eq_val),
        "Nicosia_Qov": float(qov_val),
        "Time_Sec": float(dur)
    }

# -----------------------------------------------------------------------------
# Experiment Runners
# -----------------------------------------------------------------------------

def run_experiment_suite(num_seeds: int = 5, max_workers: int = None):
    max_w = max_workers or max(1, (os.cpu_count() or 4) - 1)
    print("=" * 80)
    print(" COMPREHENSIVE LFR SYNTHETIC BENCHMARK EXPERIMENTAL SUITE")
    print(f" Evaluated Algorithms ({len(ALGORITHMS)}): {list(ALGORITHMS.keys())}")
    print(f" Independent Seeds per Config: {num_seeds} | CPU Workers: {max_w}")
    print("=" * 80)
    
    tasks = []
    
    # -------------------------------------------------------------------------
    # 1. Experiment 1: Mixing Parameter mu Sweep (mu in [0.1, 0.2, ..., 0.8])
    # -------------------------------------------------------------------------
    mu_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    print(f"\n[1/4] Preparing Experiment 1: Mixing Parameter mu Sweep ({mu_values})...")
    for mu in mu_values:
        for s in range(42, 42 + num_seeds):
            G, gt = generate_lfr_overlapping(n=1000, mu=mu, overlap_n=100, overlap_m=2, seed=s)
            for algo in ALGORITHMS:
                tasks.append((algo, "Sweep_Mu", mu, G, gt, s))
                
    # -------------------------------------------------------------------------
    # 2. Experiment 2: Overlapping Nodes On Sweep (On in [50, 100, 200, 300, 400, 500])
    # -------------------------------------------------------------------------
    on_values = [50, 100, 200, 300, 400, 500]
    print(f"[2/4] Preparing Experiment 2: Overlapping Nodes On Sweep ({on_values})...")
    for on in on_values:
        for s in range(42, 42 + num_seeds):
            G, gt = generate_lfr_overlapping(n=1000, mu=0.2, overlap_n=on, overlap_m=2, seed=s)
            for algo in ALGORITHMS:
                tasks.append((algo, "Sweep_On", on, G, gt, s))
                
    # -------------------------------------------------------------------------
    # 3. Experiment 3: Number of Memberships Om Sweep (Om in [2, 3, 4, 5, 6])
    # -------------------------------------------------------------------------
    om_values = [2, 3, 4, 5, 6]
    print(f"[3/4] Preparing Experiment 3: Overlapping Memberships Om Sweep ({om_values})...")
    for om in om_values:
        for s in range(42, 42 + num_seeds):
            G, gt = generate_lfr_overlapping(n=1000, mu=0.2, overlap_n=200, overlap_m=om, seed=s)
            for algo in ALGORITHMS:
                tasks.append((algo, "Sweep_Om", om, G, gt, s))
                
    # -------------------------------------------------------------------------
    # 4. Experiment 4: Standard Published LFR Configurations
    # -------------------------------------------------------------------------
    named_configs = {
        "LFR0 (mu=0.1, On=100, Om=2)": {"mu": 0.1, "on": 100, "om": 2},
        "LFR1 (mu=0.3, On=100, Om=2)": {"mu": 0.3, "on": 100, "om": 2},
        "LFR2 (mu=0.1, On=200, Om=3)": {"mu": 0.1, "on": 200, "om": 3},
        "LFR3 (mu=0.3, On=200, Om=3)": {"mu": 0.3, "on": 200, "om": 3},
    }
    print(f"[4/4] Preparing Experiment 4: Published Literature Configs ({list(named_configs.keys())})...")
    for name, cfg in named_configs.items():
        for s in range(42, 42 + num_seeds):
            G, gt = generate_lfr_overlapping(n=1000, mu=cfg["mu"], overlap_n=cfg["on"], overlap_m=cfg["om"], seed=s)
            for algo in ALGORITHMS:
                tasks.append((algo, "Named_Configs", name, G, gt, s))
                
    print(f"\n---> Submitting {len(tasks)} Total Evaluation Tasks across {max_w} CPU Workers...")
    
    results = []
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=max_w) as executor:
        futures = [executor.submit(evaluate_single_trial, t) for t in tasks]
        done_count = 0
        total_tasks = len(tasks)
        for f in as_completed(futures):
            res = f.result()
            results.append(res)
            done_count += 1
            if done_count % 20 == 0 or done_count == total_tasks:
                elapsed = time.time() - t_start
                rate = done_count / elapsed if elapsed > 0 else 0
                print(f"  Progress: {done_count}/{total_tasks} ({done_count/total_tasks*100:4.1f}%) | Elapsed: {elapsed:5.1f}s | Rate: {rate:4.1f} tasks/s")
                
    df_raw = pd.DataFrame(results)
    raw_csv = BENCH_DIR / "lfr_comprehensive_suite_raw_trials.csv"
    df_raw.to_csv(raw_csv, index=False)
    print(f"\nSaved raw trials to: {raw_csv}")
    
    # Generate aggregated summary tables & plots
    generate_summary_and_plots(df_raw)

def generate_summary_and_plots(df: pd.DataFrame):
    print("\n" + "=" * 80)
    print(" GENERATING AGGREGATED BENCHMARK SUMMARY & PUBLICATION-GRADE PLOTS")
    print("=" * 80)
    
    agg_df = df.groupby(["Experiment", "Param_Value", "Algorithm"]).agg({
        "gNMI": ["mean", "std"],
        "Pairwise_F1": ["mean", "std"],
        "Shen_EQ": ["mean", "std"],
        "Nicosia_Qov": ["mean", "std"],
        "Time_Sec": ["mean", "std"],
    }).reset_index()
    
    agg_df.columns = ["_".join(c).strip("_") for c in agg_df.columns.values]
    summary_csv = BENCH_DIR / "lfr_comprehensive_suite_summary.csv"
    agg_df.to_csv(summary_csv, index=False)
    print(f"Saved aggregated summary to: {summary_csv}")
    
    palette = {
        "OHP-MOCD (Proposed)": ("#1b9e77", "o", "-"),
        "SLPA (2011)": ("#e7298a", "s", "--"),
        "MCMOEA (2016)": ("#7570b3", "^", "-."),
    }
    
    # Plot Experiment 1: Mixing Parameter mu Sweep
    df_mu = agg_df[agg_df["Experiment"] == "Sweep_Mu"].sort_values("Param_Value")
    if not df_mu.empty:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for algo in ALGORITHMS:
            sub = df_mu[df_mu["Algorithm"] == algo]
            col, mark, ls = palette[algo]
            axes[0].errorbar(sub["Param_Value"], sub["gNMI_mean"], yerr=sub["gNMI_std"], label=algo, color=col, marker=mark, linestyle=ls, capsize=3)
            axes[1].errorbar(sub["Param_Value"], sub["Shen_EQ_mean"], yerr=sub["Shen_EQ_std"], label=algo, color=col, marker=mark, linestyle=ls, capsize=3)
            axes[2].plot(sub["Param_Value"], sub["Time_Sec_mean"], label=algo, color=col, marker=mark, linestyle=ls)
            
        axes[0].set_title("Ground-Truth Recovery ($gNMI$)", fontweight="bold")
        axes[0].set_xlabel(r"Mixing Parameter $\mu$")
        axes[0].set_ylabel("Generalized NMI")
        axes[0].grid(True, linestyle="--", alpha=0.5)
        axes[0].legend(loc="upper right")
        
        axes[1].set_title("Extended Modularity ($EQ$)", fontweight="bold")
        axes[1].set_xlabel(r"Mixing Parameter $\mu$")
        axes[1].set_ylabel("Shen Modularity $EQ$")
        axes[1].grid(True, linestyle="--", alpha=0.5)
        
        axes[2].set_title("Computational Runtime (s)", fontweight="bold")
        axes[2].set_xlabel(r"Mixing Parameter $\mu$")
        axes[2].set_ylabel("Time (seconds)")
        axes[2].set_yscale("log")
        axes[2].grid(True, linestyle="--", alpha=0.5)
        
        fig.suptitle(r"LFR Benchmark: Robustness to Mixing Parameter $\mu$ ($N=1000, O_n=100, O_m=2$)", fontweight="bold")
        fig.savefig(PLOTS_DIR / "lfr_exp1_mixing_param_mu_sweep.png", dpi=300, bbox_inches="tight")
        fig.savefig(PLOTS_DIR / "lfr_exp1_mixing_param_mu_sweep.pdf", bbox_inches="tight")
        plt.close(fig)
        print("Saved lfr_exp1_mixing_param_mu_sweep.png & .pdf")
        
    # Plot Experiment 2: Overlapping Nodes On Sweep
    df_on = agg_df[agg_df["Experiment"] == "Sweep_On"].sort_values("Param_Value")
    if not df_on.empty:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for algo in ALGORITHMS:
            sub = df_on[df_on["Algorithm"] == algo]
            col, mark, ls = palette[algo]
            axes[0].errorbar(sub["Param_Value"], sub["gNMI_mean"], yerr=sub["gNMI_std"], label=algo, color=col, marker=mark, linestyle=ls, capsize=3)
            axes[1].errorbar(sub["Param_Value"], sub["Nicosia_Qov_mean"], yerr=sub["Nicosia_Qov_std"], label=algo, color=col, marker=mark, linestyle=ls, capsize=3)
            
        axes[0].set_title("Ground-Truth Recovery ($gNMI$)", fontweight="bold")
        axes[0].set_xlabel("Number of Overlapping Nodes ($O_n$)")
        axes[0].set_ylabel("Generalized NMI")
        axes[0].grid(True, linestyle="--", alpha=0.5)
        axes[0].legend(loc="upper right")
        
        axes[1].set_title("Nicosia Overlapping Modularity ($Q_{ov}$)", fontweight="bold")
        axes[1].set_xlabel("Number of Overlapping Nodes ($O_n$)")
        axes[1].set_ylabel("Nicosia $Q_{ov}$")
        axes[1].grid(True, linestyle="--", alpha=0.5)
        
        fig.suptitle(r"LFR Benchmark: Scalability to Overlapping Node Count $O_n$ ($N=1000, \mu=0.2, O_m=2$)", fontweight="bold")
        fig.savefig(PLOTS_DIR / "lfr_exp2_overlapping_nodes_on_sweep.png", dpi=300, bbox_inches="tight")
        fig.savefig(PLOTS_DIR / "lfr_exp2_overlapping_nodes_on_sweep.pdf", bbox_inches="tight")
        plt.close(fig)
        print("Saved lfr_exp2_overlapping_nodes_on_sweep.png & .pdf")
        
    # Display Table for Named Configs
    df_named = agg_df[agg_df["Experiment"] == "Named_Configs"]
    if not df_named.empty:
        print("\n" + "=" * 80)
        print(" PUBLISHED LFR CONFIGURATIONS BENCHMARK TABLE")
        print("=" * 80)
        pvt_gnmi = df_named.pivot(index="Param_Value", columns="Algorithm", values="gNMI_mean")
        print("--- Ground-Truth Recovery (gNMI) ---")
        print(pvt_gnmi.to_string())
        
        pvt_eq = df_named.pivot(index="Param_Value", columns="Algorithm", values="Shen_EQ_mean")
        print("\n--- Extended Modularity (Shen EQ) ---")
        print(pvt_eq.to_string())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LFR Comprehensive Benchmark Suite")
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds per config (default: 5)")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel CPU workers")
    args = parser.parse_args()
    
    run_experiment_suite(num_seeds=args.seeds, max_workers=args.workers)
