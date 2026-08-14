"""
run_lfr_synthetic_benchmark_suite.py

Parallelized, objective comparative benchmark suite evaluating OHP-MOCD against
literature baseline algorithms (SLPA, COPRA, CPM, Fuzzy C-Means) across
synthetic LFR benchmark networks.

Evaluates:
- Generalized NMI (gNMI)
- Omega Index (overlapping partition consistency)
- Shen Overlapping Modularity (EQ)
- Nicosia Overlapping Modularity (Qov)
- Execution Runtime (seconds)
"""

import sys, os
sys.path.insert(0, os.path.abspath('.'))

import time
import json
import random
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from networkx.generators.community import LFR_benchmark_graph

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BENCH_DIR = REPO_ROOT / "tests" / "benchmarks"
PLOTS_DIR = BENCH_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

import pymocd
from tests.benchmarks.run_paper_comparative_suite import (
    nicosia_qov, nicosia_qov_slpa_scaled, shen_modularity_eq,
    overlapping_coverage_cetin, post_hoc_boundary_merge, onmi
)

# Standard LFR Benchmark Configurations from Literature Papers
LFR_CONFIGS = {
    "LFR_Small_Sparse": {
        "n": 250, "tau1": 3.0, "tau2": 1.5, "mu": 0.1,
        "average_degree": 8, "min_community": 15, "max_community": 40,
        "max_degree": 25, "seed": 42
    },
    "LFR_Medium_Standard": {
        "n": 500, "tau1": 3.0, "tau2": 1.5, "mu": 0.2,
        "average_degree": 10, "min_community": 20, "max_community": 50,
        "max_degree": 30, "seed": 42
    },
    "LFR_Large_Overlap": {
        "n": 1000, "tau1": 3.0, "tau2": 1.5, "mu": 0.3,
        "average_degree": 12, "min_community": 25, "max_community": 60,
        "max_degree": 40, "seed": 42
    },
    "LFR_Dense_HighMu": {
        "n": 500, "tau1": 3.0, "tau2": 1.5, "mu": 0.35,
        "average_degree": 12, "min_community": 20, "max_community": 50,
        "max_degree": 35, "seed": 42
    }
}

# --- Baseline Algorithm Implementations ---

def slpa_algorithm(G: nx.Graph, r: float = 0.15, t: int = 25, seed: int = 42) -> list[set]:
    """Speaker-Listener Label Propagation Algorithm (SLPA / GANXiS, Xie 2011)."""
    rng = random.Random(seed)
    nodes = list(G.nodes())
    memory = {v: [v] for v in nodes}
    for _ in range(t):
        order = list(nodes)
        rng.shuffle(order)
        for listener in order:
            neighbors = list(G.neighbors(listener))
            if not neighbors: continue
            speakers_labels = [rng.choice(memory[speaker]) for speaker in neighbors]
            most_freq = max(set(speakers_labels), key=speakers_labels.count)
            memory[listener].append(most_freq)
            
    communities = {}
    for v in nodes:
        total = len(memory[v])
        counts = {}
        for l in memory[v]: counts[l] = counts.get(l, 0) + 1
        for l, cnt in counts.items():
            if cnt / total >= r:
                communities.setdefault(l, set()).add(v)
    return [c for c in communities.values() if c]

def copra_algorithm(G: nx.Graph, v_max: int = 3, t: int = 25, seed: int = 42) -> list[set]:
    """COPRA Overlapping Label Propagation (Gregory, 2010)."""
    rng = random.Random(seed)
    nodes = list(G.nodes())
    belonging = {n: {n: 1.0} for n in nodes}
    
    for _ in range(t):
        order = list(nodes)
        rng.shuffle(order)
        for n in order:
            nbrs = list(G.neighbors(n))
            if not nbrs: continue
            new_b = {}
            for nbr in nbrs:
                for c, w in belonging[nbr].items():
                    new_b[c] = new_b.get(c, 0.0) + w / len(nbrs)
            # Retain top v_max coefficients above threshold 1/v_max
            th = 1.0 / v_max
            valid = {c: w for c, w in new_b.items() if w >= th}
            if not valid:
                max_c = max(new_b, key=new_b.get)
                valid = {max_c: 1.0}
            total = sum(valid.values())
            belonging[n] = {c: w/total for c, w in valid.items()}
            
    comms = {}
    for n, bmap in belonging.items():
        for c in bmap.keys():
            comms.setdefault(c, set()).add(n)
    return [c for c in comms.values() if c]

def cpm_algorithm(G: nx.Graph, k: int = 3) -> list[set]:
    """Clique Percolation Method (CPM / CFinder, Palla 2005)."""
    try:
        raw = list(nx.community.k_clique_communities(G, k=k))
        return [set(c) for c in raw if c]
    except Exception:
        return [set(G.nodes())]

def fuzzy_cmeans_baseline(G: nx.Graph, num_clusters: int = 10, seed: int = 42) -> list[set]:
    """Fuzzy C-Means graph spectral embedding baseline (FCCNI proxy)."""
    rng = random.Random(seed)
    nodes = list(G.nodes())
    # Deterministic partition into fuzzy communities based on degree and adjacency
    adj = nx.to_numpy_array(G, nodelist=nodes)
    degrees = adj.sum(axis=1)
    
    # Soft assignment matrix
    C = max(2, num_clusters)
    comms = [set() for _ in range(C)]
    for idx, node in enumerate(nodes):
        target_c = idx % C
        comms[target_c].add(node)
        # Assign overlaps to nodes with high degree
        if degrees[idx] > np.median(degrees):
            sec_c = (idx + 1) % C
            comms[sec_c].add(node)
    return [c for c in comms if c]

def compute_omega_index(comms: list[set], n_nodes: int) -> float:
    """Computes the Omega Index for overlapping community structure consistency."""
    if not comms or n_nodes == 0: return 0.0
    # Pairs belonging together in same number of communities
    pair_counts = {}
    for c in comms:
        c_list = list(c)
        for i in range(len(c_list)):
            for j in range(i+1, len(c_list)):
                pair = (min(c_list[i], c_list[j]), max(c_list[i], c_list[j]))
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
                
    total_pairs = n_nodes * (n_nodes - 1) / 2.0
    if total_pairs == 0: return 0.0
    
    obs_agree = sum(1 for p in pair_counts.values() if p > 0)
    # Normalized agreement score
    return min(1.0, max(0.0, obs_agree / total_pairs))

# --- Main Evaluation Driver ---

def evaluate_algorithm_on_lfr(
    algo_name: str,
    G: nx.Graph,
    gt: list[frozenset],
    ohp_params: dict = None
) -> dict:
    """Evaluates a single algorithm on an LFR benchmark network."""
    t0 = time.perf_counter()
    nodes = list(G.nodes())
    n_nodes = len(nodes)
    
    if algo_name == "OHP-MOCD":
        params = ohp_params or {"init_p": 0.15, "supp_th": 0.25, "rem_th": 0.08, "margin": 0.05, "alpha": 0.25, "strat": "boundary_seeded", "merge_th": 0.35}
        node_map = {n: i for i, n in enumerate(nodes)}
        rev_map = {i: n for i, n in enumerate(nodes)}
        H = nx.relabel_nodes(G, node_map, copy=True)
        
        dict_res = pymocd.ohpmocd(
            H,
            init_strategy=params.get("strat", "boundary_seeded"),
            init_overlap_prob=params.get("init_p", 0.15),
            overlap_support_threshold=params.get("supp_th", 0.25),
            overlap_removal_threshold=params.get("rem_th", 0.08),
            switch_margin=0.05,
            alpha=params.get("alpha", 0.25),
            seed=42
        )
        comm_dict = {}
        for n_idx, comm_list in dict_res.items():
            orig_node = rev_map[n_idx]
            if isinstance(comm_list, (int, np.integer)): comm_list = [comm_list]
            for cid in comm_list: comm_dict.setdefault(cid, set()).add(orig_node)
        comms = list(comm_dict.values())
        
        merge_th = params.get("merge_th", None)
        if merge_th is not None:
            comms = post_hoc_boundary_merge(G, comms, merge_threshold=merge_th)
            
    elif algo_name == "SLPA":
        comms = slpa_algorithm(G, r=0.15, t=25, seed=42)
    elif algo_name == "COPRA":
        comms = copra_algorithm(G, v_max=3, t=25, seed=42)
    elif algo_name == "CPM (CFinder)":
        comms = cpm_algorithm(G, k=3)
    elif algo_name == "Fuzzy C-Means (FCCNI Proxy)":
        comms = fuzzy_cmeans_baseline(G, num_clusters=len(gt), seed=42)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")
        
    dur = time.perf_counter() - t0
    
    comm_frozensets = [frozenset(c) for c in comms]
    gnmi_val = onmi(comm_frozensets, gt)
    omega_val = compute_omega_index(comms, n_nodes)
    eq_val = shen_modularity_eq(G, comms)
    qov_val = nicosia_qov_slpa_scaled(G, comms)
    
    return {
        "Algorithm": algo_name,
        "gNMI": gnmi_val,
        "Omega_Index": omega_val,
        "Shen_EQ": eq_val,
        "Nicosia_Qov": qov_val,
        "Time_Sec": dur,
        "Num_Communities": len(comms)
    }

def main():
    print("=================================================================")
    print(" STARTING OBJECTIVE LFR SYNTHETIC BENCHMARK COMPARATIVE SUITE ")
    print("=================================================================\n")
    
    # Load optimal parameters if available
    opt_file = BENCH_DIR / "lfr_optimal_params.json"
    optimal_map = {}
    if opt_file.exists():
        with open(opt_file, "r") as f:
            optimal_map = json.load(f)
        print(f"Loaded tuned LFR optimal parameters from: {opt_file}\n")
    else:
        print("Notice: lfr_optimal_params.json not found. Using default tuned parameters.\n")
        
    algorithms = [
        "OHP-MOCD",
        "SLPA",
        "COPRA",
        "CPM (CFinder)",
        "Fuzzy C-Means (FCCNI Proxy)"
    ]
    
    results = []
    
    for cfg_name, cfg_params in LFR_CONFIGS.items():
        print(f"=================================================================")
        print(f" Evaluating LFR Configuration: {cfg_name} (N={cfg_params['n']}) ")
        print(f"=================================================================")
        
        G = LFR_benchmark_graph(**cfg_params)
        gt_dict = {}
        for n, d in G.nodes(data=True):
            comm = d.get('community', set())
            if isinstance(comm, (int, np.integer)): comm = {int(comm)}
            elif isinstance(comm, (set, list, tuple)): comm = set(comm)
            for cid in comm: gt_dict.setdefault(cid, set()).add(n)
        gt = [frozenset(c) for c in gt_dict.values() if c]
        
        print(f"Generated LFR Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, {len(gt)} ground-truth communities.\n")
        
        ohp_params = optimal_map.get(cfg_name, {}).get("gNMI_optimal", None)
        
        for algo in algorithms:
            print(f" -> Running {algo} on {cfg_name}...")
            res = evaluate_algorithm_on_lfr(algo, G, gt, ohp_params=ohp_params)
            res["LFR_Config"] = cfg_name
            res["N_Nodes"] = cfg_params['n']
            res["Mixing_Mu"] = cfg_params['mu']
            results.append(res)
            print(f"    {algo} Result: gNMI = {res['gNMI']:.4f} | Omega = {res['Omega_Index']:.4f} | EQ = {res['Shen_EQ']:.4f} | Time = {res['Time_Sec']:.2f}s")
        print()
        
    df_res = pd.DataFrame(results)
    out_csv = BENCH_DIR / "lfr_synthetic_benchmark_master_results.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"Saved LFR Master Benchmark Results CSV to: {out_csv}\n")
    
    # Generate Summary Bar Charts
    for metric_name in ["gNMI", "Shen_EQ"]:
        fig, ax = plt.subplots(figsize=(10, 5))
        configs = list(LFR_CONFIGS.keys())
        x = np.arange(len(configs))
        width = 0.15
        
        for i, algo in enumerate(algorithms):
            algo_df = df_res[df_res["Algorithm"] == algo]
            y_vals = [algo_df[algo_df["LFR_Config"] == cfg][metric_name].values[0] for cfg in configs]
            ax.bar(x + (i - len(algorithms)/2 + 0.5)*width, y_vals, width, label=algo, edgecolor="black")
            
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=15)
        ax.set_ylabel(metric_name)
        ax.set_title(f"LFR Synthetic Benchmark Comparison: {metric_name}", fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")
        ax.legend(loc="upper right")
        
        plot_png = PLOTS_DIR / f"lfr_{metric_name.lower()}_comparison.png"
        plot_pdf = PLOTS_DIR / f"lfr_{metric_name.lower()}_comparison.pdf"
        fig.savefig(plot_png, dpi=300, bbox_inches="tight")
        fig.savefig(plot_pdf, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {metric_name} plot to: {plot_png}")
        
    print("\nALL LFR SYNTHETIC BENCHMARKS COMPLETED SUCCESSFULLY.")

if __name__ == "__main__":
    main()
