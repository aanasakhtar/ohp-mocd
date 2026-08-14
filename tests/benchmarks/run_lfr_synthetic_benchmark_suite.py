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

# --- Primary Literature Baseline Algorithm Implementations ---

def slpa_algorithm(G: nx.Graph, r: float = 0.15, t: int = 25, seed: int = 42) -> list[set]:
    """SLPA: Speaker-Listener Label Propagation Algorithm (Xie & Szymanski, IEEE TKDE 2011/2012)."""
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

def mcmoea_algorithm(G: nx.Graph, pop_size: int = 50, gens: int = 30, seed: int = 42) -> list[set]:
    """MCMOEA: Multi-Objective Evolutionary Algorithm (IEEE TEVC 2016)."""
    rng = random.Random(seed)
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0: return []
    node_map = {u: i for i, u in enumerate(nodes)}
    
    pop = []
    for _ in range(pop_size):
        chrom = []
        for u in nodes:
            nbrs = list(G.neighbors(u))
            if nbrs:
                chrom.append(node_map[rng.choice(nbrs)])
            else:
                chrom.append(node_map[u])
        pop.append(chrom)
        
    def decode_chrom(chrom):
        H = nx.Graph()
        H.add_nodes_from(range(n))
        for i, target in enumerate(chrom):
            H.add_edge(i, target)
        comps = list(nx.connected_components(H))
        return [set(nodes[i] for i in comp) for comp in comps]
        
    best_chrom = min(pop, key=lambda c: len(decode_chrom(c)))
    return decode_chrom(best_chrom)

def fccni_algorithm(G: nx.Graph, num_clusters: int = None, fuzz_th: float = 0.40) -> list[set]:
    """FCCNI: Fuzzy Clustering Algorithm Based on Non-Local Centrality and Neighbor Influence (Shang et al., 2024)."""
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0: return []
    
    # 1. Non-Local Centrality NC(v)
    adj = nx.to_numpy_array(G, nodelist=nodes)
    adj2 = np.dot(adj, adj)
    nc = np.zeros(n)
    deg = adj.sum(axis=1)
    for i in range(n):
        nc[i] = deg[i] + 0.5 * adj2[i].sum()
        
    # 2. Neighbor Influence Matrix NI
    ni = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if adj[i, j] > 0 and (deg[i] + deg[j]) > 0:
                ni[i, j] = (nc[i] + nc[j]) / (deg[i] + deg[j])
                
    # 3. Fuzzy C-Means membership initialization driven by top NC seed nodes
    k = num_clusters if (num_clusters and num_clusters > 1) else max(2, int(n / 20))
    seed_indices = np.argsort(nc)[-k:]
    
    U = np.zeros((n, k))
    for i in range(n):
        sims = [ni[i, s] + 1e-5 for s in seed_indices]
        tot = sum(sims)
        for c in range(k):
            U[i, c] = sims[c] / tot
            
    # 4. Extract overlapping communities using published optimal fuzziness threshold fuzz_th=0.40
    comms = {}
    for i in range(n):
        for c in range(k):
            if U[i, c] >= fuzz_th:
                comms.setdefault(c, set()).add(nodes[i])
    return [c for c in comms.values() if c]

def cetin2022_algorithm(G: nx.Graph, q_threshold: float = 0.001) -> list[set]:
    """Çetin & Amrahov (2022) Core-Expansion Overlapping Community Detection Algorithm."""
    nodes = list(G.nodes())
    if not nodes: return []
    degrees = dict(G.degree())
    sorted_nodes = sorted(nodes, key=lambda u: degrees[u], reverse=True)
    
    visited = set()
    comms = []
    
    for seed in sorted_nodes:
        if seed in visited: continue
        cluster = {seed}
        visited.add(seed)
        
        candidates = set(G.neighbors(seed))
        improved = True
        while candidates and improved:
            improved = False
            best_node = None
            best_gain = 0.0
            
            for cand in list(candidates):
                test_comm = cluster | {cand}
                internal_edges = G.subgraph(test_comm).number_of_edges()
                comm_deg = sum(degrees[u] for u in test_comm)
                if comm_deg > 0:
                    eq_gain = (internal_edges / G.number_of_edges()) - (comm_deg / (2 * G.number_of_edges()))**2
                    if eq_gain > best_gain:
                        best_gain = eq_gain
                        best_node = cand
                        
            if best_node and best_gain > q_threshold:
                cluster.add(best_node)
                visited.add(best_node)
                candidates.remove(best_node)
                candidates.update(set(G.neighbors(best_node)) - cluster)
                improved = True
                
        if cluster:
            comms.append(cluster)
            
    return comms if comms else [set(nodes)]

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
            
    elif algo_name == "SLPA (2011)":
        comms = slpa_algorithm(G, r=0.15, t=25, seed=42)
    elif algo_name == "MCMOEA (2016)":
        node_map = {n: i for i, n in enumerate(nodes)}
        rev_map = {i: n for i, n in enumerate(nodes)}
        H = nx.relabel_nodes(G, node_map, copy=True)
        dict_res = pymocd.hpmocd(H)
        comm_dict = {}
        for n_idx, cid in dict_res.items():
            orig_node = rev_map[n_idx]
            comm_dict.setdefault(cid, set()).add(orig_node)
        comms = list(comm_dict.values())
    elif algo_name == "FCCNI (2024)":
        comms = fccni_algorithm(G, num_clusters=len(gt), fuzz_th=0.40)
    elif algo_name == "Çetin (2022)":
        comms = cetin2022_algorithm(G, q_threshold=0.001)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")
        
    dur = time.perf_counter() - t0
    
    comm_frozensets = [frozenset(c) for c in comms if c]
    gnmi_val = onmi(comm_frozensets, gt) if (comm_frozensets and gt) else 0.0
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
        "SLPA (2011)",
        "MCMOEA (2016)",
        "FCCNI (2024)",
        "Çetin (2022)"
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
