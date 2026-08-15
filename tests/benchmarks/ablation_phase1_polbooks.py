"""
ablation_phase1_polbooks.py

Evaluates Polbooks across 15 seeds (42..56) to collect full diagnostic metrics:
  - Raw community count: mean, std, min, max (range)
  - Merged community count: mean, std, min, max (range)
  - Avg memberships per node: mean, std
  - Shen Modularity (EQ): mean, std, peak
  - Nicosia Qov: mean, std, peak
"""

import sys, os
from pathlib import Path
import numpy as np
import networkx as nx

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pymocd
from tests.benchmarks.run_paper_comparative_suite import (
    load_polbooks, nicosia_qov, shen_modularity_eq
)
from tests.benchmarks.utils.merge import post_hoc_boundary_merge

def evaluate_polbooks_15_seeds(label: str):
    G_obj = load_polbooks()
    G = G_obj[0] if isinstance(G_obj, tuple) else G_obj
    nodes = list(G.nodes())
    node_map = {n: i for i, n in enumerate(nodes)}
    rev_map = {i: n for i, n in enumerate(nodes)}
    H = nx.relabel_nodes(G, node_map, copy=True)
    
    seeds = list(range(42, 42 + 15))
    
    raw_counts = []
    merged_counts = []
    avg_mems = []
    eq_list = []
    qov_list = []
    
    for seed in seeds:
        dict_res = pymocd.ohpmocd(
            H,
            pop_size=100,
            num_gens=100,
            cross_rate=0.8,
            mut_rate=0.5,
            init_strategy="boundary_seeded",
            init_overlap_prob=0.10,
            seed=seed
        )
        
        comm_dict = {}
        total_mems = 0
        for n_idx, cl in dict_res.items():
            orig = rev_map[n_idx]
            if isinstance(cl, (int, np.integer)):
                cl = [cl]
            total_mems += len(cl)
            for cid in cl:
                comm_dict.setdefault(cid, set()).add(orig)
                
        raw_comms = list(comm_dict.values())
        merged_comms = post_hoc_boundary_merge(G, raw_comms)
        
        raw_counts.append(len(raw_comms))
        merged_counts.append(len(merged_comms))
        avg_mems.append(total_mems / len(nodes))
        eq_list.append(shen_modularity_eq(G, merged_comms))
        qov_list.append(nicosia_qov(G, merged_comms))
        
    print(f"\n=======================================================")
    print(f" RESULTS FOR POLBOOKS: {label}")
    print(f"=======================================================")
    print(f"Raw Community Count:    {np.mean(raw_counts):.2f} ± {np.std(raw_counts):.2f} (min={np.min(raw_counts)}, max={np.max(raw_counts)}, range={np.max(raw_counts)-np.min(raw_counts)})")
    print(f"Merged Community Count: {np.mean(merged_counts):.2f} ± {np.std(merged_counts):.2f} (min={np.min(merged_counts)}, max={np.max(merged_counts)}, range={np.max(merged_counts)-np.min(merged_counts)})")
    print(f"Avg Memberships/Node:   {np.mean(avg_mems):.4f} ± {np.std(avg_mems):.4f}")
    print(f"Shen Modularity (EQ):   {np.mean(eq_list):.4f} ± {np.std(eq_list):.4f} (peak={np.max(eq_list):.4f})")
    print(f"Nicosia Modularity Qov: {np.mean(qov_list):.4f} ± {np.std(qov_list):.4f} (peak={np.max(qov_list):.4f})")
    
    return {
        "raw_mean": np.mean(raw_counts),
        "raw_std": np.std(raw_counts),
        "raw_min": np.min(raw_counts),
        "raw_max": np.max(raw_counts),
        "merged_mean": np.mean(merged_counts),
        "merged_std": np.std(merged_counts),
        "merged_min": np.min(merged_counts),
        "merged_max": np.max(merged_counts),
        "mems_mean": np.mean(avg_mems),
        "mems_std": np.std(avg_mems),
        "eq_mean": np.mean(eq_list),
        "eq_std": np.std(eq_list),
        "eq_peak": np.max(eq_list),
        "qov_mean": np.mean(qov_list),
        "qov_std": np.std(qov_list),
        "qov_peak": np.max(qov_list),
    }

if __name__ == "__main__":
    evaluate_polbooks_15_seeds("Phase1 Ratio = 0.0 (Uniform Membership)")
