"""
eval_automatic_peak_merge_experiment.py

Experimental comparison script evaluating Option 1: Automatic Peak Modularity Merging
(100% parameter-free, stopping automatically when Delta EQ <= 0)
side-by-side with Standard Fixed Threshold Merging across benchmark datasets.
"""

import sys, os
sys.path.insert(0, os.path.abspath('.'))

import time
import numpy as np
import networkx as nx
import pandas as pd
from pathlib import Path
import pymocd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BENCH_DIR = REPO_ROOT / "tests" / "benchmarks"

from tests.benchmarks.run_paper_comparative_suite import (
    load_karate, load_dolphins, load_lesmis, load_polbooks, load_football,
    load_netscience, load_celegans, extract_ground_truth,
    nicosia_qov_slpa_scaled, shen_modularity_eq, overlapping_coverage_cetin,
    post_hoc_boundary_merge, onmi
)

def automatic_peak_modularity_merge(G: nx.Graph, communities: list[set]) -> list[set]:
    """
    Option 1: Parameter-Free Automatic Peak Modularity Merge.
    Iteratively merges neighboring community pairs (C_i, C_j) that yield the highest
    positive global modularity gain (Delta EQ > 0). Stops automatically when max Delta EQ <= 0.
    Uses O(m) inter-community edge precomputation and incremental O(1) ΔQ gain updates.
    """
    return post_hoc_boundary_merge(G, communities, merge_threshold='auto')

def run_experiment_on_dataset(net_name: str, loader_fn) -> list[dict]:
    """Runs standard OHP-MOCD vs Option 1 Automatic Peak Merge OHP-MOCD side-by-side."""
    print(f"--- Evaluating Dataset: {net_name} ---")
    G_obj = loader_fn()
    G = G_obj[0] if isinstance(G_obj, tuple) else G_obj
    gt = extract_ground_truth(G, net_name)
    
    nodes = list(G.nodes())
    node_map = {n: i for i, n in enumerate(nodes)}
    rev_map = {i: n for i, n in enumerate(nodes)}
    H = nx.relabel_nodes(G, node_map, copy=True)
    
    # 1. Run NSGA-II evolution to get raw Pareto communities
    t0 = time.perf_counter()
    dict_res = pymocd.ohpmocd(
        H,
        init_strategy="boundary_seeded",
        init_overlap_prob=0.15,
        overlap_support_threshold=0.25,
        overlap_removal_threshold=0.08,
        switch_margin=0.05,
        alpha=0.25,
        seed=42
    )
    dur_evo = time.perf_counter() - t0
    
    comm_dict = {}
    for n_idx, comm_list in dict_res.items():
        orig_node = rev_map[n_idx]
        if isinstance(comm_list, (int, np.integer)): comm_list = [comm_list]
        for cid in comm_list: comm_dict.setdefault(cid, set()).add(orig_node)
    raw_comms = list(comm_dict.values())
    
    # Variant A: Raw Unmerged OHP-MOCD
    eq_raw = shen_modularity_eq(G, raw_comms)
    qov_raw = nicosia_qov_slpa_scaled(G, raw_comms)
    gnmi_raw = onmi([frozenset(c) for c in raw_comms if c], gt) if gt else 0.0
    
    # Variant B: Standard Fixed Threshold Merge (theta_merge = 0.50)
    comms_std = post_hoc_boundary_merge(G, raw_comms, merge_threshold=0.50)
    eq_std = shen_modularity_eq(G, comms_std)
    qov_std = nicosia_qov_slpa_scaled(G, comms_std)
    gnmi_std = onmi([frozenset(c) for c in comms_std if c], gt) if gt else 0.0
    
    # Variant C: Option 1 — Parameter-Free Automatic Peak Modularity Merge
    t_opt1 = time.perf_counter()
    comms_opt1 = automatic_peak_modularity_merge(G, raw_comms)
    dur_opt1 = time.perf_counter() - t_opt1
    eq_opt1 = shen_modularity_eq(G, comms_opt1)
    qov_opt1 = nicosia_qov_slpa_scaled(G, comms_opt1)
    gnmi_opt1 = onmi([frozenset(c) for c in comms_opt1 if c], gt) if gt else 0.0
    
    print(f"    Raw (No Merge) : EQ = {eq_raw:.4f} | Qov_SLPA = {qov_raw:.4f} | gNMI = {gnmi_raw:.4f} | Comms = {len(raw_comms)}")
    print(f"    Std (Fixed 0.5): EQ = {eq_std:.4f} | Qov_SLPA = {qov_std:.4f} | gNMI = {gnmi_std:.4f} | Comms = {len(comms_std)}")
    print(f"    Option 1 (Peak): EQ = {eq_opt1:.4f} | Qov_SLPA = {qov_opt1:.4f} | gNMI = {gnmi_opt1:.4f} | Comms = {len(comms_opt1)} (Merge Time: {dur_opt1:.3f}s)\n")
    
    return [
        {"Dataset": net_name, "Variant": "Raw (No Merge)", "EQ": eq_raw, "Qov_SLPA": qov_raw, "gNMI": gnmi_raw, "Num_Comms": len(raw_comms), "Is_Parameter_Free": False},
        {"Dataset": net_name, "Variant": "Std Threshold (0.50)", "EQ": eq_std, "Qov_SLPA": qov_std, "gNMI": gnmi_std, "Num_Comms": len(comms_std), "Is_Parameter_Free": False},
        {"Dataset": net_name, "Variant": "Option 1 (Automatic Peak)", "EQ": eq_opt1, "Qov_SLPA": qov_opt1, "gNMI": gnmi_opt1, "Num_Comms": len(comms_opt1), "Is_Parameter_Free": True},
    ]

def main():
    print("=================================================================")
    print(" EXPERIMENT: OPTION 1 AUTOMATIC PEAK MODULARITY MERGING ")
    print("=================================================================\n")
    
    datasets = [
        ("Karate", load_karate),
        ("Dolphins", load_dolphins),
        ("Lesmis", load_lesmis),
        ("Polbooks", load_polbooks),
        ("Football", load_football),
        ("Netscience", load_netscience),
        ("Celegans", load_celegans),
    ]
    
    results = []
    for net_name, loader_fn in datasets:
        res_rows = run_experiment_on_dataset(net_name, loader_fn)
        results.extend(res_rows)
        
    df = pd.DataFrame(results)
    out_csv = BENCH_DIR / "automatic_peak_merge_experiment_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved experiment results to: {out_csv}")
    print("\nOPTION 1 EXPERIMENT COMPLETED SUCCESSFULLY.")

if __name__ == "__main__":
    main()
