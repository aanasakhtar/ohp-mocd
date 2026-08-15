"""
compare_three_membership_modes.py

Fast Side-by-Side Comparison of 3 OHP-MOCD Membership Formulations across 15 seeds:
  1. Uniform Membership (r_{v,c} = 1 / |M(v)|, alpha = -1.0)
  2. Pure DWI Membership (Degree-Weighted Neighborhood Influence, alpha = 0.0)
  3. Blended OCCSA + DWI (alpha = 0.50, or per-dataset alpha)

Reports side-by-side:
  - Shen Extended Modularity (EQ) [mean ± std, peak]
  - Nicosia Qov [mean ± std, peak]
  - Ground Truth gNMI / ONMI (when available)
  - Merged Community Count (mean ± std)
  - Avg Memberships per Node
"""

import sys, os, time
import concurrent.futures
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pymocd
from evaluation.metrics import onmi
from tests.benchmarks.run_paper_comparative_suite import (
    load_karate, load_dolphins, load_lesmis, load_polbooks, load_football,
    nicosia_qov, shen_modularity_eq, extract_ground_truth
)
from tests.benchmarks.utils.merge import post_hoc_boundary_merge

DATASETS = [
    ("Karate", load_karate),
    ("Dolphins", load_dolphins),
    ("Lesmis", load_lesmis),
    ("Polbooks", load_polbooks),
    ("Football", load_football),
]

MODES = [
    ("Uniform (1/|M(v)|)", -1.0),
    ("Pure DWI (deg-weighted)", 0.0),
    ("Blended (OCCSA+DWI, alpha=0.5)", 0.5),
    ("OCCSA-heavy (alpha=0.75)", 0.75),
]

def eval_trial(args):
    net_name, alpha_val, edge_list, gt, seed_val = args
    G = nx.Graph(edge_list)
    nodes = list(G.nodes())
    node_map = {n: i for i, n in enumerate(nodes)}
    rev_map = {i: n for i, n in enumerate(nodes)}
    H = nx.relabel_nodes(G, node_map, copy=True)
    
    t0 = time.perf_counter()
    dict_res = pymocd.ohpmocd(
        H,
        pop_size=100,
        num_gens=100,
        cross_rate=0.8,
        mut_rate=0.5,
        init_strategy="boundary_seeded",
        init_overlap_prob=0.10,
        alpha=alpha_val,
        seed=seed_val
    )
    dur = time.perf_counter() - t0
    
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
    
    eq = shen_modularity_eq(G, merged_comms)
    qov = nicosia_qov(G, merged_comms)
    
    gnmi_val = 0.0
    if gt is not None:
        comm_fsets = [frozenset(c) for c in merged_comms]
        gnmi_val = onmi(comm_fsets, gt)
        
    return {
        "EQ": eq,
        "Qov": qov,
        "gNMI": gnmi_val,
        "n_merged": len(merged_comms),
        "avg_mems": total_mems / len(nodes),
        "time": dur,
    }

def main():
    print("============================================================================")
    print(" SIDE-BY-SIDE EVALUATION: UNIFORM vs PURE DWI vs BLENDED OCCSA+DWI (15 SEEDS)")
    print("============================================================================")
    
    max_workers = max(1, (os.cpu_count() or 4) - 1)
    results_rows = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for net_name, loader in DATASETS:
            print(f"\n---> Evaluating Dataset: {net_name} ...")
            G_obj = loader()
            G = G_obj[0] if isinstance(G_obj, tuple) else G_obj
            edge_list = list(G.edges())
            gt = extract_ground_truth(G, net_name)
            
            for mode_label, alpha_val in MODES:
                tasks = [
                    (net_name, alpha_val, edge_list, gt, seed_val)
                    for seed_val in range(42, 42 + 15)
                ]
                futures = [executor.submit(eval_trial, t) for t in tasks]
                res_list = [f.result() for f in futures]
                
                eqs = [r["EQ"] for r in res_list]
                qovs = [r["Qov"] for r in res_list]
                gnmis = [r["gNMI"] for r in res_list]
                counts = [r["n_merged"] for r in res_list]
                mems = [r["avg_mems"] for r in res_list]
                
                print(f"  [{mode_label:32s}] Shen EQ: {np.mean(eqs):.4f} ± {np.std(eqs):.4f} (peak {np.max(eqs):.4f}) | gNMI: {np.mean(gnmis):.4f} (peak {np.max(gnmis):.4f}) | Qov: {np.mean(qovs):.4f} | Comms: {np.mean(counts):.2f} | Mems/node: {np.mean(mems):.3f}")
                
                results_rows.append({
                    "Dataset": net_name,
                    "Nodes": G.number_of_nodes(),
                    "Edges": G.number_of_edges(),
                    "Membership_Mode": mode_label,
                    "Alpha": alpha_val,
                    "EQ_mean": np.mean(eqs),
                    "EQ_std": np.std(eqs),
                    "EQ_peak": np.max(eqs),
                    "gNMI_mean": np.mean(gnmis),
                    "gNMI_peak": np.max(gnmis),
                    "Qov_mean": np.mean(qovs),
                    "Qov_peak": np.max(qovs),
                    "Comms_mean": np.mean(counts),
                    "Avg_Mems_Per_Node": np.mean(mems),
                })
                
    df = pd.DataFrame(results_rows)
    out_csv = REPO_ROOT / "tests" / "benchmarks" / "membership_modes_side_by_side.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[DONE] Saved side-by-side results to {out_csv}")

if __name__ == "__main__":
    main()
