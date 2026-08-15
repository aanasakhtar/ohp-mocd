"""
exp1_search_budget_celegans_email.py

Evaluates the Search-Budget Hypothesis on Celegans (N=297) and Email (N=1005) across 15 seeds:
  - Baseline Budget: pop_size=100, num_gens=100
  - Expanded Budget: pop_size=300, num_gens=300

Collects:
  - Nicosia Qov (SLPA linear clamp)
  - Shen Extended Modularity (EQ)
  - Raw Community Count
  - Merged Community Count
  - Avg Memberships per Node
  - Execution Time
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
from tests.benchmarks.run_paper_comparative_suite import (
    load_celegans, load_email, nicosia_qov_slpa, shen_modularity_eq
)
from tests.benchmarks.utils.merge import post_hoc_boundary_merge

DATASETS = [
    ("Celegans", load_celegans),
    ("Email", load_email),
]

CONFIGS = [
    ("Baseline (pop=100, gens=100)", 100, 100),
    ("Expanded (pop=300, gens=300)", 300, 300),
]

def eval_single_trial(args):
    net_name, pop, gens, edge_list, seed_val = args
    G = nx.Graph(edge_list)
    nodes = list(G.nodes())
    node_map = {n: i for i, n in enumerate(nodes)}
    rev_map = {i: n for i, n in enumerate(nodes)}
    H = nx.relabel_nodes(G, node_map, copy=True)
    
    t0 = time.perf_counter()
    dict_res = pymocd.ohpmocd(
        H,
        pop_size=pop,
        num_gens=gens,
        cross_rate=0.8,
        mut_rate=0.5,
        init_strategy="boundary_seeded",
        init_overlap_prob=0.10,
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
    
    qov = nicosia_qov_slpa(G, merged_comms)
    eq = shen_modularity_eq(G, merged_comms)
    
    return {
        "Qov": qov,
        "EQ": eq,
        "raw_count": len(raw_comms),
        "merged_count": len(merged_comms),
        "avg_mems": total_mems / len(nodes),
        "time": dur,
    }

def main():
    print("================================================================================")
    print(" EXPERIMENT 1: SEARCH BUDGET HYPOTHESIS ON CELEGANS & EMAIL (15 SEEDS)")
    print("================================================================================")
    
    max_workers = max(1, (os.cpu_count() or 4) - 1)
    results = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for net_name, loader in DATASETS:
            print(f"\n---> Dataset: {net_name} ...")
            G_obj = loader()
            G = G_obj[0] if isinstance(G_obj, tuple) else G_obj
            edge_list = list(G.edges())
            
            for cfg_name, pop, gens in CONFIGS:
                print(f"  Running {cfg_name} over 15 seeds in parallel...")
                tasks = [
                    (net_name, pop, gens, edge_list, seed)
                    for seed in range(42, 42 + 15)
                ]
                futures = [executor.submit(eval_single_trial, t) for t in tasks]
                res_list = [f.result() for f in futures]
                
                qovs = [r["Qov"] for r in res_list]
                eqs = [r["EQ"] for r in res_list]
                raws = [r["raw_count"] for r in res_list]
                merged = [r["merged_count"] for r in res_list]
                mems = [r["avg_mems"] for r in res_list]
                times = [r["time"] for r in res_list]
                
                print(f"    [{cfg_name:28s}] Nicosia Qov: {np.mean(qovs):.4f} ± {np.std(qovs):.4f} (peak {np.max(qovs):.4f}) | Shen EQ: {np.mean(eqs):.4f} ± {np.std(eqs):.4f} | Merged Comms: {np.mean(merged):.2f} | Mems/node: {np.mean(mems):.3f} | Time/seed: {np.mean(times):.2f}s")
                
                results.append({
                    "Dataset": net_name,
                    "Nodes": G.number_of_nodes(),
                    "Edges": G.number_of_edges(),
                    "Config": cfg_name,
                    "PopSize": pop,
                    "NumGens": gens,
                    "Qov_mean": np.mean(qovs),
                    "Qov_std": np.std(qovs),
                    "Qov_peak": np.max(qovs),
                    "EQ_mean": np.mean(eqs),
                    "EQ_std": np.std(eqs),
                    "EQ_peak": np.max(eqs),
                    "Raw_Comms_mean": np.mean(raws),
                    "Raw_Comms_std": np.std(raws),
                    "Merged_Comms_mean": np.mean(merged),
                    "Merged_Comms_std": np.std(merged),
                    "Avg_Mems_Per_Node": np.mean(mems),
                    "Time_Per_Seed": np.mean(times),
                })
                
    df = pd.DataFrame(results)
    out_csv = REPO_ROOT / "tests" / "benchmarks" / "exp1_search_budget_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[DONE] Saved search-budget results to {out_csv}")

if __name__ == "__main__":
    main()
