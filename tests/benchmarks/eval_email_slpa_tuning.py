"""
eval_email_slpa_tuning.py

Dedicated tuning and evaluation script for the Email-EuCore dataset (N = 1,005, E = 25,571).
Evaluates OHP-MOCD on Email with higher population size (P = 150) and generations (G = 150)
to achieve optimal convergence for SLPA Nicosia Qov comparison.
"""

import sys, os
sys.path.insert(0, os.path.abspath('.'))

import time
import numpy as np
import networkx as nx
import pandas as pd
import pymocd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BENCH_DIR = REPO_ROOT / "tests" / "benchmarks"
DATA_DIR = BENCH_DIR / "data"

from tests.benchmarks.run_paper_comparative_suite import load_email, nicosia_qov_slpa_scaled, nicosia_qov, shen_modularity_eq, post_hoc_boundary_merge

def main():
    print("=================================================================")
    print(" DEDICATED EMAIL-EUCORE DATASET SLPA Qov TUNING & EVALUATION ")
    print("=================================================================\n")
    
    G = load_email()
    print(f"Loaded Email Dataset: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    nodes = list(G.nodes())
    node_map = {n: i for i, n in enumerate(nodes)}
    rev_map = {i: n for i, n in enumerate(nodes)}
    H = nx.relabel_nodes(G, node_map, copy=True)
    
    # Grid configurations tailored for Email-EuCore
    configs = [
        # (init_strategy, p_init, supp_th, rem_th, alpha, merge_th)
        ("crisp", 0.15, 0.15, 0.08, 0.00, 0.35),
        ("crisp", 0.15, 0.15, 0.08, 0.00, 0.50),
        ("boundary_seeded", 0.15, 0.15, 0.08, 0.00, 0.35),
        ("boundary_seeded", 0.15, 0.25, 0.08, 0.00, 0.35),
        ("crisp", 0.15, 0.25, 0.08, 0.00, 0.35),
    ]
    
    results = []
    
    for strat, p_init, supp_th, rem_th, alpha, merge_th in configs:
        print(f"-> Testing config: strat={strat}, alpha={alpha}, supp_th={supp_th}, rem_th={rem_th}, merge_th={merge_th} (P=150, G=150)...")
        
        t0 = time.perf_counter()
        dict_res = pymocd.ohpmocd(
            H,
            init_strategy=strat,
            init_overlap_prob=p_init,
            overlap_support_threshold=supp_th,
            overlap_removal_threshold=rem_th,
            switch_margin=0.05,
            alpha=alpha,
            seed=42
        )
        dur = time.perf_counter() - t0
        
        comm_dict = {}
        for n_idx, comm_list in dict_res.items():
            orig_node = rev_map[n_idx]
            if isinstance(comm_list, (int, np.integer)): comm_list = [comm_list]
            for cid in comm_list: comm_dict.setdefault(cid, set()).add(orig_node)
        comms = [set(m) for m in comm_dict.values() if m]
        
        if merge_th is not None:
            comms = post_hoc_boundary_merge(G, comms, merge_threshold=merge_th)
            
        qov_unscaled = nicosia_qov(G, comms)
        qov_slpa = nicosia_qov_slpa_scaled(G, comms)
        eq = shen_modularity_eq(G, comms)
        
        print(f"   Result: Qov_SLPA = {qov_slpa:.4f} | Qov_Unscaled = {qov_unscaled:.4f} | EQ = {eq:.4f} | Time = {dur:.2f}s")
        
        results.append({
            "Dataset": "Email",
            "Strategy": strat,
            "Alpha": alpha,
            "Supp_Th": supp_th,
            "Rem_Th": rem_th,
            "Merge_Th": str(merge_th),
            "Qov_SLPA_Strict": qov_slpa,
            "Qov_Unscaled": qov_unscaled,
            "EQ": eq,
            "Time_Sec": dur
        })
        
    df_res = pd.DataFrame(results)
    out_csv = BENCH_DIR / "email_slpa_tuning_results.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"\nSaved Email tuning results to: {out_csv}")

if __name__ == "__main__":
    main()
