"""
run_step4_sanity_diagnostic.py

Sanity check diagnostic on all 5 datasets with the clean, uniform-only, phase1-resolved code
across 15 seeds (42..56).
"""

import sys, os, time
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

def run_diagnostics():
    print("===================================================================================")
    print(" STEP 4 SANITY DIAGNOSTIC: UNIFORM-ONLY, PHASE1-RESOLVED CLEAN CORE (15 SEEDS)")
    print("===================================================================================")
    
    rows = []
    seeds = list(range(42, 42 + 15))
    
    for net_name, loader in DATASETS:
        G_obj = loader()
        G = G_obj[0] if isinstance(G_obj, tuple) else G_obj
        nodes = list(G.nodes())
        node_map = {n: i for i, n in enumerate(nodes)}
        rev_map = {i: n for i, n in enumerate(nodes)}
        H = nx.relabel_nodes(G, node_map, copy=True)
        gt = extract_ground_truth(G, net_name)
        
        raw_counts = []
        merged_counts = []
        avg_mems = []
        max_mems = []
        eq_list = []
        qov_list = []
        gnmi_list = []
        
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
            cur_max_mem = 0
            for n_idx, cl in dict_res.items():
                orig = rev_map[n_idx]
                if isinstance(cl, (int, np.integer)):
                    cl = [cl]
                total_mems += len(cl)
                cur_max_mem = max(cur_max_mem, len(cl))
                for cid in cl:
                    comm_dict.setdefault(cid, set()).add(orig)
                    
            raw_comms = list(comm_dict.values())
            merged_comms = post_hoc_boundary_merge(G, raw_comms)
            
            raw_counts.append(len(raw_comms))
            merged_counts.append(len(merged_comms))
            avg_mems.append(total_mems / len(nodes))
            max_mems.append(cur_max_mem)
            eq_list.append(shen_modularity_eq(G, merged_comms))
            qov_list.append(nicosia_qov(G, merged_comms))
            
            if gt is not None:
                comm_fsets = [frozenset(c) for c in merged_comms]
                gnmi_list.append(onmi(comm_fsets, gt))
            else:
                gnmi_list.append(0.0)
                
        print(f"\n---> Dataset: {net_name} (|V|={G.number_of_nodes()}, |E|={G.number_of_edges()})")
        print(f"  • Raw Community Count:    {np.mean(raw_counts):.2f} ± {np.std(raw_counts):.2f} [min={np.min(raw_counts)}, max={np.max(raw_counts)}, range={np.max(raw_counts)-np.min(raw_counts)}]")
        print(f"  • Merged Community Count: {np.mean(merged_counts):.2f} ± {np.std(merged_counts):.2f} [min={np.min(merged_counts)}, max={np.max(merged_counts)}, range={np.max(merged_counts)-np.min(merged_counts)}]")
        print(f"  • Avg Memberships/Node:   {np.mean(avg_mems):.4f} ± {np.std(avg_mems):.4f} [Peak Max Mem per Node: {np.max(max_mems)}]")
        print(f"  • Shen Modularity (EQ):   {np.mean(eq_list):.4f} ± {np.std(eq_list):.4f} [Peak: {np.max(eq_list):.4f}]")
        print(f"  • Nicosia Modularity Qov: {np.mean(qov_list):.4f} ± {np.std(qov_list):.4f} [Peak: {np.max(qov_list):.4f}]")
        if gt is not None:
            print(f"  • Ground Truth (gNMI):    {np.mean(gnmi_list):.4f} ± {np.std(gnmi_list):.4f} [Peak: {np.max(gnmi_list):.4f}]")
            
        rows.append({
            "Dataset": net_name,
            "Nodes": G.number_of_nodes(),
            "Edges": G.number_of_edges(),
            "Raw_Comms_mean": np.mean(raw_counts),
            "Raw_Comms_std": np.std(raw_counts),
            "Raw_Comms_range": f"{np.min(raw_counts)}-{np.max(raw_counts)}",
            "Merged_Comms_mean": np.mean(merged_counts),
            "Merged_Comms_std": np.std(merged_counts),
            "Merged_Comms_range": f"{np.min(merged_counts)}-{np.max(merged_counts)}",
            "Avg_Mems_Node": np.mean(avg_mems),
            "Max_Mem_Node": np.max(max_mems),
            "Shen_EQ_mean": np.mean(eq_list),
            "Shen_EQ_peak": np.max(eq_list),
            "Nicosia_Qov_mean": np.mean(qov_list),
            "Nicosia_Qov_peak": np.max(qov_list),
            "gNMI_mean": np.mean(gnmi_list) if gt is not None else np.nan,
            "gNMI_peak": np.max(gnmi_list) if gt is not None else np.nan,
        })
        
    df = pd.DataFrame(rows)
    out_csv = REPO_ROOT / "tests" / "benchmarks" / "sanity_diagnostic_step4.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[DONE] Saved Step 4 diagnostic results to {out_csv}")

if __name__ == "__main__":
    run_diagnostics()
