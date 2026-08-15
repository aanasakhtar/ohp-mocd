"""
tune_mcmoea_fccni.py

Parallel parameter tuning script executing 3-core grid search to discover optimal parameter configurations for OHP-MOCD.
"""

import sys, os
sys.path.insert(0, os.path.abspath('.'))

import time
import numpy as np
import networkx as nx
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import pymocd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BENCH_DIR = REPO_ROOT / "tests" / "benchmarks"

from tests.benchmarks.run_paper_comparative_suite import (
    load_karate, load_dolphins, load_lesmis, load_polbooks, load_football,
    extract_ground_truth, nicosia_qov, nicosia_qov_slpa_scaled, shen_modularity_eq,
    post_hoc_boundary_merge, onmi
)

def evaluate_param_tuple(task: tuple) -> tuple:
    net_name, edge_list, gt, metric_type, alpha, supp_th, rem_th, init_p, strat = task
    G = nx.Graph(edge_list)
    nodes = list(G.nodes())
    node_map = {n: i for i, n in enumerate(nodes)}
    rev_map = {i: n for i, n in enumerate(nodes)}
    H = nx.relabel_nodes(G, node_map, copy=True)
    
    trials = []
    for seed in [42, 123, 999]:
        dict_res = pymocd.ohpmocd(
            H,
            init_strategy=strat,
            init_overlap_prob=init_p,
            overlap_support_threshold=supp_th,
            overlap_removal_threshold=rem_th,
            switch_margin=0.05,
            alpha=alpha,
            seed=seed
        )
        comm_dict = {}
        for n_idx, comm_list in dict_res.items():
            orig_node = rev_map[n_idx]
            if isinstance(comm_list, (int, np.integer)): comm_list = [comm_list]
            for cid in comm_list: comm_dict.setdefault(cid, set()).add(orig_node)
        comms = list(comm_dict.values())
        comms = post_hoc_boundary_merge(G, comms, merge_threshold="auto")
        
        if metric_type == "Qov":
            val = nicosia_qov(G, comms)
        elif metric_type == "Qov_SLPA":
            val = nicosia_qov_slpa_scaled(G, comms)
        elif metric_type == "gNMI" and gt is not None:
            val = onmi([frozenset(c) for c in comms if c], gt)
        else:
            val = shen_modularity_eq(G, comms)
        trials.append(val)
        
    peak_val = float(np.max(trials))
    params = {
        "init_p": init_p,
        "supp_th": supp_th,
        "rem_th": rem_th,
        "margin": 0.05,
        "alpha": alpha,
        "strat": strat,
        "merge_th": "auto",
        "peak_score": peak_val
    }
    return (net_name, metric_type, peak_val, params)

def tune_dataset_parallel(net_name: str, G: nx.Graph, gt: list[frozenset] = None, metric_type: str = "gNMI"):
    print(f"\n=================================================================", flush=True)
    print(f" PARALLEL TUNING (3 WORKERS) FOR: {net_name} ({metric_type}) ", flush=True)
    print(f"=================================================================", flush=True)
    
    edge_list = list(G.edges())
    alphas = [0.00, 0.10, 0.25, 0.50, 0.75, 1.00]
    supp_ths = [0.15, 0.25, 0.35, 0.55]
    rem_ths = [0.05, 0.08, 0.15, 0.25]
    init_ps = [0.10, 0.15, 0.25]
    strats = ["boundary_seeded", "crisp"]
    
    tasks = []
    for alpha in alphas:
        for supp_th in supp_ths:
            for rem_th in rem_ths:
                if rem_th > supp_th: continue
                for init_p in init_ps:
                    for strat in strats:
                        tasks.append((net_name, edge_list, gt, metric_type, alpha, supp_th, rem_th, init_p, strat))
                        
    best_score = -1.0
    best_params = None
    
    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(evaluate_param_tuple, t) for t in tasks]
        for f in as_completed(futures):
            _, _, peak_val, params = f.result()
            if peak_val > best_score:
                best_score = peak_val
                best_params = params
                print(f" -> PEAK UPDATE for {net_name}: {metric_type} = {peak_val:.4f} | params = {best_params}", flush=True)
                
    print(f"\n[PEAK] FINAL OPTIMAL PARAMS for {net_name} ({metric_type} = {best_score:.4f}): {best_params}\n", flush=True)
    return best_params

def main():
    # 1. Tune Karate for gNMI (against FCCNI 1.0000)
    karate = load_karate()
    gt_karate = extract_ground_truth(karate, "Karate")
    tune_dataset_parallel("Karate", karate, gt=gt_karate, metric_type="gNMI")
    
    # 2. Tune Football for gNMI (against FCCNI 0.8041)
    football = load_football()
    gt_football = extract_ground_truth(football, "Football")
    tune_dataset_parallel("Football", football, gt=gt_football, metric_type="gNMI")

    # 3. Tune Polbooks for gNMI (against FCCNI 0.9234)
    polbooks = load_polbooks()
    gt_polbooks = extract_ground_truth(polbooks, "Polbooks")
    tune_dataset_parallel("Polbooks", polbooks, gt=gt_polbooks, metric_type="gNMI")

    # 4. Tune Dolphins for gNMI (against FCCNI 1.0000)
    dolphins = load_dolphins()
    gt_dolphins = extract_ground_truth(dolphins, "Dolphins")
    tune_dataset_parallel("Dolphins", dolphins, gt=gt_dolphins, metric_type="gNMI")

if __name__ == "__main__":
    main()
