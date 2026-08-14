"""
lfr_grid_search_tuning.py

Multi-core parallelized Grid Search hyperparameter tuning script for OHP-MOCD
on LFR synthetic benchmark networks.

Discovers optimal parameter tuples (init_strategy, alpha, supp_th, rem_th, merge_th)
for each standard LFR benchmark configuration.
"""

import sys, os
sys.path.insert(0, os.path.abspath('.'))

import time
import json
import itertools
import numpy as np
import networkx as nx
import pandas as pd
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

# Grid Search Search Space
GRID_STRATEGIES = ["boundary_seeded", "crisp"]
GRID_ALPHAS = [0.00, 0.25, 0.50, 0.75, 1.00]
GRID_SUPP_THS = [0.15, 0.25, 0.35, 0.55]
GRID_REM_THS = [0.05, 0.08, 0.15, 0.25]
GRID_MERGE_THS = [None, 0.35, 0.50]

def generate_lfr_network(cfg_params: dict) -> tuple[nx.Graph, list[frozenset]]:
    """Generates LFR graph and extracts ground truth communities."""
    G = LFR_benchmark_graph(**cfg_params)
    
    # Extract ground truth communities from node attributes
    gt_dict = {}
    for n, d in G.nodes(data=True):
        comm = d.get('community', set())
        if isinstance(comm, (int, np.integer)):
            comm = {int(comm)}
        elif isinstance(comm, (set, list, tuple)):
            comm = set(comm)
        for cid in comm:
            gt_dict.setdefault(cid, set()).add(n)
            
    gt = [frozenset(c) for c in gt_dict.values() if c]
    return G, gt

def evaluate_lfr_grid_candidate(task_tuple: tuple) -> dict:
    """Worker task evaluating a single grid candidate on an LFR network."""
    cfg_name, edge_list, gt, strat, alpha, supp_th, rem_th, merge_th = task_tuple
    
    G = nx.Graph(edge_list)
    nodes = list(G.nodes())
    node_map = {n: i for i, n in enumerate(nodes)}
    rev_map = {i: n for i, n in enumerate(nodes)}
    H = nx.relabel_nodes(G, node_map, copy=True)
    
    t0 = time.perf_counter()
    try:
        dict_res = pymocd.ohpmocd(
            H,
            init_strategy=strat,
            init_overlap_prob=0.15,
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
            if isinstance(comm_list, (int, np.integer)):
                comm_list = [comm_list]
            for cid in comm_list:
                comm_dict.setdefault(cid, set()).add(orig_node)
        comms = list(comm_dict.values())
        
        if merge_th is not None:
            comms = post_hoc_boundary_merge(G, comms, merge_threshold=merge_th)
            
        comm_frozensets = [frozenset(c) for c in comms]
        gnmi_score = onmi(comm_frozensets, gt)
        qov_slpa = nicosia_qov_slpa_scaled(G, comms)
        eq_score = shen_modularity_eq(G, comms)
        
        return {
            "Config": cfg_name,
            "Strategy": strat,
            "Alpha": alpha,
            "Supp_Th": supp_th,
            "Rem_Th": rem_th,
            "Merge_Th": str(merge_th),
            "gNMI": gnmi_score,
            "Qov_SLPA": qov_slpa,
            "EQ": eq_score,
            "Time": dur,
            "Status": "OK"
        }
    except Exception as err:
        return {
            "Config": cfg_name,
            "Strategy": strat,
            "Alpha": alpha,
            "Supp_Th": supp_th,
            "Rem_Th": rem_th,
            "Merge_Th": str(merge_th),
            "gNMI": 0.0,
            "Qov_SLPA": 0.0,
            "EQ": 0.0,
            "Time": 0.0,
            "Status": f"ERR: {err}"
        }

def main():
    print("=================================================================")
    print(" STARTING MULTI-CORE LFR SYNTHETIC BENCHMARK GRID SEARCH TUNING ")
    print("=================================================================\n")
    
    all_results = []
    optimal_params = {}
    
    max_workers = max(1, (os.cpu_count() or 4) - 1)
    print(f"Parallel Execution active with max_workers = {max_workers}\n")
    
    for cfg_name, cfg_params in LFR_CONFIGS.items():
        print(f"--- Generating LFR Network: {cfg_name} ---")
        t_gen = time.perf_counter()
        G, gt = generate_lfr_network(cfg_params)
        print(f"    Graph generated in {time.perf_counter() - t_gen:.2f}s: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, {len(gt)} ground-truth communities.")
        
        edge_list = list(G.edges())
        
        # Build Cartesian combinations
        cartesian_combinations = list(itertools.product(
            GRID_STRATEGIES, GRID_ALPHAS, GRID_SUPP_THS, GRID_REM_THS, GRID_MERGE_THS
        ))
        print(f"    Evaluating {len(cartesian_combinations)} parameter candidates in parallel...")
        
        tasks = [
            (cfg_name, edge_list, gt, strat, alpha, supp_th, rem_th, merge_th)
            for strat, alpha, supp_th, rem_th, merge_th in cartesian_combinations
        ]
        
        config_results = []
        t_start = time.perf_counter()
        completed = 0
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(evaluate_lfr_grid_candidate, task): task for task in tasks}
            
            for future in as_completed(future_map):
                res = future.result()
                config_results.append(res)
                all_results.append(res)
                completed += 1
                if completed % 100 == 0 or completed == len(tasks):
                    elapsed = time.perf_counter() - t_start
                    print(f"    [Progress] {completed}/{len(tasks)} candidates evaluated ({completed/len(tasks)*100:.1f}%) | Elapsed: {elapsed:.1f}s")
                    
        df_cfg = pd.DataFrame(config_results)
        best_gnmi_row = df_cfg.loc[df_cfg['gNMI'].idxmax()]
        best_eq_row = df_cfg.loc[df_cfg['EQ'].idxmax()]
        
        print(f"\n---> Best gNMI for {cfg_name} ({best_gnmi_row['gNMI']:.4f}): strat={best_gnmi_row['Strategy']}, alpha={best_gnmi_row['Alpha']}, supp_th={best_gnmi_row['Supp_Th']}, rem_th={best_gnmi_row['Rem_Th']}, merge_th={best_gnmi_row['Merge_Th']}")
        print(f"---> Best EQ for {cfg_name}   ({best_eq_row['EQ']:.4f}): strat={best_eq_row['Strategy']}, alpha={best_eq_row['Alpha']}, supp_th={best_eq_row['Supp_Th']}, rem_th={best_eq_row['Rem_Th']}, merge_th={best_eq_row['Merge_Th']}\n")
        
        optimal_params[cfg_name] = {
            "gNMI_optimal": {
                "strat": str(best_gnmi_row['Strategy']),
                "alpha": float(best_gnmi_row['Alpha']),
                "supp_th": float(best_gnmi_row['Supp_Th']),
                "rem_th": float(best_gnmi_row['Rem_Th']),
                "merge_th": None if str(best_gnmi_row['Merge_Th']) == 'None' else float(best_gnmi_row['Merge_Th']),
                "max_gNMI": float(best_gnmi_row['gNMI']),
            },
            "EQ_optimal": {
                "strat": str(best_eq_row['Strategy']),
                "alpha": float(best_eq_row['Alpha']),
                "supp_th": float(best_eq_row['Supp_Th']),
                "rem_th": float(best_eq_row['Rem_Th']),
                "merge_th": None if str(best_eq_row['Merge_Th']) == 'None' else float(best_eq_row['Merge_Th']),
                "max_EQ": float(best_eq_row['EQ']),
            }
        }
        
    df_all = pd.DataFrame(all_results)
    csv_out = BENCH_DIR / "lfr_grid_search_full_results.csv"
    df_all.to_csv(csv_out, index=False)
    print(f"Saved full grid search results to: {csv_out}")
    
    json_out = BENCH_DIR / "lfr_optimal_params.json"
    with open(json_out, "w") as f:
        json.dump(optimal_params, f, indent=4)
    print(f"Saved optimal parameters to: {json_out}\n")
    print("ALL LFR GRID SEARCH TUNING COMPLETED SUCCESSFULLY.")

if __name__ == "__main__":
    main()
