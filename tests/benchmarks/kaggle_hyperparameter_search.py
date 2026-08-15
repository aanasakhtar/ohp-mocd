"""
kaggle_hyperparameter_search.py

Comprehensive, Parallelized Grid Search on OHP-MOCD's 4 Core Hyperparameters:
  1. Population Size (pop_size): e.g. [100, 200, 300]
  2. Generations (num_gens): e.g. [100, 200, 300]
  3. Crossover Rate (cross_rate): e.g. [0.60, 0.70, 0.80, 0.90]
  4. Mutation Rate (mut_rate): e.g. [0.30, 0.50, 0.70]
  + Initial Overlap Probability (init_overlap_prob): [0.05, 0.10, 0.20]
  + Initialization Strategy: ["boundary_seeded", "crisp"]

Designed for Kaggle & Local High-Performance Multi-Core Execution.

Usage:
  python tests/benchmarks/kaggle_hyperparameter_search.py --grid_type standard --seeds 10
  python tests/benchmarks/kaggle_hyperparameter_search.py --grid_type full --seeds 15 --datasets Karate Dolphins Polbooks Football
"""

import sys, os, time, argparse, itertools
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
    load_netscience, load_celegans, load_email,
    nicosia_qov_slpa, nicosia_qov_unscaled, shen_modularity_eq, extract_ground_truth
)
from tests.benchmarks.utils.merge import post_hoc_boundary_merge

ALL_LOADERS = {
    "Karate": load_karate,
    "Dolphins": load_dolphins,
    "Lesmis": load_lesmis,
    "Polbooks": load_polbooks,
    "Football": load_football,
    "Netscience": load_netscience,
    "Celegans": load_celegans,
    "Email": load_email,
}

GRID_PRESETS = {
    "quick": {
        "pop_size": [100, 200],
        "num_gens": [100, 200],
        "cross_rate": [0.70, 0.85],
        "mut_rate": [0.30, 0.50],
        "init_overlap_prob": [0.10],
        "init_strategy": ["boundary_seeded"],
    },
    "standard": {
        "pop_size": [100, 200, 300],
        "num_gens": [100, 200, 300],
        "cross_rate": [0.60, 0.75, 0.90],
        "mut_rate": [0.30, 0.50, 0.70],
        "init_overlap_prob": [0.05, 0.10, 0.20],
        "init_strategy": ["boundary_seeded", "crisp"],
    },
    "full": {
        "pop_size": [100, 200, 300, 400],
        "num_gens": [100, 200, 300, 400],
        "cross_rate": [0.50, 0.65, 0.80, 0.95],
        "mut_rate": [0.20, 0.40, 0.60, 0.80],
        "init_overlap_prob": [0.05, 0.10, 0.15, 0.25],
        "init_strategy": ["boundary_seeded", "crisp"],
    }
}

def eval_single_param_run(args):
    net_name, pop, gens, cross, mut, init_prob, strat, edge_list, gt, seed = args
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
        cross_rate=cross,
        mut_rate=mut,
        init_strategy=strat,
        init_overlap_prob=init_prob,
        seed=seed
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
    
    qov_slpa = nicosia_qov_slpa(G, merged_comms)
    eq = shen_modularity_eq(G, merged_comms)
    
    gnmi = 0.0
    if gt is not None:
        comm_fsets = [frozenset(c) for c in merged_comms]
        gnmi = onmi(comm_fsets, gt)
        
    return {
        "Dataset": net_name,
        "PopSize": pop,
        "NumGens": gens,
        "CrossRate": cross,
        "MutRate": mut,
        "InitProb": init_prob,
        "Strategy": strat,
        "Seed": seed,
        "Qov_SLPA": qov_slpa,
        "EQ": eq,
        "gNMI": gnmi,
        "Raw_Comms": len(raw_comms),
        "Merged_Comms": len(merged_comms),
        "Avg_Mems_Node": total_mems / len(nodes),
        "Runtime_Sec": dur,
    }

def run_grid_search(datasets: list[str], grid_type: str, num_seeds: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    param_grid = GRID_PRESETS.get(grid_type, GRID_PRESETS["standard"])
    
    keys = list(param_grid.keys())
    combinations = list(itertools.product(*[param_grid[k] for k in keys]))
    
    print("================================================================================")
    print(f" KAGGLE OHP-MOCD HYPERPARAMETER GRID SEARCH ({grid_type.upper()} PRESET)")
    print(f" Total Unique Configurations: {len(combinations)} | Seeds per Config: {num_seeds}")
    print(f" Target Datasets ({len(datasets)}): {', '.join(datasets)}")
    print("================================================================================")
    
    max_workers = max(1, (os.cpu_count() or 4) - 1)
    all_raw_rows = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for net_name in datasets:
            if net_name not in ALL_LOADERS:
                print(f"Skipping unknown dataset: {net_name}")
                continue
            print(f"\n---> Starting Parameter Sweep on: {net_name} ...")
            G_obj = ALL_LOADERS[net_name]()
            G = G_obj[0] if isinstance(G_obj, tuple) else G_obj
            edge_list = list(G.edges())
            gt = extract_ground_truth(G, net_name)
            
            tasks = []
            for combo in combinations:
                param_dict = dict(zip(keys, combo))
                for seed in range(42, 42 + num_seeds):
                    tasks.append((
                        net_name,
                        param_dict["pop_size"],
                        param_dict["num_gens"],
                        param_dict["cross_rate"],
                        param_dict["mut_rate"],
                        param_dict["init_overlap_prob"],
                        param_dict["init_strategy"],
                        edge_list,
                        gt,
                        seed
                    ))
                    
            print(f"  Submitting {len(tasks)} parallel evaluation tasks across {max_workers} CPU workers...")
            t0 = time.perf_counter()
            futures = [executor.submit(eval_single_param_run, t) for t in tasks]
            results = [f.result() for f in futures]
            dur = time.perf_counter() - t0
            all_raw_rows.extend(results)
            print(f"  [DONE] {net_name} completed in {dur:.2f}s ({dur/60:.2f} min).")
            
    df_raw = pd.DataFrame(all_raw_rows)
    raw_path = out_dir / "kaggle_param_search_raw_trials.csv"
    df_raw.to_csv(raw_path, index=False)
    print(f"\nSaved raw trial results to {raw_path}")
    
    # Aggregated Summary by Hyperparameter Configuration
    group_cols = ["Dataset", "PopSize", "NumGens", "CrossRate", "MutRate", "InitProb", "Strategy"]
    agg_df = df_raw.groupby(group_cols).agg(
        Qov_SLPA_mean=("Qov_SLPA", "mean"),
        Qov_SLPA_std=("Qov_SLPA", "std"),
        Qov_SLPA_peak=("Qov_SLPA", "max"),
        EQ_mean=("EQ", "mean"),
        EQ_peak=("EQ", "max"),
        gNMI_mean=("gNMI", "mean"),
        gNMI_peak=("gNMI", "max"),
        Merged_Comms_mean=("Merged_Comms", "mean"),
        Avg_Mems_Node_mean=("Avg_Mems_Node", "mean"),
        Runtime_mean=("Runtime_Sec", "mean")
    ).reset_index()
    
    agg_path = out_dir / "kaggle_param_search_ranked_summary.csv"
    agg_df.to_csv(agg_path, index=False)
    print(f"Saved aggregated ranked summary to {agg_path}")
    
    # Best Per Dataset Table
    print("\n" + "="*80)
    print(" TOP PERFORMING CONFIGURATIONS PER DATASET (BY NICOSIA Qov & gNMI)")
    print("="*80)
    for net in datasets:
        net_df = agg_df[agg_df["Dataset"] == net]
        if net_df.empty: continue
        best_qov_row = net_df.sort_values(by="Qov_SLPA_mean", ascending=False).iloc[0]
        best_gnmi_row = net_df.sort_values(by="gNMI_mean", ascending=False).iloc[0]
        
        print(f"\nDataset: {net}")
        print(f"  • Best for Nicosia Qov : pop={best_qov_row['PopSize']}, gens={best_qov_row['NumGens']}, cross={best_qov_row['CrossRate']}, mut={best_qov_row['MutRate']}, prob={best_qov_row['InitProb']}, strat={best_qov_row['Strategy']}")
        print(f"    -> Qov: {best_qov_row['Qov_SLPA_mean']:.4f} ± {best_qov_row['Qov_SLPA_std']:.4f} (Peak: {best_qov_row['Qov_SLPA_peak']:.4f}) | Shen EQ: {best_qov_row['EQ_mean']:.4f}")
        if best_gnmi_row['gNMI_mean'] > 0:
            print(f"  • Best for Ground Truth: pop={best_gnmi_row['PopSize']}, gens={best_gnmi_row['NumGens']}, cross={best_gnmi_row['CrossRate']}, mut={best_gnmi_row['MutRate']}, prob={best_gnmi_row['InitProb']}, strat={best_gnmi_row['Strategy']}")
            print(f"    -> gNMI: {best_gnmi_row['gNMI_mean']:.4f} (Peak: {best_gnmi_row['gNMI_peak']:.4f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kaggle Hyperparameter Grid Search for OHP-MOCD")
    parser.add_argument("--grid_type", type=str, choices=["quick", "standard", "full"], default="standard", help="Grid size preset")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds per configuration")
    parser.add_argument("--datasets", nargs="+", default=["Karate", "Dolphins", "Lesmis", "Polbooks", "Football", "Netscience", "Celegans"], help="Datasets to evaluate")
    parser.add_argument("--out_dir", type=str, default=str(REPO_ROOT / "tests" / "benchmarks"), help="Output directory")
    args = parser.parse_args()
    
    run_grid_search(
        datasets=args.datasets,
        grid_type=args.grid_type,
        num_seeds=args.seeds,
        out_dir=Path(args.out_dir)
    )
