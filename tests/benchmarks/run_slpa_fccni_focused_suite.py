"""
run_slpa_fccni_focused_suite.py

Focused Benchmark Suite strictly targeting SLPA (2011) and FCCNI (2024).
Evaluates OHP-MOCD over N independent seeds (default: 15 seeds, configurable).

Target Papers & Metrics:
  1. SLPA (Xie & Szymanski, 2011): Nicosia Qov (Linear Clamp f(r) = max(0, min(1, 60r - 30)))
     Datasets: Karate, Dolphins, Lesmis, Polbooks, Football, Netscience, Celegans, Email
  2. FCCNI (Shang et al., 2024): Ground Truth gNMI & Shen Extended Modularity EQ
     Datasets: Karate, Dolphins, Polbooks, Football

Saves:
  - tests/benchmarks/slpa_fccni_focused_results.csv
  - tests/benchmarks/slpa_fccni_master_summary.csv
"""

import sys, os, time, argparse
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

# Ground truth reported baselines
SLPA_REPORTED = {
    "Karate": {"mean": 0.65, "std": 0.21, "r": 0.33},
    "Dolphins": {"mean": 0.76, "std": 0.03, "r": 0.45},
    "Lesmis": {"mean": 0.78, "std": 0.03, "r": 0.45},
    "Polbooks": {"mean": 0.83, "std": 0.01, "r": 0.45},
    "Football": {"mean": 0.70, "std": 0.01, "r": 0.45},
    "Netscience": {"mean": 0.85, "std": 0.01, "r": 0.45},
    "Celegans": {"mean": 0.31, "std": 0.22, "r": 0.35},
    "Email": {"mean": 0.64, "std": 0.03, "r": 0.45},
}

FCCNI_REPORTED = {
    "Karate": {"gNMI_max": 1.0000, "gNMI_avg": 0.9650, "std": 0.0450, "lambda": 0.5},
    "Dolphins": {"gNMI_max": 1.0000, "gNMI_avg": 0.9720, "std": 0.0380, "lambda": 0.5},
    "Polbooks": {"gNMI_max": 0.9234, "gNMI_avg": 0.8840, "std": 0.0520, "lambda": 0.0},
    "Football": {"gNMI_max": 0.8041, "gNMI_avg": 0.7820, "std": 0.0210, "lambda": 0.5},
}

DATASET_LOADERS = {
    "Karate": load_karate,
    "Dolphins": load_dolphins,
    "Lesmis": load_lesmis,
    "Polbooks": load_polbooks,
    "Football": load_football,
    "Netscience": load_netscience,
    "Celegans": load_celegans,
    "Email": load_email,
}

def eval_single_seed_run(args):
    net_name, init_strategy, edge_list, gt, pop_size, num_gens, cross_rate, mut_rate, init_overlap_prob, seed_val = args
    G = nx.Graph(edge_list)
    nodes = list(G.nodes())
    node_map = {n: i for i, n in enumerate(nodes)}
    rev_map = {i: n for i, n in enumerate(nodes)}
    H = nx.relabel_nodes(G, node_map, copy=True)
    
    t0 = time.perf_counter()
    dict_res = pymocd.ohpmocd(
        H,
        pop_size=pop_size,
        num_gens=num_gens,
        cross_rate=cross_rate,
        mut_rate=mut_rate,
        init_strategy=init_strategy,
        init_overlap_prob=init_overlap_prob,
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
    
    qov_slpa = nicosia_qov_slpa(G, merged_comms)
    qov_unscaled = nicosia_qov_unscaled(G, merged_comms)
    eq = shen_modularity_eq(G, merged_comms)
    
    onmi_val = 0.0
    if gt is not None:
        comm_frozensets = [frozenset(c) for c in merged_comms]
        onmi_val = onmi(comm_frozensets, gt)
        
    return {
        "net_name": net_name,
        "init_strategy": init_strategy,
        "seed": seed_val,
        "Qov_SLPA": qov_slpa,
        "Qov_Unscaled": qov_unscaled,
        "EQ": eq,
        "gNMI": onmi_val,
        "raw_comms": len(raw_comms),
        "merged_comms": len(merged_comms),
        "avg_mems": total_mems / len(nodes),
        "time": dur,
    }

def run_suite(num_seeds: int = 15, pop_size: int = 100, num_gens: int = 100, cross_rate: float = 0.8, mut_rate: float = 0.5, init_overlap_prob: float = 0.10):
    print("================================================================================")
    print(f" RUNNING FOCUSED COMPARATIVE SUITE: SLPA (2011) & FCCNI (2024) [{num_seeds} SEEDS]")
    print(f" Configuration: pop={pop_size}, gens={num_gens}, cross={cross_rate}, mut={mut_rate}, init_prob={init_overlap_prob}")
    print("================================================================================")
    
    max_workers = max(1, (os.cpu_count() or 4) - 1)
    detailed_rows = []
    summary_rows = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 1. SLPA Comparisons
        print("\n--- 1. Evaluating SLPA Comparison Networks (Metric: Nicosia Qov) ---")
        for net_name in ["Karate", "Dolphins", "Lesmis", "Polbooks", "Football", "Netscience", "Celegans", "Email"]:
            print(f"  -> Dataset: {net_name} ...")
            loader = DATASET_LOADERS[net_name]
            G_obj = loader()
            G = G_obj[0] if isinstance(G_obj, tuple) else G_obj
            edge_list = list(G.edges())
            gt = extract_ground_truth(G, net_name)
            
            for strat in ["boundary_seeded", "crisp"]:
                tasks = [
                    (net_name, strat, edge_list, gt, pop_size, num_gens, cross_rate, mut_rate, init_overlap_prob, seed)
                    for seed in range(42, 42 + num_seeds)
                ]
                futures = [executor.submit(eval_single_seed_run, t) for t in tasks]
                res_list = [f.result() for f in futures]
                detailed_rows.extend(res_list)
                
                qovs = [r["Qov_SLPA"] for r in res_list]
                eqs = [r["EQ"] for r in res_list]
                mems = [r["avg_mems"] for r in res_list]
                comms = [r["merged_comms"] for r in res_list]
                times = [r["time"] for r in res_list]
                
                slpa_info = SLPA_REPORTED.get(net_name, {})
                slpa_mean = slpa_info.get("mean", 0.0)
                slpa_std = slpa_info.get("std", 0.0)
                
                summary_rows.append({
                    "Target_Paper": "SLPA (2011)",
                    "Dataset": net_name,
                    "Nodes": G.number_of_nodes(),
                    "Edges": G.number_of_edges(),
                    "Strategy": strat,
                    "Metric": "Nicosia Qov",
                    "Paper_Reported_Score": f"{slpa_mean:.4f} ± {slpa_std:.4f}",
                    "OHP_MOCD_Mean": np.mean(qovs),
                    "OHP_MOCD_Std": np.std(qovs),
                    "OHP_MOCD_Peak": np.max(qovs),
                    "Delta_vs_Reported": np.mean(qovs) - slpa_mean,
                    "Shen_EQ_Mean": np.mean(eqs),
                    "Merged_Comms_Mean": np.mean(comms),
                    "Avg_Mems_Per_Node": np.mean(mems),
                    "Time_Per_Seed": np.mean(times),
                })
                print(f"     [{strat:15s}] Qov: {np.mean(qovs):.4f} ± {np.std(qovs):.4f} (peak {np.max(qovs):.4f}) vs SLPA {slpa_mean:.4f} | Comms: {np.mean(comms):.1f} | Time: {np.mean(times):.2f}s")
                
        # 2. FCCNI Comparisons
        print("\n--- 2. Evaluating FCCNI Comparison Networks (Metric: gNMI) ---")
        for net_name in ["Karate", "Dolphins", "Polbooks", "Football"]:
            print(f"  -> Dataset: {net_name} ...")
            loader = DATASET_LOADERS[net_name]
            G_obj = loader()
            G = G_obj[0] if isinstance(G_obj, tuple) else G_obj
            edge_list = list(G.edges())
            gt = extract_ground_truth(G, net_name)
            
            for strat in ["boundary_seeded", "crisp"]:
                tasks = [
                    (net_name, strat, edge_list, gt, pop_size, num_gens, cross_rate, mut_rate, init_overlap_prob, seed)
                    for seed in range(42, 42 + num_seeds)
                ]
                futures = [executor.submit(eval_single_seed_run, t) for t in tasks]
                res_list = [f.result() for f in futures]
                
                gnmis = [r["gNMI"] for r in res_list]
                eqs = [r["EQ"] for r in res_list]
                times = [r["time"] for r in res_list]
                
                fccni_info = FCCNI_REPORTED.get(net_name, {})
                fccni_max = fccni_info.get("gNMI_max", 0.0)
                fccni_avg = fccni_info.get("gNMI_avg", 0.0)
                
                summary_rows.append({
                    "Target_Paper": "FCCNI (2024)",
                    "Dataset": net_name,
                    "Nodes": G.number_of_nodes(),
                    "Edges": G.number_of_edges(),
                    "Strategy": strat,
                    "Metric": "gNMI",
                    "Paper_Reported_Score": f"max={fccni_max:.4f}, avg={fccni_avg:.4f}",
                    "OHP_MOCD_Mean": np.mean(gnmis),
                    "OHP_MOCD_Std": np.std(gnmis),
                    "OHP_MOCD_Peak": np.max(gnmis),
                    "Delta_vs_Reported": np.mean(gnmis) - fccni_avg,
                    "Shen_EQ_Mean": np.mean(eqs),
                    "Merged_Comms_Mean": np.mean([r['merged_comms'] for r in res_list]),
                    "Avg_Mems_Per_Node": np.mean([r['avg_mems'] for r in res_list]),
                    "Time_Per_Seed": np.mean(times),
                })
                print(f"     [{strat:15s}] gNMI: {np.mean(gnmis):.4f} ± {np.std(gnmis):.4f} (peak {np.max(gnmis):.4f}) vs FCCNI max={fccni_max:.4f} | Shen EQ: {np.mean(eqs):.4f}")
                
    df_det = pd.DataFrame(detailed_rows)
    df_det.to_csv(REPO_ROOT / "tests" / "benchmarks" / "slpa_fccni_focused_results.csv", index=False)
    
    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(REPO_ROOT / "tests" / "benchmarks" / "slpa_fccni_master_summary.csv", index=False)
    
    print("\n[DONE] Saved focused results to:")
    print(f"  • {REPO_ROOT / 'tests' / 'benchmarks' / 'slpa_fccni_focused_results.csv'}")
    print(f"  • {REPO_ROOT / 'tests' / 'benchmarks' / 'slpa_fccni_master_summary.csv'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run focused benchmark suite for SLPA and FCCNI")
    parser.add_argument("--seeds", type=int, default=15, help="Number of independent seeds")
    parser.add_argument("--pop", type=int, default=100, help="Population size")
    parser.add_argument("--gens", type=int, default=100, help="Number of generations")
    parser.add_argument("--cross", type=float, default=0.8, help="Crossover rate")
    parser.add_argument("--mut", type=float, default=0.5, help="Mutation rate")
    parser.add_argument("--init_prob", type=float, default=0.10, help="Initial overlap probability")
    args = parser.parse_args()
    
    run_suite(
        num_seeds=args.seeds,
        pop_size=args.pop,
        num_gens=args.gens,
        cross_rate=args.cross,
        mut_rate=args.mut,
        init_overlap_prob=args.init_prob
    )
