"""
run_ultimate_head_to_head_suite.py

The Ultimate, Comprehensive Head-to-Head Comparative Suite:
Executes OHP-MOCD (Boundary-Seeded & Crisp) side-by-side against 4 executed baseline algorithms:
  1. OHP-MOCD (Boundary-Seeded) — Proposed
  2. OHP-MOCD (Crisp) — Proposed
  3. SLPA (Xie & Szymanski, IEEE TKDE 2012)
  4. MCMOEA (Wen et al., IEEE TEVC 2016) — Rust Native
  5. FCCNI (Shang et al., Applied Soft Computing 2024) — Pure Python
  6. Çetin 2022 (Çetin & Amrahov, Kybernetika 2022) — Pure Python

Across 13 Comprehensive Datasets covering all structural spectrums:
  - Real-World (Small, Medium, Large, Dense, Sparse, Biological, Social, Communication):
      Karate, Dolphins, Lesmis, Polbooks, Football, Netscience, Celegans, Email-EuCore
  - Synthetic LFR Benchmarks (Small, Large, Dense, Sparse, High-Noise, High-Overlap):
      LFR-Small-Dense (1k, avg_k=15, mu=0.1, on=300, om=2)
      LFR-Small-Sparse (1k, avg_k=5, mu=0.1, on=300, om=2)
      LFR-Medium-Noise (1k, avg_k=10, mu=0.3, on=300, om=2)
      LFR-High-Overlap (1k, avg_k=10, mu=0.1, on=300, om=3)
      LFR-Large-Complex (5k, avg_k=10, mu=0.1, on=1500, om=2)

Parallelism: Safe ProcessPoolExecutor with max_workers = os.cpu_count() - 1.
Outputs: tests/benchmarks/ultimate_head_to_head_summary.csv
"""

import sys
import time
import os
import math
import numpy as np
import pandas as pd
import networkx as nx
import concurrent.futures
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pymocd
from evaluation.metrics import onmi
from tests.benchmarks.baselines import run_slpa, run_mcmoea, run_fccni, run_cetin2022
from tests.benchmarks.run_paper_comparative_suite import (
    nicosia_qov, shen_modularity_eq, overlapping_coverage_cetin,
    load_karate, load_dolphins, load_lesmis, load_polbooks,
    load_football, load_netscience, load_celegans, load_email
)
from tests.benchmarks.run_lfr_benchmark import generate_lfr_overlapping

BENCH_DIR = REPO_ROOT / "tests" / "benchmarks"
N_SEEDS = 3
N_WORKERS = max(1, (os.cpu_count() or 4) - 1)

def evaluate_partition(G: nx.Graph, comms: list[frozenset], gt: list[frozenset] = None) -> dict:
    comm_sets = [set(c) for c in comms if c]
    if not comm_sets:
        comm_sets = [{n} for n in G.nodes()]

    qov = nicosia_qov(G, comm_sets)
    eq  = shen_modularity_eq(G, comm_sets)
    cov = overlapping_coverage_cetin(G, comm_sets)
    gnmi = onmi(comms, gt) if gt else float("nan")

    return {"Qov": qov, "EQ": eq, "Coverage": cov, "gNMI": gnmi}

def run_dataset_algorithm_task(task_tuple: tuple) -> dict:
    net_name, algo_name, seed, edge_list, gt = task_tuple
    G = nx.Graph(edge_list)
    nodes = sorted(G.nodes())
    node_map = {n: i for i, n in enumerate(nodes)}
    rev_map  = {i: n for i, n in enumerate(nodes)}
    H = nx.relabel_nodes(G, node_map, copy=True)

    t0 = time.perf_counter()
    comms = []

    try:
        if algo_name == "OHP-MOCD (BS)":
            res = pymocd.ohpmocd(
                H,
                pop_size=100,
                num_gens=100,
                cross_rate=0.8,
                mut_rate=0.5,
                init_strategy="boundary_seeded",
                init_overlap_prob=0.10,
                seed=seed
            )
            cd = {}
            for n_idx, cl in res.items():
                orig = rev_map.get(n_idx, n_idx)
                if isinstance(cl, (int, np.integer)): cl = [cl]
                for c in cl: cd.setdefault(c, set()).add(orig)
            comms = [frozenset(s) for s in cd.values()]

        elif algo_name == "OHP-MOCD (Crisp)":
            res = pymocd.ohpmocd(
                H,
                pop_size=100,
                num_gens=100,
                cross_rate=0.8,
                mut_rate=0.5,
                init_strategy="crisp",
                seed=seed
            )
            cd = {}
            for n_idx, cl in res.items():
                orig = rev_map.get(n_idx, n_idx)
                if isinstance(cl, (int, np.integer)): cl = [cl]
                for c in cl: cd.setdefault(c, set()).add(orig)
            comms = [frozenset(s) for s in cd.values()]

        elif algo_name == "SLPA (Xie 2012)":
            comms = run_slpa(G, T=100, r=0.05, seed=seed)

        elif algo_name == "MCMOEA (Wen 2016)":
            comms = run_mcmoea(G, pop_size=50, num_gens=50)

        elif algo_name == "FCCNI (Shang 2024)":
            comms = run_fccni(G, max_iters=50, tau=0.35, seed=seed)

        elif algo_name == "Çetin (2022)":
            comms = run_cetin2022(G, overlap_threshold=0.30, seed=seed)

    except Exception as e:
        dur = time.perf_counter() - t0
        return {"net_name": net_name, "algo": algo_name, "seed": seed,
                "Qov": float("nan"), "EQ": float("nan"), "Coverage": float("nan"),
                "gNMI": float("nan"), "Time": dur, "error": str(e)}

    dur = time.perf_counter() - t0
    metrics = evaluate_partition(G, comms, gt)
    metrics.update({"net_name": net_name, "algo": algo_name, "seed": seed, "Time": dur, "error": None})
    return metrics

def main():
    print("=" * 80)
    print(" ULTIMATE COMPREHENSIVE HEAD-TO-HEAD BENCHMARK SUITE")
    print(f" CPU Parallelism: {N_WORKERS} worker processes active")
    print("=" * 80)

    # 1. Real-World Networks
    print("\n[1/2] Loading Real-World Datasets...")
    rw_datasets = {}
    rw_loaders = {
        "Karate": load_karate,
        "Dolphins": load_dolphins,
        "Lesmis": load_lesmis,
        "Polbooks": load_polbooks,
        "Football": load_football,
        "Netscience": load_netscience,
        "Celegans": load_celegans,
        "Email": load_email,
    }
    for name, ldr in rw_loaders.items():
        try:
            G = ldr()
            rw_datasets[name] = (G, None)
            print(f"   Loaded {name:<12}: {G.number_of_nodes():>5} nodes, {G.number_of_edges():>6} edges")
        except Exception as e:
            print(f"   Failed loading {name}: {e}")

    # Extract Football ground truth
    fb_G, _ = rw_datasets["Football"]
    comm_map = {}
    for n, d in fb_G.nodes(data=True):
        val = d.get('value', d.get('club', None))
        if val is not None: comm_map.setdefault(val, set()).add(n)
    gt_fb = [frozenset(c) for c in comm_map.values()] if comm_map else None
    rw_datasets["Football"] = (fb_G, gt_fb)

    # 2. Comprehensive LFR Datasets across structural spectrums
    print("\n[2/2] Generating Synthetic LFR Benchmark Spectrum...")
    lfr_configs = [
        ("LFR-Small-Dense",  {"n": 1000, "mu": 0.1, "on": 300, "om": 2, "tau1": 3, "tau2": 1.5, "avg_deg": 15, "max_deg": 50, "min_c": 10, "max_c": 50}),
        ("LFR-Small-Sparse", {"n": 1000, "mu": 0.1, "on": 300, "om": 2, "tau1": 3, "tau2": 1.5, "avg_deg": 5,  "max_deg": 25, "min_c": 10, "max_c": 50}),
        ("LFR-Medium-Noise", {"n": 1000, "mu": 0.3, "on": 300, "om": 2, "tau1": 3, "tau2": 1.5, "avg_deg": 10, "max_deg": 50, "min_c": 10, "max_c": 50}),
        ("LFR-High-Overlap", {"n": 1000, "mu": 0.1, "on": 300, "om": 3, "tau1": 3, "tau2": 1.5, "avg_deg": 10, "max_deg": 50, "min_c": 10, "max_c": 50}),
        ("LFR-Large-Complex",{"n": 5000, "mu": 0.1, "on": 1500,"om": 2, "tau1": 3, "tau2": 1.5, "avg_deg": 10, "max_deg": 50, "min_c": 10, "max_c": 50}),
    ]
    for lbl, cfg in lfr_configs:
        try:
            G, gt, _ = generate_lfr_overlapping(cfg, seed=42)
            rw_datasets[lbl] = (G, gt)
            print(f"   Generated {lbl:<18}: {G.number_of_nodes():>5} nodes, {G.number_of_edges():>6} edges")
        except Exception as e:
            print(f"   Failed generating {lbl}: {e}")

    algos = ["OHP-MOCD (BS)", "OHP-MOCD (Crisp)", "SLPA (Xie 2012)", "MCMOEA (Wen 2016)", "FCCNI (Shang 2024)", "Çetin (2022)"]

    tasks = []
    for dname, (G, gt) in rw_datasets.items():
        edge_list = list(G.edges())
        for algo in algos:
            for seed in range(N_SEEDS):
                tasks.append((dname, algo, seed, edge_list, gt))

    print(f"\nTotal tasks in parallel queue: {len(tasks)} ({len(rw_datasets)} datasets x {len(algos)} algos x {N_SEEDS} seeds)")
    print(f"Dispatching across {N_WORKERS} worker processes...\n")

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(run_dataset_algorithm_task, t): t for t in tasks}
        done = 0
        for f in concurrent.futures.as_completed(futs):
            done += 1
            res = f.result()
            results.append(res)
            if done % 20 == 0 or done == len(tasks):
                print(f"  [{done:>3}/{len(tasks)}] {res['net_name']:<18} | {res['algo']:<18} | Qov={res['Qov']:.4f} | EQ={res['EQ']:.4f} | Time={res['Time']:.2f}s")

    df = pd.DataFrame(results)
    df.to_csv(BENCH_DIR / "ultimate_raw_runs.csv", index=False)

    summary = (
        df.groupby(["net_name", "algo"])
        .agg(
            Qov_max=("Qov", "max"), Qov_mean=("Qov", "mean"),
            EQ_max=("EQ", "max"), EQ_mean=("EQ", "mean"),
            Coverage_mean=("Coverage", "mean"),
            gNMI_max=("gNMI", "max"), gNMI_mean=("gNMI", "mean"),
            Time_mean=("Time", "mean")
        )
        .reset_index()
    )

    out_csv = BENCH_DIR / "ultimate_head_to_head_summary.csv"
    summary.to_csv(out_csv, index=False)
    print(f"\nSaved master summary to: {out_csv}")

    print("\n" + "=" * 105)
    print(" MASTER COMPARATIVE SUMMARY: NICOSIA Qov ACROSS ALL ALGORITHMS")
    print("=" * 105)
    pivot_qov = summary.pivot(index="net_name", columns="algo", values="Qov_max")
    print(pivot_qov.to_string())

    print("\n" + "=" * 105)
    print(" MASTER COMPARATIVE SUMMARY: gNMI ACROSS GROUND-TRUTH NETWORKS")
    print("=" * 105)
    pivot_gnmi = summary.pivot(index="net_name", columns="algo", values="gNMI_max")
    print(pivot_gnmi.dropna(how='all').to_string())

    print("\nULTIMATE HEAD-TO-HEAD BENCHMARK SUITE COMPLETE.")

if __name__ == "__main__":
    main()
