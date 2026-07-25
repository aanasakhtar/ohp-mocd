"""
run_init_experiments.py — Experiment runner for comparing population initialization strategies in OHP-MOCD.

Evaluates Crisp, RandomOverlap, and BoundarySeeded initialization strategies across multiple random seeds
on LFR Disjoint, LFR Overlapping, and DBLP benchmark datasets.

Outputs:
  - src/core/algorithms/data/init_strategy_summary.csv
  - src/core/algorithms/data/init_strategy_convergence.csv
"""

import csv
import time
from pathlib import Path
import networkx as nx
import pymocd

from load_lfr import load_lfr_disjoint, load_lfr_overlapping
from load_dblp import load_dblp


DATA_DIR = Path(__file__).parent


def evaluate_ground_truth_metrics(G: nx.Graph, ground_truth_communities: list[frozenset], pred_partition: dict):
    """Computes NMI, AMI, and ARI between ground truth community sets and predicted primary communities."""
    if not ground_truth_communities:
        return 0.0, 0.0, 0.0

    # Build ground truth node -> community id mapping
    gt_map = {}
    for cid, cmty in enumerate(ground_truth_communities):
        for node in cmty:
            if node not in gt_map:
                gt_map[node] = cid

    nodes = [n for n in G.nodes() if n in gt_map and n in pred_partition]
    if not nodes:
        return 0.0, 0.0, 0.0

    y_true = [gt_map[n] for n in nodes]
    
    # Extract primary community for predicted partition
    y_pred = []
    for n in nodes:
        val = pred_partition[n]
        if isinstance(val, list):
            y_pred.append(val[0] if val else -1)
        else:
            y_pred.append(val)

    nmi, ami, ari = pymocd.gt_metrics(y_true, y_pred)
    return nmi, ami, ari


def compute_overlap_stats(partition: dict):
    """Computes overlap statistics: number of overlapping nodes, average memberships, and max memberships."""
    num_overlapping = 0
    total_memberships = 0
    max_memberships = 0

    for val in partition.values():
        if isinstance(val, list):
            count = len(val)
        else:
            count = 1
        
        if count > 1:
            num_overlapping += 1
        total_memberships += count
        if count > max_memberships:
            max_memberships = count

    n = max(len(partition), 1)
    avg_memberships = total_memberships / n
    return num_overlapping, avg_memberships, max_memberships


def run_experiment_suite(num_seeds: int = 20, num_gens: int = 50, pop_size: int = 50):
    print("=" * 70)
    print("STARTING OHP-MOCD INITIALIZATION STRATEGY EXPERIMENTS")
    print(f"Configuration: num_seeds={num_seeds}, num_gens={num_gens}, pop_size={pop_size}")
    print("=" * 70)

    # 1. Prepare Datasets
    datasets = []
    
    print("\n[Dataset 1/3] Generating LFR Disjoint Graph...")
    try:
        G_lfr_d, cmty_lfr_d = load_lfr_disjoint()
        datasets.append(("LFR_disjoint", G_lfr_d, cmty_lfr_d))
    except Exception as e:
        print(f"Failed to load LFR Disjoint: {e}")

    print("\n[Dataset 2/3] Generating LFR Overlapping Graph...")
    try:
        G_lfr_o, cmty_lfr_o = load_lfr_overlapping()
        datasets.append(("LFR_overlapping", G_lfr_o, cmty_lfr_o))
    except Exception as e:
        print(f"Failed to load LFR Overlapping: {e}")

    print("\n[Dataset 3/3] Loading DBLP Benchmark Graph...")
    try:
        G_dblp, cmty_dblp = load_dblp()
        datasets.append(("DBLP", G_dblp, cmty_dblp))
    except Exception as e:
        print(f"Failed to load DBLP: {e}")

    strategies = [
        ("crisp", 0.0),
        ("random_overlap", 0.2),
        ("boundary_seeded", 0.2),
    ]

    summary_rows = []
    convergence_rows = []

    summary_csv_path = DATA_DIR / "init_strategy_summary.csv"
    convergence_csv_path = DATA_DIR / "init_strategy_convergence.csv"

    for dataset_name, G, gt_communities in datasets:
        print(f"\n" + "-" * 60)
        print(f"BENCHMARKING DATASET: {dataset_name} ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
        print("-" * 60)

        for strat_name, overlap_prob in strategies:
            print(f"  -> Strategy: {strat_name} (overlap_prob={overlap_prob}) across {num_seeds} seeds...")
            
            for seed_idx in range(1, num_seeds + 1):
                seed = 1000 + seed_idx

                t0 = time.perf_counter()
                
                alg = pymocd.OhpMocd(
                    G,
                    debug_level=0,
                    pop_size=pop_size,
                    num_gens=num_gens,
                    max_memberships_per_node=2,
                    init_strategy=strat_name,
                    init_overlap_prob=overlap_prob,
                    seed=seed,
                )

                front = alg.generate_pareto_front()
                t1 = time.perf_counter()
                runtime_ms = (t1 - t0) * 1000.0

                # Select max-Q partition from Pareto front
                best_part = None
                best_objs = None
                best_q = -float("inf")
                for part, objs in front:
                    intra, inter = objs[0], objs[1]
                    q = 1.0 - intra - inter
                    if q > best_q:
                        best_q = q
                        best_part = part
                        best_objs = objs

                if best_part is None:
                    continue

                nmi, ami, ari = evaluate_ground_truth_metrics(G, gt_communities, best_part)
                num_over, avg_mem, max_mem = compute_overlap_stats(best_part)

                summary_rows.append({
                    "dataset": dataset_name,
                    "strategy": strat_name,
                    "overlap_prob": overlap_prob,
                    "seed": seed,
                    "runtime_ms": f"{runtime_ms:.2f}",
                    "intra": f"{best_objs[0]:.6f}",
                    "inter": f"{best_objs[1]:.6f}",
                    "max_Q": f"{best_q:.6f}",
                    "nmi": f"{nmi:.4f}",
                    "ami": f"{ami:.4f}",
                    "ari": f"{ari:.4f}",
                    "num_overlapping_nodes": num_over,
                    "avg_memberships": f"{avg_mem:.4f}",
                    "max_memberships": max_mem,
                })

                history = alg.get_convergence_history()
                for gen_idx, q_val in enumerate(history):
                    convergence_rows.append({
                        "dataset": dataset_name,
                        "strategy": strat_name,
                        "seed": seed,
                        "generation": gen_idx,
                        "best_Q": f"{q_val:.6f}",
                    })

    # Export CSV Files
    print("\n" + "=" * 70)
    print("SAVING EXPERIMENT RESULTS TO CSV")
    print("=" * 70)

    summary_headers = [
        "dataset", "strategy", "overlap_prob", "seed", "runtime_ms",
        "intra", "inter", "max_Q", "nmi", "ami", "ari",
        "num_overlapping_nodes", "avg_memberships", "max_memberships"
    ]
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_headers)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"  [Summary CSV] Saved to: {summary_csv_path}")

    conv_headers = ["dataset", "strategy", "seed", "generation", "best_Q"]
    with open(convergence_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=conv_headers)
        writer.writeheader()
        writer.writerows(convergence_rows)
    print(f"  [Convergence CSV] Saved to: {convergence_csv_path}")

    print("\nEXPERIMENTS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    run_experiment_suite(num_seeds=20, num_gens=30, pop_size=30)
