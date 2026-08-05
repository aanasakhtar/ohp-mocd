"""
compare_all_algorithms.py — Comprehensive Benchmark Suite for Overlapping Community Detection.

Evaluates algorithms across 5 Diverse Real-World & Synthetic Benchmark Datasets:
  1. LFR Overlapping Benchmark (1,000 nodes, 20% overlap)
  2. DBLP Co-authorship Network (10,000 nodes, 75.5% overlap)
  3. Amazon Co-purchasing Network (10,000 nodes, 97.1% overlap)
  4. Facebook Social Circles Network (4,039 nodes, 18.5% overlap, Full Network)
  5. YouTube User Interest Groups Network (10,000 nodes, 15.2% overlap)

Algorithms evaluated:
  1. OHP-MOCD (Rust Native - BoundarySeeded, Unseeded, Optimal Defaults)
  2. OHP-MOCD (Rust Native - Crisp, Unseeded)
  3. MCMOEA (Rust Native - Wen et al. 2016, Bounded Clique Depth)
  4. SLPA (Speaker-listener Label Propagation, Xie et al. 2011)
  5. CPM-Fixed (Clique Percolation Method, Palla et al. 2005)
  6. HP-MOCD Baseline (Disjoint MOEA, Santos et al. 2024)

Outputs:
  - CSV report: tests/benchmarks/all_algorithms_comparison.csv
  - Publication figures: tests/benchmarks/plots/ (PNG & PDF)
"""

import sys
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pymocd
from evaluation.metrics import evaluate_overlapping, evaluate_disjoint
from data.load_lfr import load_lfr_overlapping, load_lfr_disjoint
from data.load_dblp import load_dblp
from data.load_amazon import load_amazon
from data.load_facebook import load_facebook
from data.load_youtube import load_youtube

from algorithms.slpa import run_slpa
from algorithms.cpm import run_cpm_ncn_fixed

PLOTS_DIR = REPO_ROOT / "tests" / "benchmarks" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Publication plot styling (300 DPI)
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "font.family": "sans-serif",
})


def dict_partition_to_frozensets(partition_dict: dict) -> list[frozenset]:
    """Converts node -> community mapping (dict[node, int] or dict[node, list[int]]) to list of frozensets."""
    community_map = {}
    for node, comms in partition_dict.items():
        if isinstance(comms, (int, np.integer)):
            if comms >= 0:
                community_map.setdefault(int(comms), set()).add(node)
        elif isinstance(comms, (list, tuple, set)):
            for c in comms:
                if c >= 0:
                    community_map.setdefault(int(c), set()).add(node)
    return [frozenset(members) for c, members in sorted(community_map.items()) if members]


def count_overlapping_nodes(partition: list[frozenset]) -> int:
    from collections import Counter
    node_counts: Counter = Counter()
    for community in partition:
        for node in community:
            node_counts[node] += 1
    return sum(1 for c in node_counts.values() if c > 1)


def count_assigned_nodes(partition: list[frozenset]) -> int:
    return len(set().union(*partition)) if partition else 0


def run_benchmark_comparison():
    print("=" * 80)
    print("STARTING OVERLAPPING COMMUNITY DETECTION BENCHMARK SUITE (UNSEEDED)")
    print("=" * 80)

    # 1. Load Benchmark Datasets
    print("\nLoading Benchmark Datasets...")
    datasets = []
    
    # Dataset 1: LFR Overlapping
    try:
        G_lfr, gt_lfr = load_lfr_overlapping()
        datasets.append(("LFR Overlapping", G_lfr, gt_lfr, True))
    except Exception as e:
        print(f"LFR Overlapping load warning: {e}")

    # Dataset 2: DBLP Co-authorship
    try:
        G_dblp, gt_dblp = load_dblp()
        datasets.append(("DBLP Co-authorship", G_dblp, gt_dblp, True))
    except Exception as e:
        print(f"DBLP load warning: {e}")

    # Dataset 3: Amazon Co-purchasing
    try:
        G_amz, gt_amz = load_amazon()
        datasets.append(("Amazon Co-purchasing", G_amz, gt_amz, True))
    except Exception as e:
        print(f"Amazon load warning: {e}")

    # Dataset 4: Facebook Social Circles
    try:
        G_fb, gt_fb = load_facebook()
        datasets.append(("Facebook Social Circles", G_fb, gt_fb, True))
    except Exception as e:
        print(f"Facebook load warning: {e}")

    # Dataset 5: YouTube User Interest Groups
    try:
        G_yt, gt_yt = load_youtube()
        datasets.append(("YouTube User Groups", G_yt, gt_yt, True))
    except Exception as e:
        print(f"YouTube load warning: {e}")

    results = []

    for dataset_name, G, ground_truth, is_overlapping in datasets:
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        print(f"\n" + "-" * 75)
        print(f"DATASET: {dataset_name} | Nodes: {n_nodes} | Edges: {n_edges}")
        print("-" * 75)

        eval_fn = evaluate_overlapping if is_overlapping else evaluate_disjoint

        # 1. OHP-MOCD (Rust Native - BoundarySeeded, Optimal Unseeded Defaults)
        print(" -> Running OHP-MOCD (Rust - BoundarySeeded)...")
        t0 = time.perf_counter()
        dict_part_b = pymocd.ohpmocd(
            G,
            max_memberships_per_node=3,
            init_strategy="boundary_seeded",
            init_overlap_prob=0.40,
            overlap_support_threshold=0.15,
            overlap_removal_threshold=0.08,
            switch_margin=0.05,
            seed=None,
        )
        t1 = time.perf_counter()
        rt_rust_b = t1 - t0
        part_rust_b = dict_partition_to_frozensets(dict_part_b)
        scores_rust_b = eval_fn(G, part_rust_b, ground_truth)
        scores_rust_b.pop("NMI", None)
        results.append({
            "dataset": dataset_name,
            "algorithm": "OHP-MOCD (Rust - BoundarySeeded)",
            "category": "Overlapping (Rust)",
            "runtime_s": rt_rust_b,
            "n_communities": len(part_rust_b),
            "n_overlapping": count_overlapping_nodes(part_rust_b),
            "n_assigned": count_assigned_nodes(part_rust_b),
            **scores_rust_b,
        })

        # 2. OHP-MOCD (Rust Native - Crisp, Unseeded)
        print(" -> Running OHP-MOCD (Rust - Crisp)...")
        t0 = time.perf_counter()
        dict_part_c = pymocd.ohpmocd(
            G,
            max_memberships_per_node=3,
            init_strategy="crisp",
            overlap_support_threshold=0.15,
            overlap_removal_threshold=0.08,
            switch_margin=0.05,
            seed=None,
        )
        t1 = time.perf_counter()
        rt_rust_c = t1 - t0
        part_rust_c = dict_partition_to_frozensets(dict_part_c)
        scores_rust_c = eval_fn(G, part_rust_c, ground_truth)
        scores_rust_c.pop("NMI", None)
        results.append({
            "dataset": dataset_name,
            "algorithm": "OHP-MOCD (Rust - Crisp)",
            "category": "Overlapping (Rust)",
            "runtime_s": rt_rust_c,
            "n_communities": len(part_rust_c),
            "n_overlapping": count_overlapping_nodes(part_rust_c),
            "n_assigned": count_assigned_nodes(part_rust_c),
            **scores_rust_c,
        })

        # 3. MCMOEA (Rust Native - Wen et al. 2016, Bounded Clique Depth)
        print(" -> Running MCMOEA (Rust Native)...")
        t0 = time.perf_counter()
        dict_mcmoea = pymocd.mcmoea(G, seed=None)
        t1 = time.perf_counter()
        rt_mcmoea = t1 - t0
        part_mcmoea = dict_partition_to_frozensets(dict_mcmoea)
        scores_mcmoea = eval_fn(G, part_mcmoea, ground_truth)
        scores_mcmoea.pop("NMI", None)
        results.append({
            "dataset": dataset_name,
            "algorithm": "MCMOEA (Rust Native)",
            "category": "Overlapping Baseline (Rust)",
            "runtime_s": rt_mcmoea,
            "n_communities": len(part_mcmoea),
            "n_overlapping": count_overlapping_nodes(part_mcmoea),
            "n_assigned": count_assigned_nodes(part_mcmoea),
            **scores_mcmoea,
        })

        # 4. SLPA (Speaker-listener Label Propagation)
        print(" -> Running SLPA...")
        part_slpa, rt_slpa = run_slpa(G, T=20, r=0.10, seed=None)
        scores_slpa = eval_fn(G, part_slpa, ground_truth)
        scores_slpa.pop("NMI", None)
        results.append({
            "dataset": dataset_name,
            "algorithm": "SLPA",
            "category": "Overlapping Baseline",
            "runtime_s": rt_slpa,
            "n_communities": len(part_slpa),
            "n_overlapping": count_overlapping_nodes(part_slpa),
            "n_assigned": count_assigned_nodes(part_slpa),
            **scores_slpa,
        })

        # 5. CPM-Fixed (Clique Percolation Method)
        print(" -> Running CPM-Fixed...")
        part_cpm, rt_cpm, _k = run_cpm_ncn_fixed(G, k_values=[3, 4, 5])
        scores_cpm = eval_fn(G, part_cpm, ground_truth)
        scores_cpm.pop("NMI", None)
        results.append({
            "dataset": dataset_name,
            "algorithm": "CPM-Fixed(k=3)",
            "category": "Overlapping Baseline",
            "runtime_s": rt_cpm,
            "n_communities": len(part_cpm),
            "n_overlapping": count_overlapping_nodes(part_cpm),
            "n_assigned": count_assigned_nodes(part_cpm),
            **scores_cpm,
        })

        # 6. HP-MOCD Baseline (Disjoint MOEA)
        print(" -> Running HP-MOCD Baseline (Disjoint)...")
        t0 = time.perf_counter()
        dict_hp = pymocd.hpmocd(G)
        t1 = time.perf_counter()
        rt_hp = t1 - t0
        part_hp = dict_partition_to_frozensets(dict_hp)
        scores_hp = eval_fn(G, part_hp, ground_truth)
        scores_hp.pop("NMI", None)
        results.append({
            "dataset": dataset_name,
            "algorithm": "HP-MOCD Baseline",
            "category": "Disjoint Baseline",
            "runtime_s": rt_hp,
            "n_communities": len(part_hp),
            "n_overlapping": count_overlapping_nodes(part_hp),
            "n_assigned": count_assigned_nodes(part_hp),
            **scores_hp,
        })

    # Export Results CSV
    csv_path = REPO_ROOT / "tests" / "benchmarks" / "all_algorithms_comparison.csv"
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"\n[CSV Saved] Benchmark results saved to: {csv_path}")

    # Generate Publication Plots
    plot_benchmark_results(df)


def plot_benchmark_results(df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("GENERATING COMPARISON PLOTS (PNG & PDF)")
    print("=" * 80)

    color_map = {
        "OHP-MOCD (Rust - BoundarySeeded)": "#1b9e77",
        "OHP-MOCD (Rust - Crisp)": "#d95f02",
        "MCMOEA (Rust Native)": "#7570b3",
        "SLPA": "#e7298a",
        "CPM-Fixed(k=3)": "#66a61e",
        "HP-MOCD Baseline": "#e6ab02",
    }

    for dataset_name in df["dataset"].unique():
        df_ds = df[df["dataset"] == dataset_name].copy()
        ds_slug = dataset_name.lower().replace(" ", "_")

        # 1. Overlapping Metrics Bar Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        metrics = ["ONMI", "Omega", "Modularity", "F1"]
        x = np.arange(len(metrics))
        width = 0.13
        algs = df_ds["algorithm"].tolist()

        for idx, alg in enumerate(algs):
            row = df_ds[df_ds["algorithm"] == alg].iloc[0]
            vals = [row.get(m, 0.0) for m in metrics]
            ax.bar(
                x + (idx - len(algs) / 2) * width + width / 2,
                vals,
                width,
                label=alg,
                color=color_map.get(alg, "#333333"),
                edgecolor="black",
                linewidth=0.8,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(["ONMI", "Omega Index (\u03a9)", "Modularity (Q)", "Pairwise F1"])
        ax.set_ylabel("Score [0, 1]")
        ax.set_title(f"Overlapping Community Detection Quality: {dataset_name}", fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")
        ax.legend(loc="upper right", framealpha=0.9)

        fig.savefig(PLOTS_DIR / f"overlapping_metrics_{ds_slug}.png", dpi=300, bbox_inches="tight")
        fig.savefig(PLOTS_DIR / f"overlapping_metrics_{ds_slug}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"  [Plot Saved] overlapping_metrics_{ds_slug}.png & .pdf")

        # 2. Overlapping Node Detection Count Bar Plot
        fig, ax = plt.subplots(figsize=(9, 4.5))
        bars = ax.bar(
            df_ds["algorithm"],
            df_ds["n_overlapping"],
            color=[color_map.get(a, "#333333") for a in df_ds["algorithm"]],
            edgecolor="black",
            linewidth=0.8,
        )
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{int(height):,}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        ax.set_ylabel("Number of Overlapping Nodes Detected")
        ax.set_title(f"Overlapping Node Recovery Count: {dataset_name}", fontweight="bold")
        ax.set_xticks(np.arange(len(df_ds["algorithm"])))
        ax.set_xticklabels(df_ds["algorithm"], rotation=20, ha="right")
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")

        fig.savefig(PLOTS_DIR / f"overlapping_nodes_{ds_slug}.png", dpi=300, bbox_inches="tight")
        fig.savefig(PLOTS_DIR / f"overlapping_nodes_{ds_slug}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"  [Plot Saved] overlapping_nodes_{ds_slug}.png & .pdf")

        # 3. Execution Runtime Log-Scale Bar Plot
        fig, ax = plt.subplots(figsize=(9, 4.5))
        bars = ax.bar(
            df_ds["algorithm"],
            df_ds["runtime_s"],
            color=[color_map.get(a, "#333333") for a in df_ds["algorithm"]],
            edgecolor="black",
            linewidth=0.8,
        )
        ax.set_yscale("log")
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}s",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )
        ax.set_ylabel("Runtime (seconds, log-scale)")
        ax.set_title(f"Algorithm Execution Speed: {dataset_name}", fontweight="bold")
        ax.set_xticks(np.arange(len(df_ds["algorithm"])))
        ax.set_xticklabels(df_ds["algorithm"], rotation=20, ha="right")
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")

        fig.savefig(PLOTS_DIR / f"runtime_log_{ds_slug}.png", dpi=300, bbox_inches="tight")
        fig.savefig(PLOTS_DIR / f"runtime_log_{ds_slug}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"  [Plot Saved] runtime_log_{ds_slug}.png & .pdf")

    print("\nBENCHMARK SUITE COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    run_benchmark_comparison()
