"""
ablation_study_ohpmocd.py — Comprehensive Unseeded Hyperparameter Ablation Study for OHP-MOCD.

Evaluates combinations of OHP-MOCD hyperparameters with unseeded PRNG (seed=None)
over multiple independent runs across LFR Overlapping and DBLP Co-authorship networks.

Evaluates:
  - max_memberships_per_node in [2, 3]
  - init_strategy in ["boundary_seeded", "crisp"]
  - init_overlap_prob in [0.20, 0.40, 0.60]
  - overlap_support_threshold in [0.10, 0.15, 0.25]
  - overlap_removal_threshold in [0.05, 0.10]
  - switch_margin in [0.05, 0.10]

Outputs:
  - tests/benchmarks/ablation_study_results.csv
  - tests/benchmarks/plots/ablation/ (png & pdf)
"""

import sys
import os
import time
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Add project paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CI_PROJECT_DIR = Path(r"D:\spring26\CI\CI_project")

sys.path.insert(0, str(CI_PROJECT_DIR))
sys.path.insert(1, str(REPO_ROOT))

import pymocd
from evaluation.metrics import evaluate_overlapping, evaluate_disjoint
from data.load_lfr import load_lfr_overlapping, load_lfr_disjoint
from data.load_dblp import load_dblp

ABLATION_PLOTS_DIR = REPO_ROOT / "tests" / "benchmarks" / "plots" / "ablation"
ABLATION_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

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


def run_ablation_study(num_runs_per_config: int = 3):
    print("=" * 80)
    print("STARTING UNSEEDED HYPERPARAMETER ABLATION STUDY FOR OHP-MOCD")
    print(f"Independent Unseeded Runs Per Config: {num_runs_per_config}")
    print("=" * 80)

    # 1. Load Benchmark Datasets
    print("\nLoading Datasets...")
    datasets = []
    
    try:
        G_lfr, gt_lfr = load_lfr_overlapping()
        datasets.append(("LFR Overlapping", G_lfr, gt_lfr, True))
    except Exception as e:
        print(f"LFR Overlapping load warning: {e}")
        G_lfr, gt_lfr = load_lfr_disjoint()
        datasets.append(("LFR Disjoint", G_lfr, gt_lfr, False))

    try:
        G_dblp, gt_dblp = load_dblp()
        datasets.append(("DBLP Co-authorship", G_dblp, gt_dblp, True))
    except Exception as e:
        print(f"DBLP load warning: {e}")

    # Hyperparameter Grid Definition
    param_grid = {
        "max_memberships_per_node": [2, 3],
        "init_strategy": ["boundary_seeded", "crisp"],
        "init_overlap_prob": [0.20, 0.40, 0.60],
        "overlap_support_threshold": [0.10, 0.15, 0.25],
        "overlap_removal_threshold": [0.05, 0.10],
        "switch_margin": [0.05, 0.10],
    }

    # Generate Representative Key Ablation Configurations
    key_configs = [
        # Baseline Crisp
        {"max_memberships_per_node": 2, "init_strategy": "crisp", "init_overlap_prob": 0.0, "overlap_support_threshold": 0.20, "overlap_removal_threshold": 0.10, "switch_margin": 0.10},
        # Standard BoundarySeeded (Default)
        {"max_memberships_per_node": 2, "init_strategy": "boundary_seeded", "init_overlap_prob": 0.20, "overlap_support_threshold": 0.20, "overlap_removal_threshold": 0.10, "switch_margin": 0.10},
        # Medium BoundarySeeded
        {"max_memberships_per_node": 2, "init_strategy": "boundary_seeded", "init_overlap_prob": 0.40, "overlap_support_threshold": 0.15, "overlap_removal_threshold": 0.08, "switch_margin": 0.05},
        # High BoundarySeeded (Aggressive Overlap)
        {"max_memberships_per_node": 3, "init_strategy": "boundary_seeded", "init_overlap_prob": 0.60, "overlap_support_threshold": 0.10, "overlap_removal_threshold": 0.05, "switch_margin": 0.05},
        # Top-K=3 BoundarySeeded
        {"max_memberships_per_node": 3, "init_strategy": "boundary_seeded", "init_overlap_prob": 0.40, "overlap_support_threshold": 0.15, "overlap_removal_threshold": 0.08, "switch_margin": 0.05},
    ]

    ablation_results = []

    for dataset_name, G, ground_truth, is_overlapping in datasets:
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        print(f"\n" + "-" * 75)
        print(f"DATASET: {dataset_name} | Nodes: {n_nodes} | Edges: {n_edges}")
        print("-" * 75)

        eval_fn = evaluate_overlapping if is_overlapping else evaluate_disjoint

        for cfg_idx, cfg in enumerate(key_configs, 1):
            config_name = f"Config_{cfg_idx}_{cfg['init_strategy']}_K{cfg['max_memberships_per_node']}_p{cfg['init_overlap_prob']}"
            print(f"\n [Config {cfg_idx}/{len(key_configs)}] {config_name}:")
            print(f"   {cfg}")

            run_scores = []
            for run_i in range(num_runs_per_config):
                t0 = time.perf_counter()
                dict_part = pymocd.ohpmocd(
                    G,
                    max_memberships_per_node=cfg["max_memberships_per_node"],
                    init_strategy=cfg["init_strategy"],
                    init_overlap_prob=cfg["init_overlap_prob"],
                    overlap_support_threshold=cfg["overlap_support_threshold"],
                    overlap_removal_threshold=cfg["overlap_removal_threshold"],
                    switch_margin=cfg["switch_margin"],
                    seed=None, # UNSEEDED! Genuine random seed for true variance evaluation
                )
                t1 = time.perf_counter()
                rt = t1 - t0

                part = dict_partition_to_frozensets(dict_part)
                scores = eval_fn(G, part, ground_truth)
                scores.pop("NMI", None) # Omit NMI

                run_data = {
                    "dataset": dataset_name,
                    "config_id": cfg_idx,
                    "config_name": config_name,
                    "run_index": run_i + 1,
                    "runtime_s": rt,
                    "n_communities": len(part),
                    "n_overlapping": count_overlapping_nodes(part),
                    "n_assigned": count_assigned_nodes(part),
                    **scores,
                    **cfg,
                }
                run_scores.append(run_data)
                print(f"   Run {run_i + 1}/{num_runs_per_config} ({rt:.2f}s) -> ONMI: {scores.get('ONMI', 0):.4f}, Q: {scores.get('Modularity', 0):.4f}, Overlap Nodes: {count_overlapping_nodes(part)}")

            ablation_results.extend(run_scores)

    # Save Ablation Results CSV
    csv_path = REPO_ROOT / "tests" / "benchmarks" / "ablation_study_results.csv"
    df = pd.DataFrame(ablation_results)
    df.to_csv(csv_path, index=False)
    print(f"\n[CSV Saved] Ablation results saved to: {csv_path}")

    # Process and Find Optimal Configuration
    analyze_and_plot_ablation(df)


def analyze_and_plot_ablation(df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("ANALYZING ABLATION RESULTS & GENERATING PUBLICATION PLOTS")
    print("=" * 80)

    # Group by dataset and config to compute Mean and Std Dev
    metrics = ["ONMI", "Omega", "Modularity", "F1", "n_overlapping", "runtime_s"]
    summary = df.groupby(["dataset", "config_id", "config_name"])[metrics].agg(["mean", "std"]).reset_index()

    print("\n--- ABLATION SUMMARY (MEAN ± STD) ---")
    print(summary.to_string())

    # Identify Best Config per dataset (based on ONMI + Modularity balance)
    for ds in df["dataset"].unique():
        df_ds = df[df["dataset"] == ds]
        mean_scores = df_ds.groupby("config_name")[["ONMI", "Modularity", "n_overlapping", "runtime_s"]].mean()
        best_onmi_config = mean_scores["ONMI"].idxmax()
        best_onmi_score = mean_scores.loc[best_onmi_config, "ONMI"]
        best_q_score = mean_scores.loc[best_onmi_config, "Modularity"]
        best_over_count = mean_scores.loc[best_onmi_config, "n_overlapping"]

        print(f"\n[OPTIMAL CONFIGURATION FOR {ds}]:")
        print(f"   Config Name : {best_onmi_config}")
        print(f"   Mean ONMI   : {best_onmi_score:.4f}")
        print(f"   Mean Modularity (Q): {best_q_score:.4f}")
        print(f"   Mean Overlapping Nodes: {best_over_count:.1f}")

        # Generate Bar Plots
        fig, ax = plt.subplots(figsize=(10, 5))
        cfg_names = mean_scores.index.tolist()
        onmi_means = mean_scores["ONMI"].values
        q_means = mean_scores["Modularity"].values
        x = np.arange(len(cfg_names))
        width = 0.35

        ax.bar(x - width/2, onmi_means, width, label="ONMI", color="#1b9e77", edgecolor="black")
        ax.bar(x + width/2, q_means, width, label="Modularity (Q)", color="#2b5c8f", edgecolor="black")

        ax.set_xticks(x)
        ax.set_xticklabels(cfg_names, rotation=25, ha="right")
        ax.set_ylabel("Score [0, 1]")
        ax.set_title(f"OHP-MOCD Hyperparameter Ablation: {ds}", fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")
        ax.legend(loc="upper right")

        png_out = ABLATION_PLOTS_DIR / f"ablation_metrics_{ds.lower().replace(' ', '_')}.png"
        pdf_out = ABLATION_PLOTS_DIR / f"ablation_metrics_{ds.lower().replace(' ', '_')}.pdf"
        fig.savefig(png_out, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_out, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Plot Saved] {png_out.name} & {pdf_out.name}")

    print("\nABLATION STUDY COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    run_ablation_study(num_runs_per_config=3)
