"""
ablation_study_ohpmocd.py — Comprehensive Unseeded Hyperparameter Ablation Study for OHP-MOCD.

Evaluates combinations of OHP-MOCD hyperparameters with unseeded PRNG (seed=None)
over multiple independent runs across real-world benchmark networks (Karate, Dolphins, Polbooks, Football, Lesmis).

Evaluates:
  - init_strategy in ["boundary_seeded", "random_overlap", "crisp"]
  - init_overlap_prob in [0.20, 0.40, 0.60]
  - overlap_support_threshold in [0.10, 0.15, 0.25]
  - overlap_removal_threshold in [0.05, 0.08, 0.12]
  - switch_margin in [0.02, 0.05, 0.10]

Outputs:
  - tests/benchmarks/ablation_study_results.csv
  - tests/benchmarks/plots/ablation/ (png & pdf)
"""

import sys
import os
import time
import zipfile
import io
import urllib.request
import re
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import concurrent.futures

# Add project paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pymocd

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

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}


def dict_partition_to_frozensets(partition_dict: dict) -> list[frozenset]:
    """Converts node -> community mapping (dict[node, list[int]]) to list of frozensets."""
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


def nicosia_qov(G: nx.Graph, communities: list[frozenset]) -> float:
    """Nicosia et al. (2009) Overlapping Modularity Qov."""
    m = G.number_of_edges()
    if m == 0:
        return 0.0
    two_m = 2.0 * m
    deg = dict(G.degree())
    
    node_belong = {}
    for comm in communities:
        for u in comm:
            node_belong[u] = node_belong.get(u, 0) + 1
            
    qov = 0.0
    for comm in communities:
        for u in comm:
            for v in comm:
                f_val = (1.0 / node_belong[u]) * (1.0 / node_belong[v])
                A_uv = 1.0 if G.has_edge(u, v) else 0.0
                k_u = deg.get(u, 0)
                k_v = deg.get(v, 0)
                qov += (A_uv - (k_u * k_v) / two_m) * f_val
                
    return qov / two_m


def load_karate() -> nx.Graph:
    return nx.karate_club_graph()


def load_lesmis() -> nx.Graph:
    return nx.les_miserables_graph()


def load_newman_gml(zip_name: str) -> nx.Graph:
    url = f'http://www-personal.umich.edu/~mejn/netdata/{zip_name}.zip'
    req = urllib.request.Request(url, headers=HEADERS)
    res = urllib.request.urlopen(req)
    z = zipfile.ZipFile(io.BytesIO(res.read()))
    gml_name = [f for f in z.namelist() if f.endswith('.gml')][0]
    content = z.read(gml_name).decode('utf-8', errors='ignore')
    return nx.parse_gml(content, label='id' if 'id' in content else 'label')


def load_dolphins() -> nx.Graph:
    return load_newman_gml('dolphins')


def load_polbooks() -> nx.Graph:
    return load_newman_gml('polbooks')


def load_football() -> nx.Graph:
    return load_newman_gml('football')


def load_netscience() -> nx.Graph:
    G = load_newman_gml('netscience')
    largest_cc = max(nx.connected_components(G), key=len)
    return G.subgraph(largest_cc).copy()


def evaluate_single_ablation_run(task_tuple: tuple) -> dict:
    """Picklable worker function for parallel ablation evaluation."""
    net_name, config_id, config_name, cfg_params, run_index, edge_list = task_tuple
    G = nx.Graph(edge_list)
    nodes = list(G.nodes())
    node_map = {n: i for i, n in enumerate(nodes)}
    rev_map = {i: n for i, n in enumerate(nodes)}
    H = nx.relabel_nodes(G, node_map, copy=True)

    t0 = time.perf_counter()
    dict_res = pymocd.ohpmocd(
        H,
        init_strategy=cfg_params["init_strategy"],
        init_overlap_prob=cfg_params["init_overlap_prob"],
        overlap_support_threshold=cfg_params["overlap_support_threshold"],
        overlap_removal_threshold=cfg_params["overlap_removal_threshold"],
        switch_margin=cfg_params["switch_margin"],
        seed=None,
    )
    t1 = time.perf_counter()
    rt = t1 - t0

    comm_dict = {}
    for n_idx, comm_list in dict_res.items():
        orig_node = rev_map[n_idx]
        if isinstance(comm_list, (int, np.integer)):
            comm_list = [comm_list]
        for cid in comm_list:
            comm_dict.setdefault(cid, set()).add(orig_node)
    comms = [frozenset(members) for members in comm_dict.values()]

    qov = nicosia_qov(G, comms)
    return {
        "dataset": net_name,
        "config_id": config_id,
        "config_name": config_name,
        "run_index": run_index,
        "runtime_s": rt,
        "n_communities": len(comms),
        "n_overlapping": count_overlapping_nodes(comms),
        "n_assigned": count_assigned_nodes(comms),
        "Qov": qov,
        **cfg_params,
    }


def run_ablation_study(num_runs_per_config: int = 5):
    print("=" * 80)
    print("STARTING UNSEEDED HYPERPARAMETER ABLATION STUDY FOR OHP-MOCD")
    print(f"Independent Unseeded Runs Per Config: {num_runs_per_config}")
    print("=" * 80)

    # 1. Load Benchmark Datasets
    print("\nLoading Real-World Benchmark Datasets...")
    dataset_loaders = [
        ("Karate", load_karate),
        ("Dolphins", load_dolphins),
        ("Lesmis", load_lesmis),
        ("Polbooks", load_polbooks),
        ("Football", load_football),
        ("Netscience", load_netscience),
    ]

    datasets = []
    for name, loader in dataset_loaders:
        try:
            g = loader()
            datasets.append((name, g))
            print(f"  Loaded {name} (N={g.number_of_nodes()}, E={g.number_of_edges()})")
        except Exception as e:
            print(f"  Warning loading {name}: {e}")

    # Generate Representative Key Ablation Configurations
    key_configs = [
        # Config 1: Baseline Crisp Initialization
        {"name": "Config_1_Crisp", "init_strategy": "crisp", "init_overlap_prob": 0.0, "overlap_support_threshold": 0.20, "overlap_removal_threshold": 0.10, "switch_margin": 0.05},
        # Config 2: Conservative BoundarySeeded
        {"name": "Config_2_Boundary_Conservative", "init_strategy": "boundary_seeded", "init_overlap_prob": 0.20, "overlap_support_threshold": 0.25, "overlap_removal_threshold": 0.12, "switch_margin": 0.10},
        # Config 3: Standard BoundarySeeded (Default Model)
        {"name": "Config_3_Boundary_Default", "init_strategy": "boundary_seeded", "init_overlap_prob": 0.40, "overlap_support_threshold": 0.15, "overlap_removal_threshold": 0.08, "switch_margin": 0.05},
        # Config 4: Aggressive BoundarySeeded
        {"name": "Config_4_Boundary_Aggressive", "init_strategy": "boundary_seeded", "init_overlap_prob": 0.60, "overlap_support_threshold": 0.10, "overlap_removal_threshold": 0.05, "switch_margin": 0.02},
        # Config 5: RandomOverlap Strategy
        {"name": "Config_5_RandomOverlap", "init_strategy": "random_overlap", "init_overlap_prob": 0.40, "overlap_support_threshold": 0.15, "overlap_removal_threshold": 0.08, "switch_margin": 0.05},
    ]

    tasks = []
    for dataset_name, G in datasets:
        edge_list = list(G.edges())
        for cfg_idx, cfg in enumerate(key_configs, 1):
            config_name = cfg["name"]
            cfg_params = {k: v for k, v in cfg.items() if k != "name"}
            for run_i in range(num_runs_per_config):
                tasks.append((dataset_name, cfg_idx, config_name, cfg_params, run_i + 1, edge_list))

    print(f"\nExecuting {len(tasks)} evaluations concurrently...", flush=True)
    max_workers = max(1, (os.cpu_count() or 4) - 1)
    print(f"Parallel Worker Pool: {max_workers} CPU cores", flush=True)

    ablation_results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(evaluate_single_ablation_run, t): t for t in tasks}
        completed = 0
        for future in concurrent.futures.as_completed(future_to_task):
            res = future.result()
            ablation_results.append(res)
            completed += 1
            if completed % 10 == 0 or completed == len(tasks):
                print(f"  Progress: [{completed}/{len(tasks)}] runs finished | Latest: {res['dataset']} - {res['config_name']} (Run {res['run_index']}) -> Qov: {res['Qov']:.4f}", flush=True)

    # Save Ablation Results CSV
    csv_path = REPO_ROOT / "tests" / "benchmarks" / "ablation_study_results.csv"
    df = pd.DataFrame(ablation_results)
    df.to_csv(csv_path, index=False)
    print(f"\n[CSV Saved] Ablation results saved to: {csv_path}", flush=True)

    # Process and Find Optimal Configuration
    analyze_and_plot_ablation(df)


def analyze_and_plot_ablation(df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("ANALYZING ABLATION RESULTS & GENERATING PUBLICATION PLOTS")
    print("=" * 80)

    # Group by dataset and config to compute Mean and Std Dev
    metrics = ["Qov", "n_communities", "n_overlapping", "runtime_s"]
    summary = df.groupby(["dataset", "config_id", "config_name"])[metrics].agg(["mean", "std"]).reset_index()

    print("\n--- ABLATION SUMMARY (MEAN ± STD) ---")
    print(summary.to_string())

    # Identify Best Config per dataset based on Nicosia Qov
    best_summary = []
    for ds in df["dataset"].unique():
        df_ds = df[df["dataset"] == ds]
        mean_scores = df_ds.groupby("config_name")[["Qov", "n_overlapping", "runtime_s"]].mean()
        best_qov_config = mean_scores["Qov"].idxmax()
        best_qov_score = mean_scores.loc[best_qov_config, "Qov"]
        best_over_count = mean_scores.loc[best_qov_config, "n_overlapping"]

        print(f"\n[OPTIMAL CONFIGURATION FOR {ds}]:")
        print(f"   Config Name            : {best_qov_config}")
        print(f"   Mean Nicosia Qov       : {best_qov_score:.4f}")
        print(f"   Mean Overlapping Nodes : {best_over_count:.1f}")

        best_summary.append({
            "Dataset": ds,
            "Optimal Config": best_qov_config,
            "Mean Qov": f"{best_qov_score:.4f}",
            "Mean Overlapping Nodes": f"{best_over_count:.1f}"
        })

        # Generate Bar Plots
        fig, ax = plt.subplots(figsize=(9, 5))
        cfg_names = mean_scores.index.tolist()
        qov_means = mean_scores["Qov"].values
        x = np.arange(len(cfg_names))
        width = 0.45

        bars = ax.bar(x, qov_means, width, label="Nicosia Qov", color="#2b5c8f", edgecolor="black")
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.4f}", ha='center', va='bottom', fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(cfg_names, rotation=25, ha="right")
        ax.set_ylabel("Nicosia Qov Score")
        ax.set_title(f"OHP-MOCD Hyperparameter Ablation: {ds}", fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")
        ax.set_ylim(0, max(qov_means) * 1.18)

        png_out = ABLATION_PLOTS_DIR / f"ablation_qov_{ds.lower()}.png"
        pdf_out = ABLATION_PLOTS_DIR / f"ablation_qov_{ds.lower()}.pdf"
        fig.savefig(png_out, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_out, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Plot Saved] {png_out.name} & {pdf_out.name}")

    print("\n" + "=" * 80)
    print("SUMMARY OF OPTIMAL HYPERPARAMETER CONFIGURATIONS")
    print("=" * 80)
    best_df = pd.DataFrame(best_summary)
    print(best_df.to_string(index=False))


if __name__ == "__main__":
    run_ablation_study(num_runs_per_config=5)

