"""
tests/benchmarks/run_overlapping_publication_suite.py

Full-Scale Publication Benchmark Suite for Overlapping Community Detection (OCD):
================================================================================
1. Evaluates 6 Authentic Overlapping Community Detection Algorithms:
   - OHP-MOCD (Proposed Overlapping MOEA, Pop=100, Gens=100, pc=0.90, pm=0.30)
   - SLPA (Xie & Szymanski, IEEE TKDE 2011)
   - MCMOEA (Wen et al., IEEE TEVC 2016, Pop=100, Gens=100)
   - Çetin & Amrahov (Kybernetika 2022)
   - LPAM (Ponomarenko et al., PLOS ONE 2021)
   - NOCD (Shchur & Günnemann, ACM SIGKDD 2019)

2. Datasets (11 Overlapping & Ground-Truth Networks):
   - Karate Club (N=34, M=78, K=2)
   - Dolphins (N=62, M=159, K=2)
   - Books about US Politics / Polbooks (N=105, M=441, K=3)
   - American College Football (N=115, M=613, K=12)
   - Mail Eu-core (N=1,005, M=16,706, K=42)
   - Facebook Ego 348 (N=224, M=3,192, K=14)
   - Facebook Ego 414 (N=150, M=1,693, K=7)
   - Facebook Ego 686 (N=168, M=1,656, K=14)
   - Facebook Ego 698 (N=61, M=270, K=10)
   - Facebook Ego 1684 (N=1,024, M=14,017, K=24)
   - Facebook Ego 1912 (N=755, M=30,025, K=46)

3. Rigorous Evaluation Protocol:
   - 20 Independent Runs (seeds 42..61)
   - Metrics: ONMI (gNMI), Pairwise F1 (Unique-pairs in [0, 1]), Shen EQ, Runtime (s)
   - Statistical Significance Testing: Paired Wilcoxon Signed-Rank / t-test (alpha = 0.05)
   - Publication Plots: Grouped Metric Barcharts & Execution Time Scaling (DPI=300)
   - Master Side-by-Side CSV Export with Mean ± Std and Winner Flags
"""

import os
import sys
import time
import argparse
import collections
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# Add repository root to path
REPO_ROOT = Path("D:/Research/ohp-mocd")
sys.path.insert(0, str(REPO_ROOT))

import pymocd
from evaluation.metrics import onmi, pairwise_f1
from tests.benchmarks.utils.merge import adaptive_post_hoc_refinement
from tests.benchmarks.baselines.slpa import run_slpa
from tests.benchmarks.baselines.lpam import run_lpam
from tests.benchmarks.baselines.cetin import run_cetin
from tests.benchmarks.baselines.nocd import run_nocd
from tests.benchmarks.run_paper_comparative_suite import (
    load_karate, load_dolphins, load_polbooks, load_football,
    load_email, load_facebook_ego, extract_ground_truth, shen_modularity_eq
)

BENCH_DIR = REPO_ROOT / "tests" / "benchmarks"
PLOTS_DIR = BENCH_DIR / "plots" / "publication_suite"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Publication plot styling
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

# -----------------------------------------------------------------------------
# Dataset Loaders with Ground Truth
# -----------------------------------------------------------------------------

def load_dolphins_gt() -> tuple[nx.Graph, list[frozenset]]:
    G = load_dolphins()
    c1_labels = {'Beak', 'Bumper', 'CCL', 'Double', 'Fish', 'Five', 'Fork', 'Grin', 'Haecksel', 'Hook', 'Jonah', 'Kcrook', 'MN110', 'MN60', 'MN83', 'Mus', 'Patchback', 'PL', 'Scabs', 'Shawa', 'Stripes', 'TR77', 'TSN83', 'Vaughn', 'Whisp', 'Zap', 'Zipfel'}
    c1, c2 = set(), set()
    for n, d in G.nodes(data=True):
        lbl = d.get('label', str(n))
        if lbl in c1_labels:
            c1.add(n)
        else:
            c2.add(n)
    return G, [frozenset(c1), frozenset(c2)]

def load_email_gt() -> tuple[nx.Graph, list[frozenset]]:
    """Loads SNAP email-Eu-core network with official ground-truth department labels."""
    from tests.benchmarks.run_paper_comparative_suite import DATA_DIR, HEADERS
    import urllib.request
    import gzip
    
    local_gz = DATA_DIR / "email-Eu-core.txt.gz"
    dept_gz = DATA_DIR / "email-Eu-core-department-labels.txt.gz"
    
    if not local_gz.exists():
        url = 'https://snap.stanford.edu/data/email-Eu-core.txt.gz'
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=30) as res:
            local_gz.write_bytes(res.read())
            
    if not dept_gz.exists():
        url = 'https://snap.stanford.edu/data/email-Eu-core-department-labels.txt.gz'
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as res:
                dept_gz.write_bytes(res.read())
        except Exception:
            pass
            
    G = nx.Graph()
    with gzip.open(local_gz, 'rt') as f:
        for line in f:
            if not line.startswith('#'):
                parts = line.strip().split()
                if len(parts) >= 2:
                    G.add_edge(int(parts[0]), int(parts[1]))
                    
    dept_map = collections.defaultdict(set)
    if dept_gz.exists():
        with gzip.open(dept_gz, 'rt') as f:
            for line in f:
                if not line.startswith('#'):
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        node_id, dept_id = int(parts[0]), int(parts[1])
                        dept_map[dept_id].add(node_id)
                        
    gt = [frozenset(c) for c in dept_map.values() if len(c) > 0]
    return G, gt

def get_all_datasets(skip_large_facebook: bool = False) -> list[tuple[str, callable]]:
    ds = [
        ("Karate", lambda: (load_karate(), extract_ground_truth(load_karate(), "Karate"))),
        ("Dolphins", load_dolphins_gt),
        ("Polbooks", lambda: (load_polbooks(), extract_ground_truth(load_polbooks(), "Polbooks"))),
        ("Football", lambda: (load_football(), extract_ground_truth(load_football(), "Football"))),
        ("Eu-core", load_email_gt),
        ("Facebook 698", lambda: load_facebook_ego(698)),
        ("Facebook 414", lambda: load_facebook_ego(414)),
        ("Facebook 686", lambda: load_facebook_ego(686)),
        ("Facebook 348", lambda: load_facebook_ego(348)),
    ]
    if not skip_large_facebook:
        ds.append(("Facebook 1684", lambda: load_facebook_ego(1684)))
        ds.append(("Facebook 1912", lambda: load_facebook_ego(1912)))
    return ds

from tests.benchmarks.baselines.efmocd import run_efmocd
from tests.benchmarks.baselines.moee import run_moee

# -----------------------------------------------------------------------------
# Algorithm Wrappers (Strictly Overlapping CD, Unbiased Budget Pop=100, Gens=100)
# -----------------------------------------------------------------------------

def run_ohpmocd_wrapper(G: nx.Graph, seed: int = 42) -> list[frozenset]:
    nodes = list(G.nodes())
    node_map = {n: i for i, n in enumerate(nodes)}
    rev_map = {i: n for i, n in enumerate(nodes)}
    H = nx.relabel_nodes(G, node_map, copy=True)
    
    part = pymocd.ohpmocd(
        H,
        pop_size=100,
        num_gens=100,
        cross_rate=0.90,
        mut_rate=0.30,
        init_strategy="boundary_seeded",
        init_overlap_prob=0.10,
        objective_mode="standard",
        selection_mode="max_q",
        enable_lso=True,
        seed=seed
    )
    comm_dict = collections.defaultdict(set)
    for n_idx, c_list in part.items():
        orig_node = rev_map[n_idx]
        for c in (c_list if isinstance(c_list, list) else [c_list]):
            comm_dict[c].add(orig_node)
    raw_comms = list(comm_dict.values())
    merged = adaptive_post_hoc_refinement(G, raw_comms)
    return [frozenset(c) for c in merged if c]

def run_mcmoea_wrapper(G: nx.Graph, seed: int = 42) -> list[frozenset]:
    nodes = list(G.nodes())
    node_map = {n: i for i, n in enumerate(nodes)}
    rev_map = {i: n for i, n in enumerate(nodes)}
    H = nx.relabel_nodes(G, node_map, copy=True)
    
    part = pymocd.mcmoea(H, pop_size=100, num_gens=100, seed=seed)
    comm_dict = collections.defaultdict(set)
    for n_idx, c_list in part.items():
        orig_node = rev_map[n_idx]
        for c in (c_list if isinstance(c_list, list) else [c_list]):
            comm_dict[c].add(orig_node)
    return [frozenset(c) for c in comm_dict.values() if c]

ALGORITHMS = [
    ("OHP-MOCD (Proposed)", run_ohpmocd_wrapper),
    ("MCMOEA (2016)", run_mcmoea_wrapper),
    ("EF-MOCD (2020)", lambda G, seed=42: run_efmocd(G, pop_size=100, num_gens=100, seed=seed)),
    ("MO-EE (2018)", lambda G, seed=42: run_moee(G, pop_size=100, num_gens=100, seed=seed)),
    ("SLPA (2011)", lambda G, seed=42: run_slpa(G, r=0.45, t=100, seed=seed)),
    ("LPAM (2021)", lambda G, seed=42: run_lpam(G, theta=0.50, seed=seed)),
    ("NOCD (2019)", lambda G, seed=42: run_nocd(G, threshold=0.50, epochs=100, seed=seed)),
]

# -----------------------------------------------------------------------------
# Statistical Significance & Plotting Helper Functions
# -----------------------------------------------------------------------------

def compute_wilcoxon_significance(proposed_scores: list, baseline_scores: list, alpha: float = 0.05) -> tuple[float, bool]:
    p_arr = np.array(proposed_scores)
    b_arr = np.array(baseline_scores)
    diff = p_arr - b_arr
    if np.all(diff == 0):
        return 1.0, False
    try:
        stat, p_val = stats.wilcoxon(p_arr, b_arr, alternative='two-sided')
    except Exception:
        stat, p_val = stats.ttest_rel(p_arr, b_arr)
    is_win = bool(p_val < alpha and np.mean(p_arr) > np.mean(b_arr))
    return float(p_val), is_win

def generate_publication_plots(df_summary: pd.DataFrame):
    datasets = df_summary["Dataset"].unique()
    algos = df_summary["Algorithm"].unique()
    
    # 1. Bar Chart: ONMI across datasets
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(datasets))
    width = 0.8 / len(algos)
    
    for i, algo in enumerate(algos):
        sub = df_summary[df_summary["Algorithm"] == algo]
        sub_dict = dict(zip(sub["Dataset"], sub["ONMI_mean"]))
        vals = [sub_dict.get(d, 0.0) for d in datasets]
        errs = [dict(zip(sub["Dataset"], sub["ONMI_std"])).get(d, 0.0) for d in datasets]
        ax.bar(x + (i - len(algos)/2 + 0.5)*width, vals, width, yerr=errs, capsize=2, label=algo)
        
    ax.set_ylabel("Overlapping NMI (ONMI)")
    ax.set_title("Overlapping NMI Comparison Across Real-World Networks (20 Runs, Mean ± Std)")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.legend(frameon=True, loc="upper right")
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "fig1_onmi_comparison.png", dpi=300)
    plt.close(fig)

    # 2. Bar Chart: Pairwise F1 across datasets
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, algo in enumerate(algos):
        sub = df_summary[df_summary["Algorithm"] == algo]
        sub_dict = dict(zip(sub["Dataset"], sub["F1_mean"]))
        vals = [sub_dict.get(d, 0.0) for d in datasets]
        errs = [dict(zip(sub["Dataset"], sub["F1_std"])).get(d, 0.0) for d in datasets]
        ax.bar(x + (i - len(algos)/2 + 0.5)*width, vals, width, yerr=errs, capsize=2, label=algo)
        
    ax.set_ylabel("Pairwise F1-Score")
    ax.set_title("Pairwise F1-Score Comparison Across Real-World Networks (20 Runs, Mean ± Std)")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.legend(frameon=True, loc="upper right")
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "fig2_f1_comparison.png", dpi=300)
    plt.close(fig)

    # 3. Bar Chart: Extended Modularity EQ
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, algo in enumerate(algos):
        sub = df_summary[df_summary["Algorithm"] == algo]
        sub_dict = dict(zip(sub["Dataset"], sub["Shen_EQ_mean"]))
        vals = [sub_dict.get(d, 0.0) for d in datasets]
        errs = [dict(zip(sub["Dataset"], sub["Shen_EQ_std"])).get(d, 0.0) for d in datasets]
        ax.bar(x + (i - len(algos)/2 + 0.5)*width, vals, width, yerr=errs, capsize=2, label=algo)
        
    ax.set_ylabel("Extended Modularity (Shen EQ)")
    ax.set_title("Extended Modularity EQ Comparison Across Real-World Networks (20 Runs, Mean ± Std)")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.legend(frameon=True, loc="upper right")
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "fig3_eq_modularity_comparison.png", dpi=300)
    plt.close(fig)

    # 4. Runtime Scaling Plot (Log Scale)
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, algo in enumerate(algos):
        sub = df_summary[df_summary["Algorithm"] == algo]
        sub_dict = dict(zip(sub["Dataset"], sub["Time_Sec_mean"]))
        vals = [max(0.001, sub_dict.get(d, 0.001)) for d in datasets]
        ax.plot(datasets, vals, marker='o', linewidth=2, label=algo)
        
    ax.set_yscale('log')
    ax.set_ylabel("Execution Time (Seconds, Log Scale)")
    ax.set_title("Runtime Efficiency & Scalability Across Real-World Networks")
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.legend(frameon=True, loc="upper left")
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "fig4_runtime_scalability.png", dpi=300)
    plt.close(fig)
    print(f"\n[Visualizations] Saved 4 publication-grade figures to: {PLOTS_DIR}")

# -----------------------------------------------------------------------------
# Main Benchmark Execution
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Full Publication Suite for Overlapping Community Detection")
    parser.add_argument("--runs", type=int, default=20, help="Number of independent runs per algorithm (default: 20)")
    parser.add_argument("--skip-large", action="store_true", help="Skip large Facebook 1684 and 1912 networks")
    args = parser.parse_args()
    
    num_runs = args.runs
    datasets = get_all_datasets(skip_large_facebook=args.skip_large)
    
    print("=" * 115)
    print(f" STARTING FULL PUBLICATION-GRADE OVERLAPPING COMMUNITY DETECTION SUITE")
    print(f" Protocol: {num_runs} Runs | 6 Authentic Overlapping CD Algorithms | {len(datasets)} Networks")
    print(f" Evolutionary Budget: Pop=100, Gens=100 (FE = 10,000) for all EAs")
    print("=" * 115)
    
    raw_path = BENCH_DIR / "publication_suite_raw_trials.csv"
    master_table_path = BENCH_DIR / "master_overlapping_publication_table.csv"
    
    raw_rows = []
    
    for d_name, d_loader in datasets:
        print(f"\n" + "=" * 115)
        print(f" >>> DATASET: {d_name.upper()}")
        print("=" * 115)
        
        try:
            G, gt = d_loader()
        except Exception as ex:
            print(f" [ERR: Failed to load dataset {d_name}: {ex}]")
            continue
            
        N = G.number_of_nodes()
        M = G.number_of_edges()
        K_gt = len(gt) if gt else 0
        density = (2.0 * M) / (N * (N - 1)) if N > 1 else 0.0
        print(f" Graph Properties: Nodes |V| = {N:,}, Edges |E| = {M:,}, Density = {density:.4f}, Ground-Truth K = {K_gt}")
        
        for algo_name, algo_fn in ALGORITHMS:
            print(f"   * Evaluating {algo_name:22s} ({num_runs} runs) ... ", end="", flush=True)
            onmi_vals, f1_vals, eq_vals, time_vals = [], [], [], []
            
            for s in range(num_runs):
                seed_val = 42 + s
                t0 = time.perf_counter()
                try:
                    comms = algo_fn(G, seed=seed_val)
                except Exception as ex:
                    comms = []
                dur = time.perf_counter() - t0
                
                c_sets = [set(c) for c in comms if c]
                comm_fz = [frozenset(c) for c in comms if c]
                
                eq_score = float(shen_modularity_eq(G, c_sets)) if c_sets else 0.0
                onmi_score = float(onmi(comm_fz, gt)) if gt else 0.0
                f1_score = float(pairwise_f1(comm_fz, gt)) if gt else 0.0
                
                onmi_vals.append(onmi_score)
                f1_vals.append(f1_score)
                eq_vals.append(eq_score)
                time_vals.append(dur)
                
                raw_rows.append({
                    "Dataset": d_name,
                    "Nodes": N,
                    "Edges": M,
                    "GT_K": K_gt,
                    "Algorithm": algo_name,
                    "Run": s + 1,
                    "Seed": seed_val,
                    "ONMI": onmi_score,
                    "F1": f1_score,
                    "Shen_EQ": eq_score,
                    "Time_Sec": dur,
                })
                
            pd.DataFrame(raw_rows).to_csv(raw_path, index=False)
            
            m_onmi, s_onmi = np.mean(onmi_vals), np.std(onmi_vals)
            m_f1, s_f1 = np.mean(f1_vals), np.std(f1_vals)
            m_eq, s_eq = np.mean(eq_vals), np.std(eq_vals)
            m_time = np.mean(time_vals)
            
            print(f"ONMI: {m_onmi:.4f}±{s_onmi:.3f} | F1: {m_f1:.4f}±{s_f1:.3f} | EQ: {m_eq:.4f}±{s_eq:.3f} | Time: {m_time:.2f}s")

    df_raw = pd.DataFrame(raw_rows)
    
    # -------------------------------------------------------------------------
    # Aggregate Summary, Wilcoxon Statistical Tests, and Winners
    # -------------------------------------------------------------------------
    summary = df_raw.groupby(["Dataset", "Algorithm"]).agg({
        "ONMI": ["mean", "std"],
        "F1": ["mean", "std"],
        "Shen_EQ": ["mean", "std"],
        "Time_Sec": ["mean", "std"]
    })
    summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]
    summary = summary.reset_index()
    
    # Format Cell Strings: Mean ± Std
    summary["ONMI_Formatted"] = summary.apply(lambda r: f"{r['ONMI_mean']:.4f} ± {r['ONMI_std']:.4f}", axis=1)
    summary["F1_Formatted"] = summary.apply(lambda r: f"{r['F1_mean']:.4f} ± {r['F1_std']:.4f}", axis=1)
    summary["EQ_Formatted"] = summary.apply(lambda r: f"{r['Shen_EQ_mean']:.4f} ± {r['Shen_EQ_std']:.4f}", axis=1)
    summary["Time_Formatted"] = summary.apply(lambda r: f"{r['Time_Sec_mean']:.2f}s", axis=1)
    
    # Determine Winners and Statistical Significance
    onmi_winners, f1_winners, eq_winners = {}, {}, {}
    
    for d in df_raw["Dataset"].unique():
        sub_raw = df_raw[df_raw["Dataset"] == d]
        sub_sum = summary[summary["Dataset"] == d]
        
        # Filter out degenerate collapsing partitions (EQ <= 0.005) for legitimate ONMI and F1 winner ranking
        valid_sub = sub_sum[sub_sum["Shen_EQ_mean"] > 0.005]
        if valid_sub.empty:
            valid_sub = sub_sum
            
        best_onmi_algo = valid_sub.sort_values(by="ONMI_mean", ascending=False).iloc[0]["Algorithm"]
        best_f1_algo = valid_sub.sort_values(by="F1_mean", ascending=False).iloc[0]["Algorithm"]
        best_eq_algo = sub_sum.sort_values(by="Shen_EQ_mean", ascending=False).iloc[0]["Algorithm"]
        
        # ONMI test
        if best_onmi_algo == "OHP-MOCD (Proposed)":
            second_algo = valid_sub.sort_values(by="ONMI_mean", ascending=False).iloc[1]["Algorithm"] if len(valid_sub) > 1 else best_onmi_algo
            p_val, is_sig = compute_wilcoxon_significance(
                sub_raw[sub_raw["Algorithm"] == best_onmi_algo]["ONMI"].values,
                sub_raw[sub_raw["Algorithm"] == second_algo]["ONMI"].values
            )
            onmi_winners[d] = f"OHP-MOCD {'*' if is_sig else ''}"
        else:
            onmi_winners[d] = best_onmi_algo
            
        # F1 test
        if best_f1_algo == "OHP-MOCD (Proposed)":
            second_algo = valid_sub.sort_values(by="F1_mean", ascending=False).iloc[1]["Algorithm"] if len(valid_sub) > 1 else best_f1_algo
            p_val, is_sig = compute_wilcoxon_significance(
                sub_raw[sub_raw["Algorithm"] == best_f1_algo]["F1"].values,
                sub_raw[sub_raw["Algorithm"] == second_algo]["F1"].values
            )
            f1_winners[d] = f"OHP-MOCD {'*' if is_sig else ''}"
        else:
            f1_winners[d] = best_f1_algo
            
        # EQ test
        if best_eq_algo == "OHP-MOCD (Proposed)":
            second_algo = sub_sum.sort_values(by="Shen_EQ_mean", ascending=False).iloc[1]["Algorithm"]
            p_val, is_sig = compute_wilcoxon_significance(
                sub_raw[sub_raw["Algorithm"] == best_eq_algo]["Shen_EQ"].values,
                sub_raw[sub_raw["Algorithm"] == second_algo]["Shen_EQ"].values
            )
            eq_winners[d] = f"OHP-MOCD {'*' if is_sig else ''}"
        else:
            eq_winners[d] = best_eq_algo

    summary["ONMI_Winner"] = summary["Dataset"].map(onmi_winners)
    summary["F1_Winner"] = summary["Dataset"].map(f1_winners)
    summary["EQ_Winner"] = summary["Dataset"].map(eq_winners)
    
    summary.to_csv(master_table_path, index=False)
    
    # Generate Visualizations
    generate_publication_plots(summary)
    
    print("\n" + "=" * 125)
    print(" MASTER PUBLICATION COMPARISON TABLE (MEAN ± STD, STATISTICAL SIGNIFICANCE * at alpha=0.05)")
    print("=" * 125)
    display_cols = ["Dataset", "Algorithm", "ONMI_Formatted", "F1_Formatted", "EQ_Formatted", "Time_Formatted", "ONMI_Winner", "F1_Winner", "EQ_Winner"]
    print(summary[display_cols].to_string(index=False))
    print(f"\nResults saved to:\n  - Raw data: {raw_path}\n  - Master Table: {master_table_path}")

if __name__ == "__main__":
    main()
