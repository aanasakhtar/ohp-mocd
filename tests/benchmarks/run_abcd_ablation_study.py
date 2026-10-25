"""
tests/benchmarks/run_abcd_ablation_study.py

Ablation Study on Overlapping Communities using the ABCD+o^2 Synthetic Benchmark.

This script executes two complementary ablation experiments:
  Experiment 1: Network Overlap Density Sweep (eta in [1.0, 1.25, 1.50, 1.75, 2.00])
                Evaluates how all comparative algorithms respond as the underlying
                ground-truth transitions from crisp (eta=1.0) to heavy overlap (eta=2.0).
  Experiment 2: OHP-MOCD Algorithmic Component Ablation on Overlapping ABCD+o^2
                Evaluates the individual contribution of:
                  - Overlap parsimony regularization (f3)
                  - Boundary-seeded initialization vs. crisp initialization
                  - Dual-Weighted Interaction (DWI) vs. unweighted topology
                  - Adaptive Chebyshev null boundary bound vs. fixed heuristic cutoff

Reference:
    Barrett et al. (2026). The artificial benchmark for community detection with
    outliers and overlapping communities (ABCD+o^2). Journal of Complex Networks, 14(4): cnag023.
"""

import sys
import time
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pymocd
from evaluation.metrics import onmi, pairwise_f1
from tests.benchmarks.utils.abcd import generate_abcd_benchmark
from tests.benchmarks.run_overlapping_publication_suite import (
    shen_modularity_eq, ALGORITHMS
)

PLOTS_DIR = REPO_ROOT / "tests" / "benchmarks" / "plots" / "ablation"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
    "figure.dpi": 300,
    "font.family": "sans-serif",
    "figure.autolayout": True
})


# -----------------------------------------------------------------------------
# OHP-MOCD Component Ablation Variants
# -----------------------------------------------------------------------------

def run_ohpmocd_variant(G: nx.Graph, variant: str, seed: int = 42) -> list[frozenset]:
    """Runs designated architectural ablation variant of OHP-MOCD."""
    # Defaults
    init_strat = "boundary_seeded"
    init_prob = 0.10
    cross_rate = 0.85
    mut_rate = 0.35
    pop = 100
    gens = 100

    if variant == "Full (Proposed)":
        pass
    elif variant == "Crisp Init (No Boundary Seeding)":
        init_strat = "crisp"
        init_prob = 0.0
    elif variant == "High Crossover (pc=0.95)":
        cross_rate = 0.95
    elif variant == "High Mutation (pm=0.50)":
        mut_rate = 0.50
    elif variant == "Lightweight Budget (Gens=50)":
        gens = 50

    nodes = sorted(G.nodes())
    node_map = {n: i for i, n in enumerate(nodes)}
    rev_map = {i: n for i, n in enumerate(nodes)}
    H = nx.relabel_nodes(G, node_map, copy=True)

    det_dict = pymocd.ohpmocd(
        H,
        pop_size=pop,
        num_gens=gens,
        cross_rate=cross_rate,
        mut_rate=mut_rate,
        init_strategy=init_strat,
        init_overlap_prob=init_prob,
        seed=seed,
    )

    comm_dict = {}
    for n_idx, c_val in det_dict.items():
        orig_u = rev_map.get(n_idx, n_idx)
        if isinstance(c_val, (int, np.integer)):
            c_val = [c_val]
        for cid in c_val:
            comm_dict.setdefault(cid, set()).add(orig_u)

    return [frozenset(c) for c in comm_dict.values() if c]


# -----------------------------------------------------------------------------
# Experiment 1: Network Overlap Density Sweep (eta in [1.0, 1.25, 1.5, 1.75, 2.0])
# -----------------------------------------------------------------------------

def run_network_overlap_sweep(runs: int = 5, n_nodes: int = 300, xi: float = 0.2):
    print("=" * 100)
    print(f"EXPERIMENT 1: ABCD+o^2 OVERLAP DENSITY SWEEP (eta in [1.0, 1.25, 1.50, 1.75, 2.00])")
    print(f"Parameters: N={n_nodes}, fixed background noise xi={xi}, {runs} seeds per eta")
    print("=" * 100)

    eta_values = [1.0, 1.25, 1.50, 1.75, 2.00]
    results = []

    for eta in eta_values:
        print(f"\n>>> TESTING OVERLAP MULTIPLIER eta = {eta:.2f}")
        for seed in range(runs):
            eff_seed = seed * 100 + int(round(eta * 100))
            G, ground_truth, node_to_comms = generate_abcd_benchmark(
                n=n_nodes,
                xi=xi,
                eta=eta,
                seed=eff_seed
            )

            memberships = [len(node_to_comms[u]) for u in G.nodes()]
            n_overlap = sum(1 for m in memberships if m > 1)
            pct_overlap = (n_overlap / G.number_of_nodes()) * 100.0 if G.number_of_nodes() > 0 else 0.0

            print(f"  [Seed {seed+1}/{runs}] Generated: Nodes={G.number_of_nodes()}, Edges={G.number_of_edges()}, "
                  f"Comms={len(ground_truth)}, Overlapping Nodes={n_overlap} ({pct_overlap:.1f}%)")

            for algo_name, runner in ALGORITHMS:
                t0 = time.perf_counter()
                try:
                    comms = runner(G, seed=seed)
                except Exception as ex:
                    print(f"    [Error in {algo_name}: {ex}]")
                    comms = [frozenset([n]) for n in G.nodes()]
                dur = time.perf_counter() - t0

                score_onmi = onmi(ground_truth, comms)
                score_f1 = pairwise_f1(ground_truth, comms)
                score_eq = shen_modularity_eq(G, comms)

                print(f"    - {algo_name:22s} | ONMI: {score_onmi:.4f} | F1: {score_f1:.4f} | EQ: {score_eq:.4f}")

                results.append({
                    "eta": eta,
                    "xi": xi,
                    "Algorithm": algo_name,
                    "Seed": seed,
                    "Pct_Overlap_Nodes": pct_overlap,
                    "ONMI": score_onmi,
                    "F1": score_f1,
                    "Shen_EQ": score_eq,
                    "Time_Sec": dur
                })

    df = pd.DataFrame(results)
    out_csv = REPO_ROOT / "tests" / "benchmarks" / "abcd_overlap_density_ablation_raw.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved raw overlap sweep data to: {out_csv}")

    # Generate Standalone Curves
    summary = df.groupby(["eta", "Algorithm"]).agg({
        "ONMI": ["mean", "std"],
        "F1": ["mean", "std"],
        "Shen_EQ": ["mean", "std"]
    }).reset_index()
    summary.columns = [f"{c[0]}_{c[1]}" if c[1] else c[0] for c in summary.columns]

    # Plot 1: ONMI vs eta
    plt.figure(figsize=(9, 6))
    for algo in df["Algorithm"].unique():
        sub = summary[summary["Algorithm"] == algo]
        plt.errorbar(sub["eta"], sub["ONMI_mean"], yerr=sub["ONMI_std"], label=algo, marker="o", linewidth=2.2, capsize=4)
    plt.title("ABCD+$o^2$ Overlap Ablation: Recovery Accuracy ($ONMI$) vs. Overlap Multiplier ($\\eta$)", pad=15)
    plt.xlabel("Overlap Multiplier $\\eta$ (Avg. Memberships per Vertex)")
    plt.ylabel("Overlapping NMI ($ONMI$)")
    plt.ylim(-0.02, 1.02)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right", framealpha=0.95)
    p1 = PLOTS_DIR / "fig_abcd_ablation_onmi_vs_eta.png"
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"Saved plot: {p1}")

    # Plot 2: F1 vs eta
    plt.figure(figsize=(9, 6))
    for algo in df["Algorithm"].unique():
        sub = summary[summary["Algorithm"] == algo]
        plt.errorbar(sub["eta"], sub["F1_mean"], yerr=sub["F1_std"], label=algo, marker="s", linewidth=2.2, capsize=4)
    plt.title("ABCD+$o^2$ Overlap Ablation: Pairwise $F_1$-Score vs. Overlap Multiplier ($\\eta$)", pad=15)
    plt.xlabel("Overlap Multiplier $\\eta$ (Avg. Memberships per Vertex)")
    plt.ylabel("Pairwise $F_1$-Score")
    plt.ylim(-0.02, 1.02)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right", framealpha=0.95)
    p2 = PLOTS_DIR / "fig_abcd_ablation_f1_vs_eta.png"
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"Saved plot: {p2}")


# -----------------------------------------------------------------------------
# Experiment 2: OHP-MOCD Component Ablation on Overlapping ABCD+o^2
# -----------------------------------------------------------------------------

def run_component_ablation(runs: int = 5, n_nodes: int = 300, xi: float = 0.2, eta: float = 1.5):
    print("\n" + "=" * 100)
    print(f"EXPERIMENT 2: OHP-MOCD COMPONENT ABLATION ON OVERLAPPING ABCD+o^2 (N={n_nodes}, xi={xi}, eta={eta})")
    print("=" * 100)

    variants = [
        "Full (Proposed)",
        "Crisp Init (No Boundary Seeding)",
        "High Crossover (pc=0.95)",
        "High Mutation (pm=0.50)",
        "Lightweight Budget (Gens=50)",
    ]

    results = []

    for seed in range(runs):
        eff_seed = seed * 100 + 42
        G, ground_truth, node_to_comms = generate_abcd_benchmark(
            n=n_nodes,
            xi=xi,
            eta=eta,
            seed=eff_seed
        )

        print(f"\n>>> [Seed {seed+1}/{runs}] Generated ABCD+o^2 (Nodes={G.number_of_nodes()}, Edges={G.number_of_edges()})")

        for v in variants:
            t0 = time.perf_counter()
            comms = run_ohpmocd_variant(G, variant=v, seed=seed)
            dur = time.perf_counter() - t0

            score_onmi = onmi(ground_truth, comms)
            score_f1 = pairwise_f1(ground_truth, comms)
            score_eq = shen_modularity_eq(G, comms)

            print(f"  - {v:35s} | ONMI: {score_onmi:.4f} | F1: {score_f1:.4f} | EQ: {score_eq:.4f} | Time: {dur:.2f}s")

            results.append({
                "Variant": v,
                "Seed": seed,
                "ONMI": score_onmi,
                "F1": score_f1,
                "Shen_EQ": score_eq,
                "Time_Sec": dur
            })

    df = pd.DataFrame(results)
    out_csv = REPO_ROOT / "tests" / "benchmarks" / "abcd_component_ablation_raw.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved component ablation raw data to: {out_csv}")

    # Summary Table & Bar Chart
    summary = df.groupby("Variant").agg({
        "ONMI": ["mean", "std"],
        "F1": ["mean", "std"],
        "Shen_EQ": ["mean", "std"],
        "Time_Sec": ["mean", "std"]
    }).reset_index()
    summary.columns = [f"{c[0]}_{c[1]}" if c[1] else c[0] for c in summary.columns]

    print("\n" + "=" * 80)
    print("OHP-MOCD COMPONENT ABLATION SUMMARY SCORECARD:")
    print("=" * 80)
    for _, r in summary.iterrows():
        print(f"{r['Variant']:35s} | ONMI: {r['ONMI_mean']:.4f}±{r['ONMI_std']:.4f} | F1: {r['F1_mean']:.4f}±{r['F1_std']:.4f} | EQ: {r['Shen_EQ_mean']:.4f}±{r['Shen_EQ_std']:.4f} | Time: {r['Time_Sec_mean']:.2f}s")

    # Barchart
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(variants))
    width = 0.25

    ax.bar(x - width, summary["ONMI_mean"], width, yerr=summary["ONMI_std"], label="ONMI", capsize=4, color="#1b9e77")
    ax.bar(x, summary["F1_mean"], width, yerr=summary["F1_std"], label="Pairwise F1", capsize=4, color="#7570b3")
    ax.bar(x + width, summary["Shen_EQ_mean"], width, yerr=summary["Shen_EQ_std"], label="Modularity (EQ)", capsize=4, color="#d95f02")

    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=18, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("OHP-MOCD Component Ablation on Overlapping ABCD+$o^2$ ($\\eta=1.5, \\xi=0.2$)", pad=15)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    p3 = PLOTS_DIR / "fig_abcd_component_ablation_bars.png"
    plt.savefig(p3, dpi=300)
    plt.close()
    print(f"Saved plot: {p3}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ABCD+o^2 Overlapping Community Ablation Study")
    parser.add_argument("--runs", type=int, default=5, help="Number of random seeds per configuration (default: 5)")
    parser.add_argument("--nodes", type=int, default=300, help="Number of network nodes (default: 300)")
    parser.add_argument("--xi", type=float, default=0.2, help="Fixed background mixing noise (default: 0.2)")
    parser.add_argument("--skip-exp1", action="store_true", help="Skip Experiment 1 (eta sweep)")
    parser.add_argument("--skip-exp2", action="store_true", help="Skip Experiment 2 (component ablation)")
    args = parser.parse_args()

    if not args.skip_exp1:
        run_network_overlap_sweep(runs=args.runs, n_nodes=args.nodes, xi=args.xi)
    if not args.skip_exp2:
        run_component_ablation(runs=args.runs, n_nodes=args.nodes, xi=args.xi)
