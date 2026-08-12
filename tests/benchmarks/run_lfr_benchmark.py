"""
run_lfr_benchmark.py

LFR Synthetic Overlapping Benchmark Suite for OHP-MOCD.

Generates LFR graphs with synthetic overlapping cover (top-k-degree nodes assigned
to additional communities via neighborhood support), matching the design in the
existing load_lfr_overlapping() infrastructure.

Four fixed configs inspired by FCCNI (Shang et al. 2024, Table 5):
  A: N=1000, mu=0.1, on=300, om=2  (analogous to LFR0)
  B: N=1000, mu=0.2, on=300, om=2  (analogous to LFR1)
  C: N=1000, mu=0.1, on=300, om=3  (analogous to LFR2)
  D: N=5000, mu=0.1, on=1500, om=2 (analogous to LFR3)

Plus varying-mu trend suite (N=1000, mu in {0.1..0.5}, on=100, om=2).

NOTE: networkx LFR requires tau2 > 1, so tau2=1.5 is used instead of FCCNI's
tau2=1.0. Community sizes use min_c=10, max_c=50. Direct gNMI comparison to
FCCNI's Table 8 values should be treated as indicative, not exact.

FCCNI Table 8 gNMI_max baselines (exact values from paper):
  LFR0: FCCNI=0.5520, SLPA=0.2649, NI-LPA=0.3604
  LFR1: FCCNI=0.5546, SLPA=0.1899, NI-LPA=0.3116
  LFR2: FCCNI=0.4280, SLPA=0.2414, NI-LPA=0.3266
  LFR3: FCCNI=0.2231, SLPA=0.2311, NI-LPA=0.4162 (NI-LPA wins here!)

Outputs: tests/benchmarks/lfr_benchmark_results.csv
"""

import sys
import time
import math
import random
import os
import numpy as np
import pandas as pd
import networkx as nx
import concurrent.futures
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pymocd
from evaluation.metrics import onmi

BENCH_DIR = REPO_ROOT / "tests" / "benchmarks"

# -----------------------------------------------------------------------
# Hyperparameters (from full-OCCSA grid search)
# -----------------------------------------------------------------------
PARAMS_SMALL = {"init_p": 0.15, "supp_th": 0.35, "rem_th": 0.25, "margin": 0.05}
PARAMS_LARGE = {"init_p": 0.15, "supp_th": 0.35, "rem_th": 0.25, "margin": 0.05}

N_SEEDS = 5
N_WORKERS = max(1, (os.cpu_count() or 4) - 1)

# -----------------------------------------------------------------------
# LFR Configuration Sets
# -----------------------------------------------------------------------

# FCCNI-inspired fixed configs (adapted for networkx compatibility: tau2=1.5)
FCCNI_INSPIRED_CONFIGS = [
    {"label": "LFR-A (N=1k,mu=0.1,on=300,om=2)", "n": 1000, "mu": 0.1,
     "on": 300, "om": 2, "tau1": 3, "tau2": 1.5, "avg_deg": 5, "max_deg": 50, "min_c": 10, "max_c": 50},
    {"label": "LFR-B (N=1k,mu=0.2,on=300,om=2)", "n": 1000, "mu": 0.2,
     "on": 300, "om": 2, "tau1": 3, "tau2": 1.5, "avg_deg": 5, "max_deg": 50, "min_c": 10, "max_c": 50},
    {"label": "LFR-C (N=1k,mu=0.1,on=300,om=3)", "n": 1000, "mu": 0.1,
     "on": 300, "om": 3, "tau1": 3, "tau2": 1.5, "avg_deg": 5, "max_deg": 50, "min_c": 10, "max_c": 50},
    {"label": "LFR-D (N=5k,mu=0.1,on=1500,om=2)", "n": 5000, "mu": 0.1,
     "on": 1500, "om": 2, "tau1": 3, "tau2": 1.5, "avg_deg": 5, "max_deg": 50, "min_c": 10, "max_c": 50},
]

# Exact FCCNI Table 8 gNMI_max baselines for reference
FCCNI_TABLE8 = {
    "LFR-A (N=1k,mu=0.1,on=300,om=2)":  {"FCCNI": 0.5520, "SLPA": 0.2649, "NI_LPA": 0.3604, "MOEA_SAov": 0.2630, "CEMOV": 0.3165},
    "LFR-B (N=1k,mu=0.2,on=300,om=2)":  {"FCCNI": 0.5546, "SLPA": 0.1899, "NI_LPA": 0.3116, "MOEA_SAov": 0.2220, "CEMOV": 0.3133},
    "LFR-C (N=1k,mu=0.1,on=300,om=3)":  {"FCCNI": 0.4280, "SLPA": 0.2414, "NI_LPA": 0.3266, "MOEA_SAov": 0.1701, "CEMOV": 0.3004},
    "LFR-D (N=5k,mu=0.1,on=1500,om=2)": {"FCCNI": 0.2231, "SLPA": 0.2311, "NI_LPA": 0.4162, "MOEA_SAov": 0.1525, "CEMOV": None},
}

# Varying-mu trend suite (MCMOEA / SLPA style)
MU_TREND_CONFIGS = [
    {"label": f"LFR-mu{int(mu*10):02d} (N=1k,mu={mu:.1f},on=100,om=2)",
     "n": 1000, "mu": mu, "on": 100, "om": 2,
     "tau1": 3, "tau2": 1.5, "avg_deg": 10, "max_deg": 50, "min_c": 10, "max_c": 50}
    for mu in [0.1, 0.2, 0.3, 0.4, 0.5]
]

ALL_CONFIGS = FCCNI_INSPIRED_CONFIGS + MU_TREND_CONFIGS

# -----------------------------------------------------------------------
# Overlapping LFR generation (inline, self-contained)
# -----------------------------------------------------------------------

def generate_lfr_overlapping(cfg, seed, max_tries=8):
    """Generate LFR graph with synthetic overlapping cover."""
    for attempt in range(max_tries):
        try:
            G = nx.LFR_benchmark_graph(
                n=cfg["n"], tau1=cfg["tau1"], tau2=cfg["tau2"], mu=cfg["mu"],
                average_degree=cfg["avg_deg"], max_degree=cfg["max_deg"],
                min_community=cfg["min_c"], max_community=cfg["max_c"],
                seed=seed + attempt * 997, max_iters=1000,
            )
            # Build community map from disjoint ground truth
            comm_sets_raw = list({frozenset(G.nodes[v]["community"]) for v in G})
            community_map = {}
            for cid, fs in enumerate(comm_sets_raw):
                for v in fs:
                    community_map.setdefault(v, []).append(cid)
            community_sets = [set(fs) for fs in comm_sets_raw]

            # Assign top-degree nodes to additional communities (synthetic overlapping)
            rng = random.Random(seed + attempt)
            nodes_by_deg = sorted(G.nodes(), key=lambda v: G.degree(v), reverse=True)
            overlap_n = min(cfg["on"], len(nodes_by_deg))
            overlap_k = cfg["om"]

            for node in nodes_by_deg[:overlap_n]:
                current = set(community_map.get(node, []))
                extra = overlap_k - len(current)
                if extra <= 0:
                    continue
                nbr_supp = Counter(
                    c for nbr in G.neighbors(node)
                    for c in community_map.get(nbr, [])
                    if c not in current
                )
                while extra > 0:
                    target = next((c for c, _ in nbr_supp.most_common() if c not in current), None)
                    if target is None:
                        avail = [c for c in range(len(community_sets)) if c not in current]
                        if not avail:
                            break
                        target = rng.choice(avail)
                    current.add(target)
                    community_map[node].append(target)
                    community_sets[target].add(node)
                    extra -= 1

            G2 = nx.Graph(G)
            G2.remove_edges_from(list(nx.selfloop_edges(G2)))
            gt = [frozenset(s) for s in community_sets if s]
            return G2, gt, community_map
        except Exception:
            continue
    raise RuntimeError(f"LFR generation failed for: {cfg['label']}")


def overlapping_f1_score(detected_comms, gt_comms, gt_overlap_nodes):
    """Pairwise F1 for overlapping node detection."""
    if not gt_overlap_nodes:
        return float("nan"), float("nan"), float("nan")
    node_to_det = {}
    for cid, c in enumerate(detected_comms):
        for v in c:
            node_to_det.setdefault(v, set()).add(cid)
    detected_overlap = {v for v in node_to_det if len(node_to_det[v]) > 1}
    gt_set = set(gt_overlap_nodes)
    tp = len(detected_overlap & gt_set)
    fp = len(detected_overlap - gt_set)
    fn = len(gt_set - detected_overlap)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


# -----------------------------------------------------------------------
# Worker
# -----------------------------------------------------------------------

def run_lfr_seed(task):
    cfg_label, cfg, seed, init_strategy = task
    try:
        G, gt, gt_comm_map = generate_lfr_overlapping(cfg, seed)
    except RuntimeError as e:
        return {"label": cfg_label, "seed": seed, "init": init_strategy,
                "gNMI": float("nan"), "Qov": float("nan"),
                "f1_overlap": float("nan"), "prec_overlap": float("nan"), "rec_overlap": float("nan"),
                "n": cfg["n"], "mu": cfg["mu"], "on": cfg["on"], "om": cfg["om"],
                "time_s": 0.0, "error": str(e)}

    nodes = sorted(G.nodes())
    node_map = {n: i for i, n in enumerate(nodes)}
    rev_map  = {i: n for i, n in enumerate(nodes)}
    H = nx.relabel_nodes(G, node_map, copy=True)

    params = PARAMS_LARGE if cfg["n"] >= 5000 else PARAMS_SMALL

    t0 = time.perf_counter()
    dict_res = pymocd.ohpmocd(
        H,
        init_strategy=init_strategy,
        init_overlap_prob=params["init_p"],
        overlap_support_threshold=params["supp_th"],
        overlap_removal_threshold=params["rem_th"],
        switch_margin=params["margin"],
        seed=None,
    )
    elapsed = time.perf_counter() - t0

    comm_dict = {}
    for n_idx, comm_list in dict_res.items():
        orig = rev_map[n_idx]
        if isinstance(comm_list, (int, np.integer)):
            comm_list = [comm_list]
        for cid in comm_list:
            comm_dict.setdefault(cid, set()).add(orig)
    comms = [frozenset(c) for c in comm_dict.values()]

    gnmi = onmi(comms, gt) if gt else float("nan")

    # Nicosia Qov
    m = G.number_of_edges()
    qov = 0.0
    if m > 0:
        deg = dict(G.degree())
        ob = {}
        for c in comm_dict.values():
            for u in c:
                ob[u] = ob.get(u, 0) + 1
        two_m = 2.0 * m
        for c in comm_dict.values():
            for u in c:
                for v in c:
                    f = (1.0 / ob[u]) * (1.0 / ob[v])
                    A = 1.0 if G.has_edge(u, v) else 0.0
                    qov += f * A - (deg.get(u, 0) * deg.get(v, 0) / two_m) * f
        qov /= two_m

    # Overlapping node detection F1
    gt_overlap_nodes = [v for v, comms_v in gt_comm_map.items() if len(comms_v) > 1]
    prec, rec, f1 = overlapping_f1_score(comms, gt, gt_overlap_nodes)

    return {
        "label": cfg_label, "seed": seed, "init": init_strategy,
        "gNMI": gnmi, "Qov": qov,
        "f1_overlap": f1, "prec_overlap": prec, "rec_overlap": rec,
        "n": cfg["n"], "mu": cfg["mu"], "on": cfg["on"], "om": cfg["om"],
        "time_s": elapsed, "error": None,
    }


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    print("=" * 70)
    print(" LFR OVERLAPPING BENCHMARK SUITE — OHP-MOCD")
    print("=" * 70)
    print(f" Configs: {len(ALL_CONFIGS)} | Seeds: {N_SEEDS} | Workers: {N_WORKERS}")
    print()

    tasks = [
        (cfg["label"], cfg, seed, strat)
        for cfg in ALL_CONFIGS
        for seed in range(N_SEEDS)
        for strat in ["boundary_seeded", "crisp"]
    ]
    print(f" Total tasks: {len(tasks)}\n")

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(run_lfr_seed, t): t for t in tasks}
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            done += 1
            r = fut.result()
            results.append(r)
            if done % 10 == 0 or done == len(tasks):
                print(f"  [{done}/{len(tasks)}] {r['label']} | {r['init'][:2]} | gNMI={r['gNMI']:.4f} | f1_ovlp={r['f1_overlap']:.4f}")

    df = pd.DataFrame(results)

    agg = (
        df.groupby(["label", "init", "n", "mu", "on", "om"])
        .agg(
            gNMI_max=("gNMI", "max"), gNMI_mean=("gNMI", "mean"), gNMI_std=("gNMI", "std"),
            Qov_max=("Qov", "max"), Qov_mean=("Qov", "mean"),
            f1_overlap_max=("f1_overlap", "max"), f1_overlap_mean=("f1_overlap", "mean"),
            prec_mean=("prec_overlap", "mean"), rec_mean=("rec_overlap", "mean"),
        )
        .reset_index()
    )

    bs = agg[agg["init"] == "boundary_seeded"].copy()
    cr = agg[agg["init"] == "crisp"].copy()
    merged = bs.merge(cr, on=["label", "n", "mu", "on", "om"], suffixes=("_BS", "_Crisp"))

    for algo in ["FCCNI", "SLPA", "NI_LPA", "MOEA_SAov", "CEMOV"]:
        merged[f"{algo}_gNMI"] = merged["label"].map(
            lambda lbl, a=algo: FCCNI_TABLE8.get(lbl, {}).get(a, None)
        )

    merged["OHP_Best_gNMI"]     = merged[["gNMI_max_BS", "gNMI_max_Crisp"]].max(axis=1)
    merged["OHP_Best_f1_ovlp"]  = merged[["f1_overlap_max_BS", "f1_overlap_max_Crisp"]].max(axis=1)
    merged["vs_FCCNI_gNMI"]     = merged.apply(
        lambda r: r["OHP_Best_gNMI"] - float(r["FCCNI_gNMI"])
        if r["FCCNI_gNMI"] is not None else float("nan"), axis=1)

    out = BENCH_DIR / "lfr_benchmark_results.csv"
    merged.to_csv(out, index=False)
    print(f"\nSaved: {out}")

    # Summary
    print()
    print(f"{'Config':<44} {'FCCNI†':>7} {'OHP_BS':>8} {'OHP_Cr':>8} {'Best':>7} {'vs FCCNI':>10} {'F1_ovlp':>9}")
    print("-" * 100)
    for _, row in merged.iterrows():
        fv = row["FCCNI_gNMI"]
        fs = f"{float(fv):.4f}" if fv is not None else "  N/A  "
        dv = row["vs_FCCNI_gNMI"]
        ds = f"{dv:+.4f}" if not math.isnan(dv) else "   N/A  "
        win = " WIN" if not math.isnan(dv) and dv >= 0 else ""
        print(f"{str(row['label']):<44} {fs:>7} {row['gNMI_max_BS']:>8.4f} {row['gNMI_max_Crisp']:>8.4f} "
              f"{row['OHP_Best_gNMI']:>7.4f} {ds:>10}{win}  f1={row['OHP_Best_f1_ovlp']:.4f}")

    print()
    print("† FCCNI values from Table 8 (Shang et al. 2024); note tau2=1.5 vs paper's tau2=1.0.")
    print("LFR BENCHMARK COMPLETE.")


if __name__ == "__main__":
    main()
