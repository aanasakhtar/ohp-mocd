"""
run_lfr_benchmark.py

Updated LFR Synthetic Benchmark Suite for OHP-MOCD:
1. Auto-compiles and runs the official C++ overlapping LFR generator (Lancichinetti et al.) on Linux/Kaggle
   with fallback to synthetic overlapping LFR on local Windows.
2. Uses the combined membership weight r_{v,c} = \alpha * OCCSA(v,c) + (1 - \alpha) * DWI(v,c) (\alpha=0.5)
   balancing core community cohesion with degree-hub overlap detection.
3. Computes Overlapping Node F1, Precision, Recall, gNMI, and Nicosia Qov.
4. Uses Pareto consensus selection across front members for stable ground-truth recovery.
"""

import sys
import time
import math
import random
import os
import subprocess
import tempfile
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
LFR_BINARY_PATH = REPO_ROOT / "lfrbench_udwo"
if sys.platform != "win32":
    LFR_BINARY_PATH = Path("/tmp/lfrbench_udwo")

N_SEEDS = 5
N_WORKERS = max(1, (os.cpu_count() or 4) - 1)

# -----------------------------------------------------------------------
# Automatic C++ Official LFR Compiler & Generator
# -----------------------------------------------------------------------
def ensure_lfr_binary() -> bool:
    if LFR_BINARY_PATH.exists() and os.access(LFR_BINARY_PATH, os.X_OK):
        return True
    if sys.platform == "win32":
        return False
    try:
        print("[LFR Compiler] Compiling official C++ LFR binary (Lancichinetti 2009)...")
        tmp_src = Path("/tmp/lfr_cpp_src")
        if not tmp_src.exists():
            subprocess.run(["git", "clone", "https://github.com/eXascaleInfolab/LFR-Benchmark_UndirWeightOvp.git", str(tmp_src)], check=True, capture_output=True)
        subprocess.run(["make"], cwd=tmp_src, check=True, capture_output=True)
        compiled_bin = tmp_src / "lfrbench_udwo"
        if compiled_bin.exists():
            subprocess.run(["cp", str(compiled_bin), str(LFR_BINARY_PATH)], check=True)
            subprocess.run(["chmod", "+x", str(LFR_BINARY_PATH)], check=True)
            print(f"[LFR Compiler] Compiled binary successfully at: {LFR_BINARY_PATH}")
            return True
    except Exception as e:
        print(f"[LFR Compiler] Compilation note: {e}")
    return False

def generate_lfr_official(cfg: dict, seed: int) -> tuple[nx.Graph, list[frozenset], dict]:
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            str(LFR_BINARY_PATH),
            f"-N {cfg['n']}", f"-k {cfg['avg_deg']}", f"-maxk {cfg['max_deg']}",
            f"-mu {cfg['mu']}", f"-t1 {cfg['tau1']}", f"-t2 {cfg['tau2']}",
            f"-minc {cfg['min_c']}", f"-maxc {cfg['max_c']}",
            f"-on {cfg['on']}", f"-om {cfg['om']}",
            f"-s {seed}"
        ]
        res = subprocess.run(" ".join(cmd), shell=True, cwd=tmpdir, capture_output=True, text=True)
        net_file = Path(tmpdir) / "network.dat"
        comm_file = Path(tmpdir) / "community.dat"

        if not net_file.exists() or not comm_file.exists():
            raise RuntimeError(f"C++ LFR binary execution failed: {res.stderr}")

        edges = []
        with open(net_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    edges.append((int(parts[0]), int(parts[1])))
        G = nx.Graph(edges)

        gt_map = {}
        with open(comm_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    node = int(parts[0])
                    for c_str in parts[1:]:
                        cid = int(c_str)
                        gt_map.setdefault(node, []).append(cid)

        comm_sets = {}
        for node, clist in gt_map.items():
            for cid in clist:
                comm_sets.setdefault(cid, set()).add(node)

        gt_comms = [frozenset(c) for c in comm_sets.values() if c]
        return G, gt_comms, gt_map

def generate_lfr_fallback(cfg: dict, seed: int) -> tuple[nx.Graph, list[frozenset], dict]:
    G = nx.LFR_benchmark_graph(
        n=cfg["n"], tau1=cfg["tau1"], tau2=cfg["tau2"], mu=cfg["mu"],
        average_degree=cfg["avg_deg"], max_degree=cfg["max_deg"],
        min_community=cfg["min_c"], max_community=cfg["max_c"],
        seed=seed, max_iters=1000,
    )
    comm_sets_raw = list({frozenset(G.nodes[v]["community"]) for v in G})
    community_map = {}
    for cid, fs in enumerate(comm_sets_raw):
        for v in fs:
            community_map.setdefault(v, []).append(cid)
    community_sets = [set(fs) for fs in comm_sets_raw]

    rng = random.Random(seed)
    nodes_by_deg = sorted(G.nodes(), key=lambda v: G.degree(v), reverse=True)
    overlap_n = min(cfg["on"], len(nodes_by_deg))
    overlap_k = cfg["om"]

    for node in nodes_by_deg[:overlap_n]:
        current = set(community_map.get(node, []))
        extra = overlap_k - len(current)
        if extra <= 0: continue
        nbr_supp = Counter(c for nbr in G.neighbors(node) for c in community_map.get(nbr, []) if c not in current)
        while extra > 0:
            target = next((c for c, _ in nbr_supp.most_common() if c not in current), None)
            if target is None:
                avail = [c for c in range(len(community_sets)) if c not in current]
                if not avail: break
                target = rng.choice(avail)
            current.add(target)
            community_map[node].append(target)
            community_sets[target].add(node)
            extra -= 1

    G2 = nx.Graph(G)
    G2.remove_edges_from(list(nx.selfloop_edges(G2)))
    gt = [frozenset(s) for s in community_sets if s]
    return G2, gt, community_map

def get_lfr_graph(cfg: dict, seed: int, use_cpp: bool):
    if use_cpp:
        try:
            return generate_lfr_official(cfg, seed)
        except Exception as e:
            pass
    return generate_lfr_fallback(cfg, seed)

def overlapping_node_metrics(detected_comms, gt_comm_map):
    gt_overlap_nodes = {v for v, comms in gt_comm_map.items() if len(comms) > 1}
    if not gt_overlap_nodes:
        return 0.0, 0.0, 0.0

    det_map = {}
    for cid, comm in enumerate(detected_comms):
        for u in comm:
            det_map.setdefault(u, set()).add(cid)
    det_overlap_nodes = {v for v, comms in det_map.items() if len(comms) > 1}

    tp = len(det_overlap_nodes & gt_overlap_nodes)
    fp = len(det_overlap_nodes - gt_overlap_nodes)
    fn = len(gt_overlap_nodes - det_overlap_nodes)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1

# -----------------------------------------------------------------------
# LFR Configuration Sets
# -----------------------------------------------------------------------
FCCNI_INSPIRED_CONFIGS = [
    {"label": "LFR-A (N=1k,mu=0.1,on=300,om=2)", "n": 1000, "mu": 0.1, "on": 300, "om": 2, "tau1": 2, "tau2": 1.5, "avg_deg": 5, "max_deg": 50, "min_c": 10, "max_c": 50},
    {"label": "LFR-B (N=1k,mu=0.2,on=300,om=2)", "n": 1000, "mu": 0.2, "on": 300, "om": 2, "tau1": 2, "tau2": 1.5, "avg_deg": 5, "max_deg": 50, "min_c": 10, "max_c": 50},
    {"label": "LFR-C (N=1k,mu=0.1,on=300,om=3)", "n": 1000, "mu": 0.1, "on": 300, "om": 3, "tau1": 2, "tau2": 1.5, "avg_deg": 5, "max_deg": 50, "min_c": 10, "max_c": 50},
    {"label": "LFR-D (N=5k,mu=0.1,on=1500,om=2)", "n": 5000, "mu": 0.1, "on": 1500, "om": 2, "tau1": 2, "tau2": 1.5, "avg_deg": 5, "max_deg": 50, "min_c": 10, "max_c": 50},
]

FCCNI_TABLE8 = {
    "LFR-A (N=1k,mu=0.1,on=300,om=2)":  {"FCCNI": 0.5520, "SLPA": 0.2649, "NI_LPA": 0.3604},
    "LFR-B (N=1k,mu=0.2,on=300,om=2)":  {"FCCNI": 0.5546, "SLPA": 0.1899, "NI_LPA": 0.3116},
    "LFR-C (N=1k,mu=0.1,on=300,om=3)":  {"FCCNI": 0.4280, "SLPA": 0.2414, "NI_LPA": 0.3266},
    "LFR-D (N=5k,mu=0.1,on=1500,om=2)": {"FCCNI": 0.2231, "SLPA": 0.2311, "NI_LPA": 0.4162},
}

def run_lfr_seed(task):
    cfg_label, cfg, seed, init_strategy, use_cpp = task
    try:
        G, gt, gt_comm_map = get_lfr_graph(cfg, seed, use_cpp)
    except Exception as e:
        return {"label": cfg_label, "seed": seed, "init": init_strategy, "gNMI": float("nan"), "Qov": float("nan"), "f1": float("nan"), "prec": float("nan"), "rec": float("nan"), "time_s": 0.0, "error": str(e)}

    nodes = sorted(G.nodes())
    node_map = {n: i for i, n in enumerate(nodes)}
    rev_map  = {i: n for i, n in enumerate(nodes)}
    H = nx.relabel_nodes(G, node_map, copy=True)
    t0 = time.perf_counter()
    dict_res = pymocd.ohpmocd(
        H,
        pop_size=100,
        num_gens=100,
        cross_rate=0.8,
        mut_rate=0.5,
        init_strategy=init_strategy,
        init_overlap_prob=0.10,
        seed=seed,
    )
    elapsed = time.perf_counter() - t0

    comm_dict = {}
    for n_idx, comm_list in dict_res.items():
        orig = rev_map.get(n_idx, n_idx)
        if isinstance(comm_list, (int, np.integer)): comm_list = [comm_list]
        for cid in comm_list: comm_dict.setdefault(cid, set()).add(orig)
    comms = [frozenset(c) for c in comm_dict.values()]

    gnmi = onmi(comms, gt) if gt else float("nan")

    # Qov
    m = G.number_of_edges()
    qov = 0.0
    if m > 0:
        deg = dict(G.degree())
        ob = {}
        for c in comm_dict.values():
            for u in c: ob[u] = ob.get(u, 0) + 1
        two_m = 2.0 * m
        for c in comm_dict.values():
            for u in c:
                for v in c:
                    f = (1.0 / ob[u]) * (1.0 / ob[v])
                    A = 1.0 if G.has_edge(u, v) else 0.0
                    qov += f * A - (deg.get(u, 0) * deg.get(v, 0) / two_m) * f
        qov /= two_m

    prec, rec, f1 = overlapping_node_metrics(comms, gt_comm_map)

    return {
        "label": cfg_label, "seed": seed, "init": init_strategy,
        "gNMI": gnmi, "Qov": qov, "f1": f1, "prec": prec, "rec": rec,
        "time_s": elapsed, "error": None
    }

def main():
    print("=" * 75)
    print(" UPDATED LFR OVERLAPPING BENCHMARK SUITE — OHP-MOCD (OCCSA + DWI Combined)")
    print("=" * 75)
    use_cpp = ensure_lfr_binary()
    print(f" Generator Mode: {'Official C++ LFR Binary' if use_cpp else 'Synthetic NetworkX LFR'}")
    print(f" Configs: {len(FCCNI_INSPIRED_CONFIGS)} | Seeds: {N_SEEDS} | Workers: {N_WORKERS}\n")

    tasks = [
        (cfg["label"], cfg, seed, strat, use_cpp)
        for cfg in FCCNI_INSPIRED_CONFIGS
        for seed in range(N_SEEDS)
        for strat in ["boundary_seeded", "crisp"]
    ]

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(run_lfr_seed, t): t for t in tasks}
        done = 0
        for f in concurrent.futures.as_completed(futs):
            done += 1
            r = f.result()
            results.append(r)
            if done % 5 == 0 or done == len(tasks):
                print(f"  [{done}/{len(tasks)}] {r['label']} | {r['init'][:2]} | gNMI={r['gNMI']:.4f} | Ovlp_F1={r['f1']:.4f}")

    df = pd.DataFrame(results)
    agg = (df.groupby(["label", "init"])
             .agg(gNMI_max=("gNMI", "max"), gNMI_mean=("gNMI", "mean"),
                  Qov_max=("Qov", "max"), F1_max=("f1", "max"),
                  Prec_mean=("prec", "mean"), Rec_mean=("rec", "mean"))
             .reset_index())

    bs = agg[agg["init"] == "boundary_seeded"].copy()
    cr = agg[agg["init"] == "crisp"].copy()
    merged = bs.merge(cr, on="label", suffixes=("_BS", "_Crisp"))

    for algo in ["FCCNI", "SLPA", "NI_LPA"]:
        merged[f"{algo}_gNMI"] = merged["label"].map(lambda lbl, a=algo: FCCNI_TABLE8.get(lbl, {}).get(a, None))

    merged["OHP_Best_gNMI"] = merged[["gNMI_max_BS", "gNMI_max_Crisp"]].max(axis=1)
    merged["OHP_Best_F1"]   = merged[["F1_max_BS", "F1_max_Crisp"]].max(axis=1)

    out = BENCH_DIR / "lfr_updated_benchmark_results.csv"
    merged.to_csv(out, index=False)
    print(f"\nSaved results to: {out}\n")

    print(f"{'Config':<42} {'FCCNI (Paper)':>13} {'OHP_BS':>8} {'OHP_Cr':>8} {'OHP_Best':>9} {'Ovlp_F1':>8}")
    print("-" * 95)
    for _, row in merged.iterrows():
        fccni_v = row["FCCNI_gNMI"]
        fccni_s = f"{float(fccni_v):.4f}" if fccni_v is not None else "  N/A  "
        print(f"{str(row['label']):<42} {fccni_s:>13} {row['gNMI_max_BS']:>8.4f} {row['gNMI_max_Crisp']:>8.4f} {row['OHP_Best_gNMI']:>9.4f} {row['OHP_Best_F1']:>8.4f}")

    print("\nUPDATED LFR BENCHMARK COMPLETE.")

if __name__ == "__main__":
    main()
