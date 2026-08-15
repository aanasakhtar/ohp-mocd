"""
exp2_celegans_membership_modes.py

Evaluates the 3 Membership Weighting Formulations on Celegans (N=297, M=2148)
across 15 seeds (42..56) at default budget (pop=100, gens=100):
  1. Uniform (r_{v,c} = 1 / |M(v)|) [Native Rust Core]
  2. Pure DWI (Degree-Weighted Neighborhood Influence)
  3. Blended OCCSA + DWI (alpha = 0.50)

Reports:
  - Nicosia Qov (SLPA linear clamp)
  - Shen Extended Modularity (EQ)
  - Merged Community Count
  - Avg Memberships per Node
  - Execution Time
"""

import sys, os, time
import concurrent.futures
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pymocd
from tests.benchmarks.run_paper_comparative_suite import (
    load_celegans, nicosia_qov_slpa, shen_modularity_eq
)
from tests.benchmarks.utils.merge import post_hoc_boundary_merge

def make_dwi_objectives(G: nx.Graph, alpha: float):
    """Creates vectorized objective functions (intra, inter) for custom alpha in [0, 1].
    alpha = 0.0 -> Pure DWI
    alpha = 0.5 -> Blended OCCSA + DWI
    alpha = 1.0 -> Pure OCCSA
    """
    two_m = 2.0 * G.number_of_edges()
    m_edges = G.number_of_edges()
    deg = dict(G.degree())
    adj = {u: list(G.neighbors(u)) for u in G.nodes()}
    edges = list(G.edges())
    
    def calc_weights(partition: dict[int, list[int]]) -> dict[int, dict[int, float]]:
        node_weights = {}
        for u, comms in partition.items():
            if not comms:
                continue
            if len(comms) == 1:
                node_weights[u] = {comms[0]: 1.0}
                continue
                
            comm_set = set(comms)
            nbrs = adj.get(u, [])
            if not nbrs:
                unif = 1.0 / len(comms)
                node_weights[u] = {c: unif for c in comms}
                continue
                
            # OCCSA (unweighted) & DWI (degree-weighted)
            occsa_counts = {c: 0.0 for c in comms}
            dwi_counts = {c: 0.0 for c in comms}
            tot_nbr_deg = sum(deg.get(v, 1) for v in nbrs)
            tot_nbr_cnt = len(nbrs)
            
            for v in nbrs:
                v_comms = partition.get(v, [])
                v_deg = deg.get(v, 1)
                for c in v_comms:
                    if c in comm_set:
                        occsa_counts[c] += 1.0
                        dwi_counts[c] += v_deg
                        
            w_map = {}
            tot_w = 0.0
            for c in comms:
                w_occsa = occsa_counts[c] / tot_nbr_cnt if tot_nbr_cnt > 0 else 1.0 / len(comms)
                w_dwi = dwi_counts[c] / tot_nbr_deg if tot_nbr_deg > 0 else 1.0 / len(comms)
                combined = alpha * w_occsa + (1.0 - alpha) * w_dwi
                w_map[c] = combined
                tot_w += combined
                
            if tot_w > 0:
                for c in w_map:
                    w_map[c] /= tot_w
            else:
                unif = 1.0 / len(comms)
                w_map = {c: unif for c in comms}
            node_weights[u] = w_map
        return node_weights

    def obj_intra(py_graph, partition: dict[int, list[int]]) -> float:
        node_weights = calc_weights(partition)
        intra_sum = 0.0
        for u, v in edges:
            w_u = node_weights.get(u, {})
            w_v = node_weights.get(v, {})
            for c, r_uc in w_u.items():
                if c in w_v:
                    intra_sum += r_uc * w_v[c]
        return float(1.0 - (intra_sum / m_edges))

    def obj_inter(py_graph, partition: dict[int, list[int]]) -> float:
        node_weights = calc_weights(partition)
        comm_degrees = {}
        for u, w_map in node_weights.items():
            d_u = deg.get(u, 0)
            for c, r_uc in w_map.items():
                comm_degrees[c] = comm_degrees.get(c, 0.0) + d_u * r_uc
        inter_sum = sum((cd / two_m) ** 2 for cd in comm_degrees.values())
        return float(inter_sum)

    return [obj_intra, obj_inter]

def run_trial(args):
    mode_name, alpha_val, edge_list, seed_val = args
    G = nx.Graph(edge_list)
    nodes = list(G.nodes())
    node_map = {n: i for i, n in enumerate(nodes)}
    rev_map = {i: n for i, n in enumerate(nodes)}
    H = nx.relabel_nodes(G, node_map, copy=True)
    
    t0 = time.perf_counter()
    if mode_name == "Uniform":
        dict_res = pymocd.ohpmocd(
            H,
            pop_size=100,
            num_gens=100,
            cross_rate=0.8,
            mut_rate=0.5,
            init_strategy="boundary_seeded",
            init_overlap_prob=0.10,
            seed=seed_val
        )
    else:
        # Custom DWI / Blended objectives
        objs = make_dwi_objectives(H, alpha_val)
        ohp_inst = pymocd.OhpMocd(
            H,
            pop_size=100,
            num_gens=100,
            cross_rate=0.8,
            mut_rate=0.5,
            init_strategy="boundary_seeded",
            init_overlap_prob=0.10,
            seed=seed_val,
            objectives=objs
        )
        dict_res = ohp_inst.run()
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
    
    qov = nicosia_qov_slpa(G, merged_comms)
    eq = shen_modularity_eq(G, merged_comms)
    
    return {
        "mode": mode_name,
        "Qov": qov,
        "EQ": eq,
        "raw_count": len(raw_comms),
        "merged_count": len(merged_comms),
        "avg_mems": total_mems / len(nodes),
        "time": dur,
    }

def main():
    print("================================================================================")
    print(" EXPERIMENT 2: CELEGANS MEMBERSHIP FORMULATION COMPARISON (15 SEEDS)")
    print("================================================================================")
    
    G = load_celegans()
    edge_list = list(G.edges())
    
    MODES = [
        ("Uniform", -1.0),
        ("Pure DWI (alpha=0.0)", 0.0),
        ("Blended OCCSA+DWI (alpha=0.5)", 0.5),
    ]
    
    max_workers = max(1, (os.cpu_count() or 4) - 1)
    results = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for mode_label, alpha_val in MODES:
            print(f"\n---> Evaluating Formulation: {mode_label} on Celegans (pop=100, gens=100)...")
            tasks = [
                (mode_label, alpha_val, edge_list, seed)
                for seed in range(42, 42 + 15)
            ]
            futures = [executor.submit(run_trial, t) for t in tasks]
            res_list = [f.result() for f in futures]
            
            qovs = [r["Qov"] for r in res_list]
            eqs = [r["EQ"] for r in res_list]
            raws = [r["raw_count"] for r in res_list]
            merged = [r["merged_count"] for r in res_list]
            mems = [r["avg_mems"] for r in res_list]
            times = [r["time"] for r in res_list]
            
            print(f"  [{mode_label:32s}] Nicosia Qov: {np.mean(qovs):.4f} ± {np.std(qovs):.4f} (peak {np.max(qovs):.4f}) | Shen EQ: {np.mean(eqs):.4f} ± {np.std(eqs):.4f} (peak {np.max(eqs):.4f}) | Merged Comms: {np.mean(merged):.2f} | Mems/node: {np.mean(mems):.3f} | Time/seed: {np.mean(times):.2f}s")
            
            results.append({
                "Dataset": "Celegans",
                "Nodes": G.number_of_nodes(),
                "Edges": G.number_of_edges(),
                "Formulation": mode_label,
                "Qov_mean": np.mean(qovs),
                "Qov_std": np.std(qovs),
                "Qov_peak": np.max(qovs),
                "EQ_mean": np.mean(eqs),
                "EQ_std": np.std(eqs),
                "EQ_peak": np.max(eqs),
                "Raw_Comms_mean": np.mean(raws),
                "Merged_Comms_mean": np.mean(merged),
                "Avg_Mems_Per_Node": np.mean(mems),
                "Time_Per_Seed": np.mean(times),
            })
            
    df = pd.DataFrame(results)
    out_csv = REPO_ROOT / "tests" / "benchmarks" / "exp2_celegans_membership_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[DONE] Saved Celegans formulation results to {out_csv}")

if __name__ == "__main__":
    main()
