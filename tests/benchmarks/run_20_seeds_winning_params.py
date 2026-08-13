"""
run_20_seeds_winning_params.py

Loads optimal dataset parameters from optimal_dataset_parameters.csv,
runs 20 independent random seeds for each dataset, and records:
  - Mean gNMI (ONMI) & Std Dev
  - Mean EQ & Std Dev
  - Mean Qov (Unscaled & SLPA-Formulated) & Std Dev
  - Execution Time Mean & Std Dev
"""

import sys, os
sys.path.insert(0, os.path.abspath('.'))

import time
import numpy as np
import networkx as nx
import pandas as pd
import pymocd
from pathlib import Path
from evaluation.metrics import onmi

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BENCH_DIR = REPO_ROOT / "tests" / "benchmarks"

# Metric Definitions
def nicosia_qov(G: nx.Graph, communities: list[set]) -> float:
    m = G.number_of_edges()
    if m == 0: return 0.0
    two_m = 2.0 * m
    deg = dict(G.degree())
    node_belong = {}
    for comm in communities:
        for u in comm: node_belong[u] = node_belong.get(u, 0) + 1
    qov = 0.0
    for comm in communities:
        for u in comm:
            for v in comm:
                f_val = (1.0 / node_belong[u]) * (1.0 / node_belong[v])
                A_uv = 1.0 if G.has_edge(u, v) else 0.0
                qov += f_val * (A_uv - (deg[u] * deg[v] / two_m))
    return float(qov / two_m)

def nicosia_qov_slpa_strict(G: nx.Graph, communities: list[set]) -> float:
    m = G.number_of_edges()
    if m == 0 or not communities: return 0.0
    two_m = 2.0 * m
    deg = dict(G.degree())
    N = G.number_of_nodes()
    node_belong = {}
    for comm in communities:
        for u in comm: node_belong[u] = node_belong.get(u, 0) + 1
    qov = 0.0
    for comm in communities:
        sum_f_c = sum(1.0 / (1.0 + np.exp(-60.0 * ((1.0 / node_belong[u]) - 0.5))) if node_belong[u] > 1 else 1.0 for u in comm)
        avg_f_c = sum_f_c / float(N)
        for u in comm:
            r_u = 1.0 / node_belong[u]
            f_u = 1.0 / (1.0 + np.exp(-60.0 * (r_u - 0.5))) if node_belong[u] > 1 else 1.0
            l_out = f_u * avg_f_c
            for v in comm:
                r_v = 1.0 / node_belong[v]
                f_v = 1.0 / (1.0 + np.exp(-60.0 * (r_v - 0.5))) if node_belong[v] > 1 else 1.0
                l_in = f_v * avg_f_c
                l_uv = f_u * f_v
                s_uv = l_out * l_in
                A_uv = 1.0 if G.has_edge(u, v) else 0.0
                k_u = deg.get(u, 0)
                k_v = deg.get(v, 0)
                qov += l_uv * A_uv - s_uv * ((k_u * k_v) / two_m)
    return float(qov / two_m)

def shen_modularity_eq(G: nx.Graph, communities: list[set]) -> float:
    m = G.number_of_edges()
    if m == 0: return 0.0
    two_m = 2.0 * m
    deg = dict(G.degree())
    node_belong = {}
    for comm in communities:
        for u in comm: node_belong[u] = node_belong.get(u, 0) + 1
    eq = 0.0
    for comm in communities:
        for u in comm:
            for v in comm:
                A_uv = 1.0 if G.has_edge(u, v) else 0.0
                eq += (1.0 / (node_belong[u] * node_belong[v])) * (A_uv - (deg[u] * deg[v] / two_m))
    return float(eq / two_m)

def post_hoc_boundary_merge(G: nx.Graph, communities: list[set], merge_threshold: float = 0.35) -> list[set]:
    if len(communities) <= 1: return communities
    m = G.number_of_edges()
    if m == 0: return communities
    two_m = 2.0 * m
    deg = dict(G.degree())
    merged_comms = [set(c) for c in communities if c]
    changed = True
    while changed and len(merged_comms) > 1:
        changed = False
        best_pair = None
        best_gain = 0.0
        for i in range(len(merged_comms)):
            for j in range(i + 1, len(merged_comms)):
                c1, c2 = merged_comms[i], merged_comms[j]
                e_inter = sum(1 for u in c1 for v in c2 if G.has_edge(u, v))
                if e_inter == 0: continue
                deg_c1 = sum(deg.get(u, 0) for u in c1)
                deg_c2 = sum(deg.get(u, 0) for u in c2)
                delta_q = (2.0 * e_inter / two_m) - (2.0 * deg_c1 * deg_c2 / (two_m * two_m))
                min_size = min(len(c1), len(c2))
                bound_ratio = e_inter / min_size if min_size > 0 else 0.0
                if delta_q > 0.0 and bound_ratio >= merge_threshold:
                    if delta_q > best_gain:
                        best_gain = delta_q
                        best_pair = (i, j)
        if best_pair is not None:
            i, j = best_pair
            merged_comms[i] = merged_comms[i].union(merged_comms[j])
            merged_comms.pop(j)
            changed = True
    return merged_comms

# Loaders
from tests.benchmarks.run_paper_comparative_suite import (
    load_karate, load_dolphins, load_lesmis, load_polbooks, load_football, load_netscience, load_celegans, extract_ground_truth
)

LOADERS = {
    "Karate": load_karate,
    "Dolphins": load_dolphins,
    "Lesmis": load_lesmis,
    "Polbooks": load_polbooks,
    "Football": load_football,
    "Netscience": load_netscience,
    "Celegans": load_celegans,
}

def main():
    opt_csv = BENCH_DIR / "optimal_dataset_parameters.csv"
    if not opt_csv.exists():
        print(f"Error: {opt_csv} does not exist. Run grid_search_dataset_params.py first.")
        return
        
    df_opt = pd.read_csv(opt_csv)
    print("=================================================================")
    print(" RUNNING 20 INDEPENDENT SEEDS FOR WINNING DATASET PARAMETERS ")
    print("=================================================================\n")
    
    summary_rows = []
    
    for _, r in df_opt.iterrows():
        net_name = r["Dataset"]
        if net_name not in LOADERS: continue
        
        loader = LOADERS[net_name]
        G_obj = loader()
        G = G_obj[0] if isinstance(G_obj, tuple) else G_obj
        nodes = list(G.nodes())
        node_map = {n: i for i, n in enumerate(nodes)}
        rev_map = {i: n for i, n in enumerate(nodes)}
        H = nx.relabel_nodes(G, node_map, copy=True)
        gt = extract_ground_truth(G, net_name)
        
        strat = r["best_strategy"]
        alpha = float(r["best_alpha"])
        p_init = float(r["best_init_overlap_prob"])
        supp_th = float(r["best_overlap_support_threshold"])
        rem_th = float(r["best_overlap_removal_threshold"])
        margin = float(r["best_switch_margin"])
        merge_th_str = str(r["best_merge_threshold"])
        merge_th = float(merge_th_str) if merge_th_str != "None" else None
        
        gnmis, eqs, qovs, qov_slpas, times = [], [], [], [], []
        
        print(f"-> Evaluating {net_name} across 15 Seeds (strat={strat}, alpha={alpha}, supp_th={supp_th}, rem_th={rem_th}, merge_th={merge_th})...")
        
        for seed in range(15):
            t0 = time.perf_counter()
            dict_res = pymocd.ohpmocd(
                H,
                init_strategy=strat,
                init_overlap_prob=p_init,
                overlap_support_threshold=supp_th,
                overlap_removal_threshold=rem_th,
                switch_margin=margin,
                alpha=alpha,
                seed=seed
            )
            dur = time.perf_counter() - t0
            
            comm_dict = {}
            for n_idx, comm_list in dict_res.items():
                orig_node = rev_map[n_idx]
                if isinstance(comm_list, (int, np.integer)): comm_list = [comm_list]
                for cid in comm_list: comm_dict.setdefault(cid, set()).add(orig_node)
            comms = [set(m) for m in comm_dict.values() if m]
            
            if merge_th is not None:
                comms = post_hoc_boundary_merge(G, comms, merge_threshold=merge_th)
                
            qov = nicosia_qov(G, comms)
            qov_slpa = nicosia_qov_slpa_strict(G, comms)
            eq = shen_modularity_eq(G, comms)
            gnmi_val = onmi([frozenset(c) for c in comms], gt) if gt is not None else 0.0
            
            gnmis.append(gnmi_val)
            eqs.append(eq)
            qovs.append(qov)
            qov_slpas.append(qov_slpa)
            times.append(dur)
            
        summary_rows.append({
            "Dataset": net_name,
            "Strategy": strat,
            "Alpha": alpha,
            "Supp_Th": supp_th,
            "Rem_Th": rem_th,
            "Merge_Th": merge_th_str,
            "Mean_gNMI": float(np.mean(gnmis)),
            "Std_gNMI": float(np.std(gnmis)),
            "Mean_EQ": float(np.mean(eqs)),
            "Std_EQ": float(np.std(eqs)),
            "Mean_Qov_Unscaled": float(np.mean(qovs)),
            "Std_Qov_Unscaled": float(np.std(qovs)),
            "Mean_Qov_SLPA_Strict": float(np.mean(qov_slpas)),
            "Std_Qov_SLPA_Strict": float(np.std(qov_slpas)),
            "Mean_Time_Sec": float(np.mean(times)),
        })
        
    df_res = pd.DataFrame(summary_rows)
    out_csv = BENCH_DIR / "winning_params_20_seeds_summary.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"\n=================================================================")
    print(f" 20 SEEDS SUMMARY COMPLETED SUCCESSFULLY")
    print(f" Saved to: {out_csv}")
    print(f"=================================================================\n")
    print(df_res.to_string())

if __name__ == "__main__":
    main()
