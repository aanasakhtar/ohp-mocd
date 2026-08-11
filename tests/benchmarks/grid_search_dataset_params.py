"""
grid_search_dataset_params.py

Performs a comprehensive grid search across key overlap parameters:
  - init_overlap_prob: [0.05, 0.10, 0.15, 0.25, 0.40]
  - overlap_support_threshold: [0.15, 0.25, 0.35, 0.45]
  - overlap_removal_threshold: [0.08, 0.15, 0.25]
  - switch_margin: [0.05, 0.10]

Pre-loads graphs ONCE to avoid HTTP rate-limiting, then evaluates across multiple seeds.
"""

import io
import time
import urllib.request
import zipfile
import concurrent.futures
from pathlib import Path
import numpy as np
import networkx as nx
import pandas as pd
import pymocd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BENCH_DIR = REPO_ROOT / "tests" / "benchmarks"
HEADERS = {'User-Agent': 'Mozilla/5.0'}

# -----------------------------------------------------------------------------
# Metric Calculations
# -----------------------------------------------------------------------------
def nicosia_qov(G: nx.Graph, communities: list[set]) -> float:
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
                qov += f_val * A_uv - (k_u * k_v / two_m) * f_val
    return float(qov / two_m)

def shen_modularity_eq(G: nx.Graph, communities: list[set]) -> float:
    m = G.number_of_edges()
    if m == 0:
        return 0.0
    two_m = 2.0 * m
    deg = dict(G.degree())
    node_belong = {}
    for comm in communities:
        for u in comm:
            node_belong[u] = node_belong.get(u, 0) + 1
            
    eq = 0.0
    for comm in communities:
        for u in comm:
            for v in comm:
                A_uv = 1.0 if G.has_edge(u, v) else 0.0
                k_u = deg.get(u, 0)
                k_v = deg.get(v, 0)
                eq += (1.0 / (node_belong[u] * node_belong[v])) * (A_uv - (k_u * k_v) / two_m)
    return float(eq / two_m)

# -----------------------------------------------------------------------------
# Dataset Loaders
# -----------------------------------------------------------------------------
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
    try:
        G = nx.parse_gml(content, label='id' if 'id' in content else 'label')
    except nx.NetworkXError:
        lines = content.splitlines()
        clean_lines = []
        seen_edges = set()
        in_edge = False
        curr_edge_lines = []
        source = None
        target = None
        for line in lines:
            if line.strip().startswith('edge'):
                in_edge = True
                curr_edge_lines = [line]
                source = None
                target = None
            elif in_edge:
                curr_edge_lines.append(line)
                if 'source' in line:
                    source = line.strip().split()[-1]
                elif 'target' in line:
                    target = line.strip().split()[-1]
                elif line.strip() == ']':
                    in_edge = False
                    edge_key = tuple(sorted([source, target])) if source and target else None
                    if edge_key not in seen_edges:
                        if edge_key:
                            seen_edges.add(edge_key)
                        clean_lines.extend(curr_edge_lines)
            else:
                clean_lines.append(line)
        content_clean = '\n'.join(clean_lines)
        G = nx.parse_gml(content_clean, label='id' if 'id' in content_clean else 'label')
    return nx.Graph(G)

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

def load_celegans() -> nx.Graph:
    return load_newman_gml('celegansneural')

def load_email() -> nx.Graph:
    return load_newman_gml('email')

# Global cache populated once by main process
DATASETS_CACHE = {}

def prefetch_all_datasets():
    print("Pre-fetching all datasets to local cache...")
    loaders = {
        "Karate": load_karate,
        "Dolphins": load_dolphins,
        "Lesmis": load_lesmis,
        "Polbooks": load_polbooks,
        "Football": load_football,
        "Netscience": load_netscience,
        "Celegans": load_celegans,
        "Email": load_email,
    }
    for name, loader in loaders.items():
        try:
            DATASETS_CACHE[name] = loader()
            print(f" Loaded {name}: {DATASETS_CACHE[name].number_of_nodes()} nodes, {DATASETS_CACHE[name].number_of_edges()} edges")
        except Exception as e:
            print(f" Warning loading {name}: {e}")
            
# -----------------------------------------------------------------------------
# Single Trial Evaluation Task
# -----------------------------------------------------------------------------
def eval_param_trial(args: tuple) -> dict:
    net_name, init_strategy, p_init, supp_th, rem_th, margin, seed, edges_list = args
    
    G = nx.Graph()
    G.add_edges_from(edges_list)
    
    nodes = list(G.nodes())
    node_map = {n: i for i, n in enumerate(nodes)}
    rev_map = {i: n for i, n in enumerate(nodes)}
    H = nx.relabel_nodes(G, node_map, copy=True)
    
    dict_res = pymocd.ohpmocd(
        H,
        init_strategy=init_strategy,
        init_overlap_prob=p_init,
        overlap_support_threshold=supp_th,
        overlap_removal_threshold=rem_th,
        switch_margin=margin,
        seed=seed
    )
    
    comm_dict = {}
    for n_idx, comm_list in dict_res.items():
        orig_node = rev_map[n_idx]
        if isinstance(comm_list, (int, np.integer)):
            comm_list = [comm_list]
        for cid in comm_list:
            comm_dict.setdefault(cid, set()).add(orig_node)
    comms = [set(m) for m in comm_dict.values() if m]
    
    qov = nicosia_qov(G, comms)
    eq = shen_modularity_eq(G, comms)
    
    return {
        "Dataset": net_name,
        "init_strategy": init_strategy,
        "init_overlap_prob": p_init,
        "overlap_support_threshold": supp_th,
        "overlap_removal_threshold": rem_th,
        "switch_margin": margin,
        "seed": seed,
        "Qov": qov,
        "EQ": eq,
    }

def main():
    print("=================================================================")
    print(" STARTING COMPREHENSIVE DATASET-SPECIFIC PARAMETER GRID SEARCH ")
    print("=================================================================")
    
    prefetch_all_datasets()
    
    # Selected Representative Parameter Configurations
    param_configs = [
        # (init_overlap_prob, overlap_support_threshold, overlap_removal_threshold, switch_margin)
        (0.40, 0.15, 0.08, 0.05), # Baseline Default
        (0.35, 0.35, 0.25, 0.05), # Selective
        (0.20, 0.30, 0.20, 0.05),
        (0.15, 0.35, 0.25, 0.05),
        (0.10, 0.40, 0.30, 0.05),
        (0.10, 0.55, 0.35, 0.05), # Celegans optimal
        (0.25, 0.25, 0.15, 0.05),
        (0.30, 0.30, 0.20, 0.05),
        (0.15, 0.25, 0.15, 0.05),
        (0.20, 0.35, 0.25, 0.05),
    ]
    
    strategies = ["boundary_seeded", "crisp"]
    seeds = [42, 123, 999]
    
    tasks = []
    for d, G in DATASETS_CACHE.items():
        edges_list = list(G.edges())
        for strat in strategies:
            for (p_init, supp_th, rem_th, margin) in param_configs:
                for s in seeds:
                    tasks.append((d, strat, p_init, supp_th, rem_th, margin, s, edges_list))
                
    print(f"\nTotal grid search tasks: {len(tasks)} across {len(DATASETS_CACHE)} datasets...")
    
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=7) as executor:
        for res in executor.map(eval_param_trial, tasks):
            results.append(res)
            
    df_raw = pd.DataFrame(results)
    
    # Aggregate by dataset and param config
    group_cols = ["Dataset", "init_strategy", "init_overlap_prob", "overlap_support_threshold", "overlap_removal_threshold", "switch_margin"]
    df_agg = df_raw.groupby(group_cols).agg({"Qov": "mean", "EQ": "mean"}).reset_index()
    
    optimal_rows = []
    for d in DATASETS_CACHE.keys():
        sub = df_agg[df_agg["Dataset"] == d]
        if sub.empty: continue
        best_qov = sub.loc[sub["Qov"].idxmax()]
        best_eq = sub.loc[sub["EQ"].idxmax()]
        
        print(f"\n--- Optimal Parameters for {d} ---")
        print(f" Best Qov ({best_qov['Qov']:.4f}): strat={best_qov['init_strategy']}, init_p={best_qov['init_overlap_prob']}, supp_th={best_qov['overlap_support_threshold']}, rem_th={best_qov['overlap_removal_threshold']}, margin={best_qov['switch_margin']}")
        print(f" Best EQ  ({best_eq['EQ']:.4f}): strat={best_eq['init_strategy']}, init_p={best_eq['init_overlap_prob']}, supp_th={best_eq['overlap_support_threshold']}, rem_th={best_eq['overlap_removal_threshold']}, margin={best_eq['switch_margin']}")
        
        optimal_rows.append({
            "Dataset": d,
            "best_strategy": best_qov["init_strategy"],
            "best_init_overlap_prob": best_qov["init_overlap_prob"],
            "best_overlap_support_threshold": best_qov["overlap_support_threshold"],
            "best_overlap_removal_threshold": best_qov["overlap_removal_threshold"],
            "best_switch_margin": best_qov["switch_margin"],
            "Max_Qov": best_qov["Qov"],
            "Max_EQ": best_eq["EQ"],
        })
        
    df_opt = pd.DataFrame(optimal_rows)
    opt_csv = BENCH_DIR / "optimal_dataset_parameters.csv"
    df_opt.to_csv(opt_csv, index=False)
    print(f"\nSaved optimal dataset parameters to: {opt_csv}")

if __name__ == "__main__":
    main()
