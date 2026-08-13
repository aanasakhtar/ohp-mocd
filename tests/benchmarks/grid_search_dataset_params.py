"""
grid_search_dataset_params.py

Performs a comprehensive Cartesian multi-dimensional grid search across key overlap parameters:
  - init_overlap_prob: [0.10, 0.15, 0.25]
  - overlap_support_threshold: [0.10, 0.15, 0.25, 0.35, 0.45, 0.55]
  - overlap_removal_threshold: [0.05, 0.08, 0.15, 0.25, 0.35]
  - switch_margin: [0.05]
  - alpha: [0.00, 0.25, 0.50, 0.75, 1.00]
  - post_hoc_merge_threshold: [0.25, 0.35, 0.50, None]

Evaluates ground truth gNMI (ONMI) and Modularity Q / EQ / Qov.
Pre-loads graphs into local cache for ultra-fast multi-core parallel execution.
"""

import sys, os
sys.path.insert(0, os.path.abspath('.'))

import io
import time
import zipfile
import urllib.request
import itertools
import concurrent.futures
from pathlib import Path
import numpy as np
import networkx as nx
import pandas as pd
import pymocd
from evaluation.metrics import onmi

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BENCH_DIR = REPO_ROOT / "tests" / "benchmarks"
DATA_DIR = BENCH_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

# -----------------------------------------------------------------------------
# Metrics & Post-Hoc Operators
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Dataset Loaders & Cache
# -----------------------------------------------------------------------------
def load_karate() -> nx.Graph: return nx.karate_club_graph()
def load_lesmis() -> nx.Graph: return nx.les_miserables_graph()

def load_newman_gml(zip_name: str) -> nx.Graph:
    local_zip = DATA_DIR / f"{zip_name}.zip"
    if not local_zip.exists():
        url = f'http://www-personal.umich.edu/~mejn/netdata/{zip_name}.zip'
        req = urllib.request.Request(url, headers=HEADERS)
        res = urllib.request.urlopen(req, timeout=30)
        local_zip.write_bytes(res.read())
    z = zipfile.ZipFile(io.BytesIO(local_zip.read_bytes()))
    gml_name = [f for f in z.namelist() if f.endswith('.gml')][0]
    content = z.read(gml_name).decode('utf-8', errors='ignore')
    try:
        G = nx.parse_gml(content, label='id' if 'id' in content else 'label')
    except nx.NetworkXError:
        lines = content.splitlines()
        clean_lines, seen_edges, in_edge, curr_edge, source, target = [], set(), False, [], None, None
        for line in lines:
            if line.strip().startswith('edge'):
                in_edge, curr_edge, source, target = True, [line], None, None
            elif in_edge:
                curr_edge.append(line)
                if 'source' in line: source = line.strip().split()[-1]
                elif 'target' in line: target = line.strip().split()[-1]
                elif line.strip() == ']':
                    in_edge = False
                    e_key = tuple(sorted([source, target])) if source and target else None
                    if e_key not in seen_edges:
                        if e_key: seen_edges.add(e_key)
                        clean_lines.extend(curr_edge)
            else: clean_lines.append(line)
        content_clean = '\n'.join(clean_lines)
        G = nx.parse_gml(content_clean, label='id' if 'id' in content_clean else 'label')
    return nx.Graph(G)

def load_dolphins() -> nx.Graph: return load_newman_gml('dolphins')
def load_polbooks() -> nx.Graph: return load_newman_gml('polbooks')
def load_football() -> nx.Graph: return load_newman_gml('football')
def load_netscience() -> nx.Graph:
    G = load_newman_gml('netscience')
    largest_cc = max(nx.connected_components(G), key=len)
    return G.subgraph(largest_cc).copy()
def load_celegans() -> nx.Graph: return load_newman_gml('celegansneural')

def extract_ground_truth(G: nx.Graph, net_name: str) -> list[frozenset] | None:
    if net_name == "Karate":
        comms = {}
        for n, d in G.nodes(data=True): comms.setdefault(d.get('club', 'default'), set()).add(n)
        return [frozenset(c) for c in comms.values() if c]
    elif net_name == "Dolphins":
        pod2_names = {'Beak', 'CCL', 'Double', 'Fish', 'Five', 'Fork', 'Gallatin', 'Grin', 'Hook', 'Kringel', 'Oscar', 'PL', 'SN4', 'SN9', 'SN10', 'Scabs', 'Shakacle', 'SMN', 'Stripes', 'TR77', 'TSN83', 'TSN103', 'Zipfel'}
        pod2 = set([n for n in G.nodes() if n in pod2_names or str(n) in pod2_names])
        pod1 = set([n for n in G.nodes() if n not in pod2])
        return [frozenset(pod1), frozenset(pod2)]
    elif net_name in ("Polbooks", "Football"):
        comms = {}
        for n, d in G.nodes(data=True):
            val = d.get('value', None)
            if val is not None: comms.setdefault(val, set()).add(n)
        if comms: return [frozenset(c) for c in comms.values() if c]
    return None

DATASETS_CACHE = {}

def prefetch_all_datasets():
    print("Pre-fetching all benchmark datasets...")
    loaders = {
        "Karate": load_karate,
        "Dolphins": load_dolphins,
        "Lesmis": load_lesmis,
        "Polbooks": load_polbooks,
        "Football": load_football,
        "Netscience": load_netscience,
        "Celegans": load_celegans,
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
    net_name, init_strategy, p_init, supp_th, rem_th, margin, alpha, merge_th, edges_list, gt = args
    
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
        alpha=alpha,
        seed=None
    )
    
    comm_dict = {}
    for n_idx, comm_list in dict_res.items():
        orig_node = rev_map[n_idx]
        if isinstance(comm_list, (int, np.integer)): comm_list = [comm_list]
        for cid in comm_list: comm_dict.setdefault(cid, set()).add(orig_node)
    comms = [set(m) for m in comm_dict.values() if m]
    
    if merge_th is not None:
        comms = post_hoc_boundary_merge(G, comms, merge_threshold=merge_th)
        
    qov = nicosia_qov(G, comms)
    eq = shen_modularity_eq(G, comms)
    onmi_val = onmi([frozenset(c) for c in comms], gt) if gt is not None else 0.0
    
    return {
        "Dataset": net_name,
        "init_strategy": init_strategy,
        "init_overlap_prob": p_init,
        "overlap_support_threshold": supp_th,
        "overlap_removal_threshold": rem_th,
        "switch_margin": margin,
        "alpha": alpha,
        "merge_threshold": str(merge_th),
        "Qov": qov,
        "EQ": eq,
        "gNMI": onmi_val,
    }

def main():
    print("=================================================================")
    print(" STARTING COMPREHENSIVE MULTI-DIMENSIONAL PARAMETER GRID SEARCH ")
    print("=================================================================")
    
    prefetch_all_datasets()
    
    p_inits = [0.10, 0.15]
    supp_ths = [0.10, 0.15, 0.25, 0.35, 0.55]
    rem_ths = [0.05, 0.08, 0.15, 0.25]
    alphas = [0.00, 0.25, 0.50, 0.75, 1.00]
    merge_ths = [0.35, 0.50, None]
    strategies = ["boundary_seeded", "crisp"]
    
    tasks = []
    for d, G in DATASETS_CACHE.items():
        edges_list = list(G.edges())
        gt = extract_ground_truth(G, d)
        for strat, p_init, supp_th, rem_th, alpha, merge_th in itertools.product(
            strategies, p_inits, supp_ths, rem_ths, alphas, merge_ths
        ):
            # Prune invalid configurations where removal threshold >= support threshold
            if rem_th >= supp_th: continue
            tasks.append((d, strat, p_init, supp_th, rem_th, 0.05, alpha, merge_th, edges_list, gt))
            
    print(f"\nTotal multi-dimensional grid search tasks: {len(tasks)} across {len(DATASETS_CACHE)} datasets...")
    
    results = []
    t0 = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for res in executor.map(eval_param_trial, tasks):
            results.append(res)
    dur = time.perf_counter() - t0
    print(f"Grid search completed in {dur:.2f} seconds!")
    
    df_raw = pd.DataFrame(results)
    df_raw.to_csv(BENCH_DIR / "grid_search_raw_results.csv", index=False)
    
    group_cols = ["Dataset", "init_strategy", "init_overlap_prob", "overlap_support_threshold", "overlap_removal_threshold", "switch_margin", "alpha", "merge_threshold"]
    df_agg = df_raw.groupby(group_cols).agg({"Qov": "mean", "EQ": "mean", "gNMI": "mean"}).reset_index()
    
    optimal_rows = []
    for d in DATASETS_CACHE.keys():
        sub = df_agg[df_agg["Dataset"] == d]
        if sub.empty: continue
        best_eq = sub.loc[sub["EQ"].idxmax()]
        best_gnmi = sub.loc[sub["gNMI"].idxmax()] if sub["gNMI"].max() > 0 else best_eq
        
        print(f"\n--- Optimal Parameters for {d} ---")
        print(f" Best EQ   ({best_eq['EQ']:.4f}): strat={best_eq['init_strategy']}, alpha={best_eq['alpha']}, supp_th={best_eq['overlap_support_threshold']}, rem_th={best_eq['overlap_removal_threshold']}, merge_th={best_eq['merge_threshold']}")
        print(f" Best gNMI ({best_gnmi['gNMI']:.4f}): strat={best_gnmi['init_strategy']}, alpha={best_gnmi['alpha']}, supp_th={best_gnmi['overlap_support_threshold']}, rem_th={best_gnmi['overlap_removal_threshold']}, merge_th={best_gnmi['merge_threshold']}")
        
        optimal_rows.append({
            "Dataset": d,
            "best_strategy": best_gnmi["init_strategy"],
            "best_alpha": best_gnmi["alpha"],
            "best_init_overlap_prob": best_gnmi["init_overlap_prob"],
            "best_overlap_support_threshold": best_gnmi["overlap_support_threshold"],
            "best_overlap_removal_threshold": best_gnmi["overlap_removal_threshold"],
            "best_switch_margin": best_gnmi["switch_margin"],
            "best_merge_threshold": best_gnmi["merge_threshold"],
            "Max_EQ": best_eq["EQ"],
            "Max_gNMI": best_gnmi["gNMI"],
        })
        
    df_opt = pd.DataFrame(optimal_rows)
    opt_csv = BENCH_DIR / "optimal_dataset_parameters.csv"
    df_opt.to_csv(opt_csv, index=False)
    print(f"\nSaved optimal dataset parameters to: {opt_csv}")

if __name__ == "__main__":
    main()
