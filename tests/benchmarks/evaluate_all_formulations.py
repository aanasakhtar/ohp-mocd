"""
evaluate_all_formulations.py

Comprehensive 4-Way Head-to-Head Evaluation across all formulation candidates:
  1. Baseline: Standard OHP-MOCD (Modularity Decomp + Max-Q Selection)
  2. Formulation 1: Ratio Cut Multi-Objective (MCMOEA/Pizzuti Ratio Cut & Association + Knee Selection)
  3. Formulation 2: Parameter-Free Memetic Boundary LSO (Radicchi Weak Support Pruning)
  4. Formulation 2+3: Memetic Boundary LSO + Utopia Knee Selection

Evaluates under identical seeds and prints comprehensive side-by-side metrics:
  - Shen Extended Modularity (EQ)
  - Nicosia Overlapping Modularity (Qov)
  - Ground-Truth Recovery (gNMI)
  - Execution Time
"""

import sys
import time
import urllib.request
import zipfile
import io
import re
import collections
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pymocd
from evaluation.metrics import onmi, pairwise_f1

DATA_DIR = REPO_ROOT / "tests" / "benchmarks" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

DATASET_PARAMS = {
    "Karate": {"pop": 200, "gens": 200, "cross": 0.75, "mut": 0.30, "prob": 0.20},
    "Dolphins": {"pop": 300, "gens": 350, "cross": 0.90, "mut": 0.30, "prob": 0.10},
    "Lesmis": {"pop": 400, "gens": 400, "cross": 0.75, "mut": 0.30, "prob": 0.10},
    "Polbooks": {"pop": 300, "gens": 400, "cross": 0.75, "mut": 0.40, "prob": 0.10},
    "Football": {"pop": 350, "gens": 350, "cross": 0.90, "mut": 0.30, "prob": 0.10},
    "Netscience": {"pop": 400, "gens": 400, "cross": 0.90, "mut": 0.40, "prob": 0.05},
}

def load_newman_gml(zip_name: str) -> nx.Graph:
    local_gml = DATA_DIR / f"{zip_name}.gml"
    if local_gml.exists():
        content = local_gml.read_text(encoding='utf-8', errors='ignore')
    else:
        url = f'http://www-personal.umich.edu/~mejn/netdata/{zip_name}.zip'
        req = urllib.request.Request(url, headers=HEADERS)
        res = urllib.request.urlopen(req, timeout=30)
        z = zipfile.ZipFile(io.BytesIO(res.read()))
        gml_name = [f for f in z.namelist() if f.endswith('.gml')][0]
        content = z.read(gml_name).decode('utf-8', errors='ignore')
        local_gml.write_text(content, encoding='utf-8')
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

def load_network(name: str):
    if name == "Karate":
        G = nx.karate_club_graph()
        gt_nodes = {}
        for n, d in G.nodes(data=True):
            club = d.get('club', 'Mr. Hi')
            gt_nodes.setdefault(0 if club == 'Mr. Hi' else 1, set()).add(n)
        return G, [frozenset(c) for c in gt_nodes.values()]
    elif name == "Lesmis":
        G = nx.les_miserables_graph()
        return G, None
    elif name == "Dolphins":
        G = load_newman_gml('dolphins')
        pod2_names = {'Beak', 'CCL', 'Double', 'Fish', 'Five', 'Fork', 'Gallatin', 'Grin', 'Hook', 'Kringel', 'Oscar', 'PL', 'SN4', 'SN9', 'SN10', 'Scabs', 'Shakacle', 'SMN', 'Stripes', 'TR77', 'TSN83', 'TSN103', 'Zipfel'}
        pod2 = set([n for n in G.nodes() if G.nodes[n].get('label', str(n)) in pod2_names or str(n) in pod2_names])
        pod1 = set([n for n in G.nodes() if n not in pod2])
        return G, [frozenset(pod1), frozenset(pod2)]
    elif name == "Polbooks":
        G = load_newman_gml('polbooks')
        gt_nodes = {}
        for n, d in G.nodes(data=True):
            val = d.get('value', 'n')
            gt_nodes.setdefault(val, set()).add(n)
        return G, [frozenset(c) for c in gt_nodes.values()]
    elif name == "Football":
        G = load_newman_gml('football')
        gt_nodes = {}
        for n, d in G.nodes(data=True):
            val = d.get('value', 0)
            gt_nodes.setdefault(val, set()).add(n)
        return G, [frozenset(c) for c in gt_nodes.values()]
    elif name == "Netscience":
        G = load_newman_gml('netscience')
        largest_cc = max(nx.connected_components(G), key=len)
        return G.subgraph(largest_cc).copy(), None
    else:
        raise ValueError(f"Unknown dataset {name}")

def dict_to_communities(part: dict) -> list[frozenset]:
    comm_map = collections.defaultdict(set)
    for n, comms in part.items():
        for c in comms:
            comm_map[c].add(n)
    return [frozenset(nodes) for nodes in comm_map.values() if nodes]

def compute_nicosia_qov(G: nx.Graph, communities: list[frozenset]) -> float:
    m = G.number_of_edges()
    if m == 0 or not communities: return 0.0
    two_m = 2.0 * m
    deg = dict(G.degree())
    N = G.number_of_nodes()
    
    node_belong = {}
    for comm in communities:
        for u in comm:
            node_belong[u] = node_belong.get(u, 0) + 1
            
    qov = 0.0
    for comm in communities:
        sum_f_c = sum(max(0.0, min(1.0, 60.0 * (1.0 / node_belong[u]) - 30.0)) for u in comm)
        avg_f_c = sum_f_c / float(N)
        
        for u in comm:
            r_u = 1.0 / node_belong[u]
            f_u = max(0.0, min(1.0, 60.0 * r_u - 30.0))
            l_out = f_u * avg_f_c
            
            for v in comm:
                r_v = 1.0 / node_belong[v]
                f_v = max(0.0, min(1.0, 60.0 * r_v - 30.0))
                l_in = f_v * avg_f_c
                
                l_uv = f_u * f_v
                s_uv = l_out * l_in
                A_uv = 1.0 if G.has_edge(u, v) else 0.0
                k_u = deg.get(u, 0)
                k_v = deg.get(v, 0)
                qov += l_uv * A_uv - s_uv * ((k_u * k_v) / two_m)
    return float(qov / two_m)

def compute_shen_eq(G: nx.Graph, communities: list[frozenset]) -> float:
    m = G.number_of_edges()
    if m == 0 or not communities: return 0.0
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

def run_evaluation(datasets=["Karate", "Dolphins", "Polbooks", "Football"], seeds=[42, 43, 44, 45, 46]):
    print("=" * 80)
    print(" MULTI-FORMULATION SYSTEMATIC BENCHMARK")
    print(" 1. Baseline: Standard OHP-MOCD (Modularity Decomp + Max-Q)")
    print(" 2. Form 1: Ratio Cut Multi-Objective (Pizzuti/MCMOEA + Knee Selection)")
    print(" 3. Form 2: Parameter-Free Memetic Boundary LSO (Radicchi Pruning)")
    print(" 4. Form 2+3: Memetic LSO + Utopia Knee Selection")
    print(f" Seeds: {seeds} | Datasets: {datasets}")
    print("=" * 80)
    
    variants = [
        ("Baseline (Standard)", {"objective_mode": "standard", "selection_mode": "max_q", "enable_lso": False}),
        ("Form 1 (Ratio Cut + Knee)", {"objective_mode": "ratio_cut", "selection_mode": "knee", "enable_lso": False}),
        ("Form 2 (Memetic LSO)", {"objective_mode": "standard", "selection_mode": "max_q", "enable_lso": True}),
        ("Form 2+3 (LSO + Knee)", {"objective_mode": "standard", "selection_mode": "knee", "enable_lso": True}),
    ]
    
    master_records = []
    
    for name in datasets:
        G, gt = load_network(name)
        params = DATASET_PARAMS[name]
        print(f"\n---> Dataset: {name} (N={G.number_of_nodes()}, E={G.number_of_edges()})")
        print(f"     Params: pop={params['pop']}, gens={params['gens']}, cross={params['cross']}, mut={params['mut']}, prob={params['prob']}")
        
        for v_name, v_opts in variants:
            eqs, qovs, gnmis, times = [], [], [], []
            for s in seeds:
                t0 = time.time()
                part = pymocd.ohpmocd(
                    G,
                    pop_size=params["pop"],
                    num_gens=params["gens"],
                    cross_rate=params["cross"],
                    mut_rate=params["mut"],
                    init_strategy="boundary_seeded",
                    init_overlap_prob=params["prob"],
                    objective_mode=v_opts["objective_mode"],
                    selection_mode=v_opts["selection_mode"],
                    enable_lso=v_opts["enable_lso"],
                    seed=s
                )
                t_elapsed = time.time() - t0
                
                comms = dict_to_communities(part)
                eq = compute_shen_eq(G, comms)
                qov = compute_nicosia_qov(G, comms)
                eqs.append(eq)
                qovs.append(qov)
                times.append(t_elapsed)
                
                if gt is not None:
                    gnmi = onmi(comms, gt)
                    gnmis.append(gnmi)
                    
            mean_eq = np.mean(eqs)
            mean_qov = np.mean(qovs)
            mean_gnmi = np.mean(gnmis) if gnmis else np.nan
            mean_t = np.mean(times)
            
            gnmi_str = f"| gNMI: {mean_gnmi:.4f}" if gnmis else ""
            print(f"  [{v_name:<26}] Shen EQ: {mean_eq:.4f} | Nicosia Qov: {mean_qov:.4f} {gnmi_str} | Time: {mean_t:.2f}s")
            
            master_records.append({
                "Dataset": name,
                "Variant": v_name,
                "Shen_EQ": mean_eq,
                "Nicosia_Qov": mean_qov,
                "gNMI": mean_gnmi,
                "Time_Sec": mean_t
            })
            
    df = pd.DataFrame(master_records)
    print("\n" + "=" * 80)
    print(" SYSTEMATIC FORMULATION COMPARISON TABLE")
    print("=" * 80)
    print(df.to_string(index=False))
    
    out_csv = REPO_ROOT / "tests" / "benchmarks" / "formulation_comparison_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved summary table to: {out_csv}")

if __name__ == "__main__":
    run_evaluation()
