"""
ab_test_objective_modes.py

Unbiased Head-to-Head A/B Empirical Evaluation:
  Variant A: Standard OHP-MOCD (Uniform r_uc + Count f3)
  Variant B: Upgraded OHP-MOCD (Direction 1 Structural Overlap Cohesion + Direction 2 Node Intimacy)

Evaluates on identical random seeds across benchmark datasets and prints side-by-side delta metrics.
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

def run_ab_comparison(datasets=["Karate", "Dolphins", "Polbooks", "Football"], seeds=[42, 43, 44, 45, 46]):
    print("=" * 80)
    print(" UNBIASED A/B TEST: STANDARD OBJECTIVES vs COHESION-INTIMACY UPGRADED OBJECTIVES")
    print(f" Seeds: {seeds} | Datasets: {datasets}")
    print("=" * 80)
    
    rows = []
    
    for name in datasets:
        G, gt = load_network(name)
        params = DATASET_PARAMS[name]
        print(f"\n---> Evaluating on: {name} (N={G.number_of_nodes()}, E={G.number_of_edges()})")
        print(f"     Params: pop={params['pop']}, gens={params['gens']}, cross={params['cross']}, mut={params['mut']}, prob={params['prob']}")
        
        std_qovs, std_eqs, std_gnmis, std_times = [], [], [], []
        upg_qovs, upg_eqs, upg_gnmis, upg_times = [], [], [], []
        
        for s in seeds:
            # Variant A: Standard Objectives
            t0 = time.time()
            part_a = pymocd.ohpmocd(
                G,
                pop_size=params["pop"],
                num_gens=params["gens"],
                cross_rate=params["cross"],
                mut_rate=params["mut"],
                init_strategy="boundary_seeded",
                init_overlap_prob=params["prob"],
                objective_mode="standard",
                seed=s
            )
            t_a = time.time() - t0
            
            # Variant B: Cohesion-Intimacy Objectives
            t0 = time.time()
            part_b = pymocd.ohpmocd(
                G,
                pop_size=params["pop"],
                num_gens=params["gens"],
                cross_rate=params["cross"],
                mut_rate=params["mut"],
                init_strategy="boundary_seeded",
                init_overlap_prob=params["prob"],
                objective_mode="cohesion_intimacy",
                seed=s
            )
            t_b = time.time() - t0
            
            comm_a = dict_to_communities(part_a)
            comm_b = dict_to_communities(part_b)
            
            # Metrics
            qov_a = compute_nicosia_qov(G, comm_a)
            qov_b = compute_nicosia_qov(G, comm_b)
            eq_a = compute_shen_eq(G, comm_a)
            eq_b = compute_shen_eq(G, comm_b)
            
            std_qovs.append(qov_a)
            std_eqs.append(eq_a)
            std_times.append(t_a)
            
            upg_qovs.append(qov_b)
            upg_eqs.append(eq_b)
            upg_times.append(t_b)
            
            if gt is not None:
                gnmi_a = onmi(comm_a, gt)
                gnmi_b = onmi(comm_b, gt)
                std_gnmis.append(gnmi_a)
                upg_gnmis.append(gnmi_b)
                
        mean_qov_a, mean_qov_b = np.mean(std_qovs), np.mean(upg_qovs)
        mean_eq_a, mean_eq_b = np.mean(std_eqs), np.mean(upg_eqs)
        mean_t_a, mean_t_b = np.mean(std_times), np.mean(upg_times)
        
        print(f"  • Standard (A)         -> Shen EQ: {mean_eq_a:.4f} | Qov: {mean_qov_a:.4f} | Time: {mean_t_a:.2f}s")
        print(f"  • Cohesion-Intimacy (B)-> Shen EQ: {mean_eq_b:.4f} | Qov: {mean_qov_b:.4f} | Time: {mean_t_b:.2f}s")
        if gt is not None:
            mean_g_a, mean_g_b = np.mean(std_gnmis), np.mean(upg_gnmis)
            print(f"    -> Ground Truth gNMI: Standard={mean_g_a:.4f} vs Upgraded={mean_g_b:.4f} (Delta: {mean_g_b - mean_g_a:+.4f})")
            
        rows.append({
            "Dataset": name,
            "Standard_EQ": mean_eq_a,
            "Upgraded_EQ": mean_eq_b,
            "EQ_Delta": mean_eq_b - mean_eq_a,
            "Standard_Qov": mean_qov_a,
            "Upgraded_Qov": mean_qov_b,
            "Qov_Delta": mean_qov_b - mean_qov_a,
            "Standard_Time": mean_t_a,
            "Upgraded_Time": mean_t_b,
        })
        
    df = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print(" A/B OBJECTIVES COMPARISON SUMMARY")
    print("=" * 80)
    print(df.to_string(index=False))
    
    out_csv = REPO_ROOT / "tests" / "benchmarks" / "ab_test_objectives_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved summary to: {out_csv}")

if __name__ == "__main__":
    run_ab_comparison()
