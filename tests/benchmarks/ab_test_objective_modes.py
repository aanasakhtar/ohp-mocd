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
            gt_nodes[n] = [0 if club == 'Mr. Hi' else 1]
        return G, gt_nodes
    elif name == "Lesmis":
        G = nx.les_miserables_graph()
        return G, None
    elif name == "Dolphins":
        G = load_newman_gml('dolphins')
        pod2_names = {'Beak', 'CCL', 'Double', 'Fish', 'Five', 'Fork', 'Gallatin', 'Grin', 'Hook', 'Kringel', 'Oscar', 'PL', 'SN4', 'SN9', 'SN10', 'Scabs', 'Shakacle', 'SMN', 'Stripes', 'TR77', 'TSN83', 'TSN103', 'Zipfel'}
        gt_nodes = {}
        for n in G.nodes():
            lbl = G.nodes[n].get('label', str(n))
            gt_nodes[n] = [1 if lbl in pod2_names or str(n) in pod2_names else 0]
        return G, gt_nodes
    elif name == "Polbooks":
        G = load_newman_gml('polbooks')
        gt_nodes = {}
        mapping = {'c': 0, 'l': 1, 'n': 2}
        for n, d in G.nodes(data=True):
            val = d.get('value', 'n')
            gt_nodes[n] = [mapping.get(val, 0)]
        return G, gt_nodes
    elif name == "Football":
        G = load_newman_gml('football')
        gt_nodes = {}
        for n, d in G.nodes(data=True):
            val = d.get('value', 0)
            gt_nodes[n] = [int(val) if str(val).isdigit() else 0]
        return G, gt_nodes
    elif name == "Netscience":
        G = load_newman_gml('netscience')
        largest_cc = max(nx.connected_components(G), key=len)
        return G.subgraph(largest_cc).copy(), None
    else:
        raise ValueError(f"Unknown dataset {name}")

def compute_nicosia_qov(G: nx.Graph, partition: dict) -> float:
    m = G.number_of_edges()
    if m == 0: return 0.0
    
    node_weights = {}
    for n, comms in partition.items():
        if not comms: continue
        unif = 1.0 / len(comms)
        node_weights[n] = {c: unif for c in comms}
        
    def f_linear(r):
        return max(0.0, min(1.0, 60.0 * r - 30.0))
        
    q_ov = 0.0
    for u, v in G.edges():
        w_u = node_weights.get(u, {})
        w_v = node_weights.get(v, {})
        common_c = set(w_u.keys()).intersection(set(w_v.keys()))
        for c in common_c:
            beta_u_v_c = f_linear(w_u[c]) * f_linear(w_v[c])
            q_ov += beta_u_v_c
            
    # Null model
    d = dict(G.degree())
    for c in set(c for comms in partition.values() for c in comms):
        comm_nodes = [n for n, comms in partition.items() if c in comms]
        deg_sum = sum(d.get(n, 0) * node_weights[n][c] for n in comm_nodes)
        q_ov -= (deg_sum / (2.0 * m)) ** 2
        
    return q_ov / (2.0 * m)

def compute_shen_eq(G: nx.Graph, partition: dict) -> float:
    m = G.number_of_edges()
    if m == 0: return 0.0
    deg = dict(G.degree())
    
    comm_to_nodes = collections.defaultdict(list)
    for n, comms in partition.items():
        for c in comms:
            comm_to_nodes[c].append(n)
            
    eq = 0.0
    for c, nodes in comm_to_nodes.items():
        for i in range(len(nodes)):
            u = nodes[i]
            o_u = len(partition[u])
            d_u = deg.get(u, 0)
            for j in range(len(nodes)):
                v = nodes[j]
                o_v = len(partition[v])
                d_v = deg.get(v, 0)
                a_uv = 1.0 if G.has_edge(u, v) else 0.0
                eq += (1.0 / (o_u * o_v)) * (a_uv - (d_u * d_v) / (2.0 * m))
                
    return eq / (2.0 * m)

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
            
            # Metrics
            qov_a = compute_nicosia_qov(G, part_a)
            qov_b = compute_nicosia_qov(G, part_b)
            eq_a = compute_shen_eq(G, part_a)
            eq_b = compute_shen_eq(G, part_b)
            
            std_qovs.append(qov_a)
            std_eqs.append(eq_a)
            std_times.append(t_a)
            
            upg_qovs.append(qov_b)
            upg_eqs.append(eq_b)
            upg_times.append(t_b)
            
            if gt is not None:
                gnmi_a = onmi(part_a, gt)
                gnmi_b = onmi(part_b, gt)
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
