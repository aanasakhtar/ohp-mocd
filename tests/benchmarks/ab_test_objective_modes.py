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

DATASET_URLS = {
    "Karate": "https://raw.githubusercontent.com/gephi/gephi/master/modules/Palettes/src/main/resources/org/gephi/layout/plugin/forceAtlas2/karate.gml",
    "Dolphins": "https://raw.githubusercontent.com/gephi/gephi/master/modules/Palettes/src/main/resources/org/gephi/layout/plugin/forceAtlas2/dolphins.gml",
    "Lesmis": "https://raw.githubusercontent.com/gephi/gephi/master/modules/Palettes/src/main/resources/org/gephi/layout/plugin/forceAtlas2/lesmiserables.gml",
    "Polbooks": "https://raw.githubusercontent.com/gephi/gephi/master/modules/Palettes/src/main/resources/org/gephi/layout/plugin/forceAtlas2/polbooks.gml",
    "Football": "https://raw.githubusercontent.com/gephi/gephi/master/modules/Palettes/src/main/resources/org/gephi/layout/plugin/forceAtlas2/football.gml",
    "Netscience": "https://raw.githubusercontent.com/gephi/gephi/master/modules/Palettes/src/main/resources/org/gephi/layout/plugin/forceAtlas2/netscience.gml",
}

DATASET_PARAMS = {
    "Karate": {"pop": 200, "gens": 200, "cross": 0.75, "mut": 0.30, "prob": 0.20},
    "Dolphins": {"pop": 300, "gens": 350, "cross": 0.90, "mut": 0.30, "prob": 0.10},
    "Lesmis": {"pop": 400, "gens": 400, "cross": 0.75, "mut": 0.30, "prob": 0.10},
    "Polbooks": {"pop": 300, "gens": 400, "cross": 0.75, "mut": 0.40, "prob": 0.10},
    "Football": {"pop": 350, "gens": 350, "cross": 0.90, "mut": 0.30, "prob": 0.10},
    "Netscience": {"pop": 400, "gens": 400, "cross": 0.90, "mut": 0.40, "prob": 0.05},
}

def load_network(name: str):
    data_dir = REPO_ROOT / "data" / "cache"
    data_dir.mkdir(parents=True, exist_ok=True)
    gml_file = data_dir / f"{name.lower()}.gml"
    
    if not gml_file.exists():
        url = DATASET_URLS[name]
        print(f"Downloading {name} from {url}...")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp:
            content = resp.read().decode('utf-8', errors='ignore')
            with open(gml_file, 'w', encoding='utf-8') as f:
                f.write(content)
                
    G = nx.read_gml(gml_file, label='id' if name in ['Karate', 'Dolphins', 'Lesmis', 'Polbooks', 'Football', 'Netscience'] else None)
    G = nx.convert_node_labels_to_integers(G)
    
    ground_truth = None
    if name == "Karate":
        gt_nodes = {}
        for n, d in G.nodes(data=True):
            val = d.get('value', d.get('Faction', d.get('club', 0)))
            gt_nodes[n] = [int(val) if str(val).isdigit() else (0 if val in ['Mr. Hi', 'Officer'] else 1)]
        ground_truth = gt_nodes
    elif name == "Football":
        gt_nodes = {}
        for n, d in G.nodes(data=True):
            gt_nodes[n] = [int(d.get('value', 0))]
        ground_truth = gt_nodes
    elif name == "Polbooks":
        gt_nodes = {}
        for n, d in G.nodes(data=True):
            val = d.get('value', 'n')
            mapping = {'c': 0, 'l': 1, 'n': 2}
            gt_nodes[n] = [mapping.get(val, 0)]
        ground_truth = gt_nodes
        
    return G, ground_truth

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
