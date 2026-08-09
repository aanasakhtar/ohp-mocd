"""
paper_baselines_benchmark.py

Comprehensive evaluation benchmark comparing OHP-MOCD against reported baseline metric scores
from four published research papers:
1. SLPA (Xie & Szymanski, 2011) - docs/1109.5720v3.pdf
2. MCMOEA (IEEE TEVC, 2016) - docs/A_Maximal_Clique_Based_Multiobjective_Evolutionary.pdf
3. Shang et al. / FCCNI (Applied Soft Computing, 2024) - docs/66797d469912c.pdf
4. Cetin & Amrahov (Kybernetika, 2022) - docs/kybernetika_paper.pdf
"""

import os
import sys
import time
import zipfile
import io
import urllib.request
import re
import numpy as np
import pandas as pd
import networkx as nx
import pymocd

# Ensure evaluation metrics can be imported
sys.path.insert(0, r'D:\Research\ohp-mocd')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.metrics import onmi, pairwise_f1

# -----------------------------------------------------------------------------
# Metric Definitions
# -----------------------------------------------------------------------------

def nicosia_qov(G: nx.Graph, communities: list[set]) -> float:
    """Nicosia et al. (2009) Overlapping Modularity Qov."""
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

def shen_extended_modularity_eq(G: nx.Graph, communities: list[set]) -> float:
    """Shen et al. (2009) Extended Modularity EQ."""
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

def overlapping_coverage(G: nx.Graph, communities: list[set]) -> float:
    """Overlapping Coverage: ratio of edges connecting nodes sharing at least one community."""
    m = G.number_of_edges()
    if m == 0:
        return 1.0
    node_comms = {}
    for cid, comm in enumerate(communities):
        for u in comm:
            node_comms.setdefault(u, set()).add(cid)
            
    intra_edges = 0
    for u, v in G.edges():
        if node_comms.get(u, set()) & node_comms.get(v, set()):
            intra_edges += 1
    return float(intra_edges / m)

# -----------------------------------------------------------------------------
# Dataset Loaders
# -----------------------------------------------------------------------------

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

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
    return nx.parse_gml(content, label='id' if 'id' in content else 'label')

def load_dolphins() -> nx.Graph:
    return load_newman_gml('dolphins')

def load_polbooks() -> nx.Graph:
    return load_newman_gml('polbooks')

def load_football() -> nx.Graph:
    return load_newman_gml('football')

def load_netscience() -> nx.Graph:
    G = load_newman_gml('netscience')
    # Use largest connected component
    largest_cc = max(nx.connected_components(G), key=len)
    return G.subgraph(largest_cc).copy()

def load_celegans() -> nx.Graph:
    url = 'http://www-personal.umich.edu/~mejn/netdata/celegansneural.zip'
    req = urllib.request.Request(url, headers=HEADERS)
    res = urllib.request.urlopen(req)
    z = zipfile.ZipFile(io.BytesIO(res.read()))
    gml_name = [f for f in z.namelist() if f.endswith('.gml')][0]
    content = z.read(gml_name).decode('utf-8', errors='ignore')
    edges = []
    for match in re.finditer(r'edge\s*\[\s*source\s+(\d+)\s+target\s+(\d+)', content):
        u, v = int(match.group(1)), int(match.group(2))
        edges.append((u, v))
    return nx.Graph(edges)

def load_email() -> nx.Graph:
    url = 'https://snap.stanford.edu/data/email-Eu-core.txt.gz'
    req = urllib.request.Request(url, headers=HEADERS)
    content = urllib.request.urlopen(req).read()
    import gzip
    with gzip.GzipFile(fileobj=io.BytesIO(content)) as gz:
        lines = gz.read().decode('utf-8').splitlines()
    edges = []
    for line in lines:
        if line.startswith('#'): continue
        parts = line.strip().split()
        if len(parts) >= 2:
            edges.append((int(parts[0]), int(parts[1])))
    return nx.Graph(edges)

# -----------------------------------------------------------------------------
# Algorithm Execution & Evaluation
# -----------------------------------------------------------------------------

def run_ohpmocd_on_graph(G: nx.Graph, pop_size: int = 100, num_gens: int = 50, max_memberships: int = 2, seed: int = 42) -> list[set]:
    """Relabels nodes to 0..N-1, runs OHP-MOCD, and maps communities back to original nodes."""
    nodes = list(G.nodes())
    node_map = {n: i for i, n in enumerate(nodes)}
    rev_map = {i: n for i, n in enumerate(nodes)}
    
    H = nx.relabel_nodes(G, node_map, copy=True)
    model = pymocd.OhpMocd(
        H,
        pop_size=pop_size,
        num_gens=num_gens,
        max_memberships_per_node=max_memberships,
        seed=seed
    )
    result_dict = model.run()
    
    # Invert node -> list of comms mapping to list of sets
    comm_dict = {}
    for node_idx, comm_list in result_dict.items():
        orig_node = rev_map[node_idx]
        for cid in comm_list:
            comm_dict.setdefault(cid, set()).add(orig_node)
            
    return list(comm_dict.values())

def benchmark_real_world_networks():
    print("=================================================================")
    print(" RUNNING OHP-MOCD ON REAL-WORLD NETWORKS ACROSS 4 PAPERS ")
    print("=================================================================")
    
    dataset_loaders = [
        ("Karate", load_karate),
        ("Dolphins", load_dolphins),
        ("Lesmis", load_lesmis),
        ("Polbooks", load_polbooks),
        ("Football", load_football),
        ("Netscience", load_netscience),
        ("Celegans", load_celegans),
        ("Email", load_email),
    ]
    
    results = []
    for name, loader in dataset_loaders:
        print(f"\nEvaluating dataset: {name}...")
        try:
            G = loader()
            print(f"  Graph: N={G.number_of_nodes()}, E={G.number_of_edges()}")
            
            # Run 5 independent runs for robust mean and std
            qov_list, eq_list, cov_list, num_comms_list, times = [], [], [], [], []
            for seed in range(5):
                t0 = time.time()
                comms = run_ohpmocd_on_graph(G, pop_size=100, num_gens=50, max_memberships=2, seed=seed+42)
                t1 = time.time()
                
                qov = nicosia_qov(G, comms)
                eq = shen_extended_modularity_eq(G, comms)
                cov = overlapping_coverage(G, comms)
                
                qov_list.append(qov)
                eq_list.append(eq)
                cov_list.append(cov)
                num_comms_list.append(len(comms))
                times.append(t1 - t0)
                
            results.append({
                "Dataset": name,
                "Nodes": G.number_of_nodes(),
                "Edges": G.number_of_edges(),
                "Comms_Avg": np.mean(num_comms_list),
                "Qov_Mean": np.mean(qov_list),
                "Qov_Std": np.std(qov_list),
                "EQ_Mean": np.mean(eq_list),
                "EQ_Std": np.std(eq_list),
                "Coverage_Mean": np.mean(cov_list),
                "Coverage_Std": np.std(cov_list),
                "Time_Avg_Sec": np.mean(times)
            })
            print(f"  Results: Qov={np.mean(qov_list):.4f} +/- {np.std(qov_list):.4f}, EQ={np.mean(eq_list):.4f}, Coverage={np.mean(cov_list):.4f}")
        except Exception as e:
            print(f"  Error running {name}: {e}")
            
    return pd.DataFrame(results)

if __name__ == "__main__":
    df_real = benchmark_real_world_networks()
    out_dir = r"D:\Research\ohp-mocd\tests\benchmarks"
    df_real.to_csv(os.path.join(out_dir, "ohpmocd_paper_baselines_real.csv"), index=False)
    print("\nBenchmark Completed Successfully. Results saved.")
