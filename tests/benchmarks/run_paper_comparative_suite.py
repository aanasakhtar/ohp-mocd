"""
run_paper_comparative_suite.py

Parallelized, Strict Literature-Reported Benchmark Suite comparing OHP-MOCD (Boundary-Seeded & Crisp)
against the EXACT numbers published in 5 baseline papers:
  Paper 1: SLPA (Xie & Szymanski, IEEE TKDE 2011/2012)
           -> Metric: Nicosia Qov across Karate, Dolphins, Lesmis, Polbooks, Football, Netscience, Celegans, Email.
  Paper 2: MCMOEA (Wen et al., IEEE TEVC 2016)
           -> Metric: Nicosia Qov across Word Association Small 1 & 2 and Scientific Collaborators (Netscience).
  Paper 3: Çetin & Amrahov (Kybernetika 2022)
           -> Metrics: Shen Extended Modularity (EQ) & Overlapping Coverage across Karate, Dolphins, Lesmis, Polbooks.
  Paper 4: LPAM (Ponomarenko et al., PLOS ONE 2021)
           -> Metrics: Overlapping NMI (ONMI) & F1 Score across Karate, Football, Polbooks.
  Paper 5: NOCD (Shchur & Günnemann, KDD / ICLR 2019)
           -> Metric: Overlapping NMI (ONMI) across Facebook Social Circles (Ego 348, 414, 686, 698, 1684, 1912).
"""

import os
import sys
import time
import argparse
import zipfile
import gzip
import tarfile
import io
import urllib.request
import collections
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import concurrent.futures

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pymocd
from evaluation.metrics import onmi, pairwise_f1
from tests.benchmarks.utils.merge import post_hoc_boundary_merge

DATA_DIR = REPO_ROOT / "data"
BENCH_DIR = REPO_ROOT / "tests" / "benchmarks"
PLOTS_DIR = BENCH_DIR / "plots" / "strict_paper_comparisons"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "font.family": "sans-serif",
})

# -----------------------------------------------------------------------------
# Metric Definitions (Strictly Matching Paper Specifications)
# -----------------------------------------------------------------------------

def nicosia_qov_slpa(G: nx.Graph, communities: list[set]) -> float:
    """Nicosia et al. (2009) Overlapping Modularity Qov with Xie & Szymanski (2011) Section IV-D linear clamp:
    f(r) = max(0.0, min(1.0, 60.0 * r - 30.0)).
    Used in SLPA (Xie & Szymanski 2011) & MCMOEA (Wen et al. 2016).
    """
    m = G.number_of_edges()
    if m == 0 or not communities:
        return 0.0
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

def nicosia_qov_unscaled(G: nx.Graph, communities: list[set]) -> float:
    """Nicosia et al. (2009) Unscaled Overlapping Modularity Qov (Plain linear belongingness r_u = 1 / |M(u)|)."""
    m = G.number_of_edges()
    if m == 0 or not communities:
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
            r_u = 1.0 / node_belong[u]
            for v in comm:
                r_v = 1.0 / node_belong[v]
                f_val = r_u * r_v
                A_uv = 1.0 if G.has_edge(u, v) else 0.0
                k_u = deg.get(u, 0)
                k_v = deg.get(v, 0)
                qov += f_val * A_uv - (k_u * k_v / two_m) * f_val
    return float(qov / two_m)

nicosia_qov = nicosia_qov_slpa

def shen_modularity_eq(G: nx.Graph, communities: list[set]) -> float:
    """Shen et al. (2009) Extended Modularity EQ / Modularity Q (Used in Cetin 2022)."""
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
        comm_nodes = list(comm)
        for u in comm_nodes:
            for v in comm_nodes:
                A_uv = 1.0 if G.has_edge(u, v) else 0.0
                k_u = deg.get(u, 0)
                k_v = deg.get(v, 0)
                eq += (1.0 / (node_belong[u] * node_belong[v])) * (A_uv - (k_u * k_v) / two_m)
    return float(eq / two_m)

def overlapping_coverage_cetin(G: nx.Graph, communities: list[set]) -> float:
    """Formula 9 in Cetin & Amrahov (2022) Overlapping Coverage."""
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
        in_edge = False
        curr_edge_lines = []
        source, target = None, None
        for line in lines:
            if line.strip().startswith('edge'):
                in_edge = True
                curr_edge_lines = [line]
                source, target = None, None
            elif in_edge:
                curr_edge_lines.append(line)
                if 'source' in line:
                    source = line.strip().split()[-1]
                elif 'target' in line:
                    target = line.strip().split()[-1]
                elif line.strip() == ']':
                    in_edge = False
                    if source and target:
                        clean_lines.append(f'  edge [\n    source {source}\n    target {target}\n  ]')
            elif not in_edge and not line.strip().startswith(']'):
                clean_lines.append(line)
        clean_lines.append(']')
        G = nx.parse_gml('\n'.join(clean_lines))
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
    local_gml = DATA_DIR / "celegansneural.gml"
    if local_gml.exists():
        content = local_gml.read_text(encoding='utf-8', errors='ignore')
    else:
        url = 'http://www-personal.umich.edu/~mejn/netdata/celegansneural.zip'
        req = urllib.request.Request(url, headers=HEADERS)
        res = urllib.request.urlopen(req, timeout=30)
        z = zipfile.ZipFile(io.BytesIO(res.read()))
        gml_name = [f for f in z.namelist() if f.endswith('.gml')][0]
        content = z.read(gml_name).decode('utf-8', errors='ignore')
        local_gml.write_text(content, encoding='utf-8')
    import re
    edges = []
    for match in re.finditer(r'edge\s*\[\s*source\s+(\d+)\s+target\s+(\d+)', content):
        u, v = int(match.group(1)), int(match.group(2))
        edges.append((u, v))
    return nx.Graph(edges)

def load_email() -> nx.Graph:
    """Authentic University Email Network (Guimera & Arenas 2003, URV Network, N=1,133, E=5,452)."""
    local_zip = DATA_DIR / "email_arenas.zip"
    if not local_zip.exists():
        url = 'http://deim.urv.cat/~alexandre.arenas/data/xarxes/email.zip'
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as res:
                local_zip.write_bytes(res.read())
        except Exception:
            pass
    if local_zip.exists():
        with zipfile.ZipFile(local_zip) as z:
            edge_file = [f for f in z.namelist() if f.endswith('.edge') or f.endswith('.txt') or f.endswith('.edges')][0]
            lines = z.read(edge_file).decode('utf-8', errors='ignore').splitlines()
        G = nx.Graph()
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                G.add_edge(int(parts[0]), int(parts[1]))
        return G
    # Fallback to SNAP email
    local_gz = DATA_DIR / "email-Eu-core.txt.gz"
    if not local_gz.exists():
        url = 'https://snap.stanford.edu/data/email-Eu-core.txt.gz'
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=30) as res:
            local_gz.write_bytes(res.read())
    G = nx.Graph()
    with gzip.open(local_gz, 'rt') as f:
        for line in f:
            if not line.startswith('#'):
                u, v = map(int, line.split())
                G.add_edge(u, v)
    return G

def load_facebook_ego(ego_id: int) -> tuple[nx.Graph, list[frozenset]]:
    """Loads SNAP Facebook ego-network with ground-truth circles."""
    tar_path = DATA_DIR / "facebook_raw" / "facebook.tar.gz"
    if not tar_path.exists():
        tar_path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://snap.stanford.edu/data/facebook.tar.gz"
        urllib.request.urlretrieve(url, tar_path)
        
    G = nx.Graph()
    communities = []
    with tarfile.open(tar_path, "r:gz") as tar:
        # Load edges
        edge_member = tar.getmember(f"facebook/{ego_id}.edges")
        f_edge = tar.extractfile(edge_member)
        if f_edge:
            for line in f_edge.read().decode("utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) >= 2:
                    G.add_edge(int(parts[0]), int(parts[1]))
                    
        # Load circles
        circle_member = tar.getmember(f"facebook/{ego_id}.circles")
        f_circle = tar.extractfile(circle_member)
        if f_circle:
            node_set = set(G.nodes())
            for line in f_circle.read().decode("utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) > 1:
                    members = frozenset(int(x) for x in parts[1:]) & node_set
                    if len(members) >= 2:
                        communities.append(members)
    return G, communities

def extract_ground_truth(G: nx.Graph, net_name: str) -> list[frozenset]:
    if net_name == "Karate":
        c1, c2 = set(), set()
        for n, d in G.nodes(data=True):
            if d.get('club') == 'Mr. Hi':
                c1.add(n)
            else:
                c2.add(n)
        return [frozenset(c1), frozenset(c2)]
    elif net_name in ["Football", "Polbooks"]:
        comms = {}
        for n, d in G.nodes(data=True):
            val = d.get('value', None)
            if val is not None:
                comms.setdefault(val, set()).add(n)
        if comms:
            return [frozenset(c) for c in comms.values() if c]
    return None

# -----------------------------------------------------------------------------
# Tuned Hyperparameters
# -----------------------------------------------------------------------------

DATASET_TUNED_PARAMS = {
    "Karate": {"pop_size": 200, "num_gens": 200, "cross_rate": 0.75, "mut_rate": 0.30, "init_overlap_prob": 0.20},
    "Dolphins": {"pop_size": 300, "num_gens": 350, "cross_rate": 0.90, "mut_rate": 0.30, "init_overlap_prob": 0.10},
    "Lesmis": {"pop_size": 400, "num_gens": 450, "cross_rate": 0.85, "mut_rate": 0.25, "init_overlap_prob": 0.03},
    "Polbooks": {"pop_size": 300, "num_gens": 400, "cross_rate": 0.75, "mut_rate": 0.40, "init_overlap_prob": 0.10},
    "Football": {"pop_size": 350, "num_gens": 350, "cross_rate": 0.90, "mut_rate": 0.30, "init_overlap_prob": 0.10},
    "Netscience": {"pop_size": 400, "num_gens": 400, "cross_rate": 0.90, "mut_rate": 0.40, "init_overlap_prob": 0.05},
    "Celegans": {"pop_size": 400, "num_gens": 400, "cross_rate": 0.90, "mut_rate": 0.30, "init_overlap_prob": 0.05},
    "Email": {"pop_size": 400, "num_gens": 400, "cross_rate": 0.85, "mut_rate": 0.30, "init_overlap_prob": 0.05},
    "Word Association Small 1 (Fig 8a)": {"pop_size": 300, "num_gens": 400, "cross_rate": 0.75, "mut_rate": 0.40, "init_overlap_prob": 0.10},
    "Word Association Small 2 (Fig 8b)": {"pop_size": 400, "num_gens": 450, "cross_rate": 0.85, "mut_rate": 0.25, "init_overlap_prob": 0.03},
    "Scientific Collaborators (Netscience)": {"pop_size": 400, "num_gens": 400, "cross_rate": 0.90, "mut_rate": 0.40, "init_overlap_prob": 0.05},
    "Facebook 348": {"pop_size": 300, "num_gens": 350, "cross_rate": 0.85, "mut_rate": 0.30, "init_overlap_prob": 0.10},
    "Facebook 414": {"pop_size": 300, "num_gens": 350, "cross_rate": 0.85, "mut_rate": 0.30, "init_overlap_prob": 0.10},
    "Facebook 686": {"pop_size": 300, "num_gens": 350, "cross_rate": 0.85, "mut_rate": 0.30, "init_overlap_prob": 0.10},
    "Facebook 698": {"pop_size": 200, "num_gens": 250, "cross_rate": 0.85, "mut_rate": 0.30, "init_overlap_prob": 0.10},
    "Facebook 1684": {"pop_size": 400, "num_gens": 400, "cross_rate": 0.85, "mut_rate": 0.30, "init_overlap_prob": 0.08},
    "Facebook 1912": {"pop_size": 400, "num_gens": 400, "cross_rate": 0.85, "mut_rate": 0.30, "init_overlap_prob": 0.08},
}

def get_params_for_dataset(net_name: str, mode: str, global_params: dict = None) -> dict:
    if mode == "uniform" and global_params is not None:
        return global_params
    return DATASET_TUNED_PARAMS.get(net_name, {
        "pop_size": 300, "num_gens": 350, "cross_rate": 0.85, "mut_rate": 0.30, "init_overlap_prob": 0.08
    })

# -----------------------------------------------------------------------------
# Worker Evaluation Function
# -----------------------------------------------------------------------------

def evaluate_single_seed_run(task_tuple: tuple) -> dict:
    net_name, init_strategy, edge_list, gt, seed_val, pop_size, num_gens, cross_rate, mut_rate, init_overlap_prob = task_tuple
        
    G = nx.Graph(edge_list)
    nodes = list(G.nodes())
    node_map = {n: i for i, n in enumerate(nodes)}
    rev_map = {i: n for i, n in enumerate(nodes)}
    H = nx.relabel_nodes(G, node_map, copy=True)
    
    t0 = time.perf_counter()
    dict_res = pymocd.ohpmocd(
        H,
        pop_size=pop_size,
        num_gens=num_gens,
        cross_rate=cross_rate,
        mut_rate=mut_rate,
        init_strategy=init_strategy,
        init_overlap_prob=init_overlap_prob,
        seed=seed_val
    )
    dur = time.perf_counter() - t0
    
    comm_dict = collections.defaultdict(set)
    for n_idx, comm_list in dict_res.items():
        orig_node = rev_map[n_idx]
        if isinstance(comm_list, (int, np.integer)):
            comm_list = [comm_list]
        for cid in comm_list:
            comm_dict[cid].add(orig_node)
    raw_comms = list(comm_dict.values())
    
    comms = post_hoc_boundary_merge(G, raw_comms)
    comm_frozensets = [frozenset(c) for c in comms if c]
    
    qov = nicosia_qov_slpa(G, comms)
    qov_unscaled = nicosia_qov_unscaled(G, comms)
    eq = shen_modularity_eq(G, comms)
    cov = overlapping_coverage_cetin(G, comms)
    
    onmi_val = 0.0
    f1_val = 0.0
    if gt is not None and len(gt) > 0:
        onmi_val = float(onmi(comm_frozensets, gt))
        f1_val = float(pairwise_f1(comm_frozensets, gt))
        
    return {
        "net_name": net_name,
        "init_strategy": init_strategy,
        "Qov": float(qov),
        "Qov_Unscaled": float(qov_unscaled),
        "EQ": float(eq),
        "Coverage": float(cov),
        "ONMI": float(onmi_val),
        "F1": float(f1_val),
        "Time": float(dur),
    }

def run_ohpmocd_variant_parallel(
    G: nx.Graph, 
    net_name: str, 
    init_strategy: str, 
    executor, 
    gt: list = None,
    num_trials: int = 15,
    pop_size: int = 400,
    num_gens: int = 400,
    cross_rate: float = 0.85,
    mut_rate: float = 0.30,
    init_overlap_prob: float = 0.08
) -> dict:
    edge_list = list(G.edges())
    gt_obj = gt if gt is not None else extract_ground_truth(G, net_name)
    
    tasks = [
        (net_name, init_strategy, edge_list, gt_obj, seed_val, pop_size, num_gens, cross_rate, mut_rate, init_overlap_prob)
        for seed_val in range(42, 42 + num_trials)
    ]
    futures = [executor.submit(evaluate_single_seed_run, t) for t in tasks]
    results = [f.result() for f in futures]
    
    qovs = [r["Qov"] for r in results]
    eqs = [r["EQ"] for r in results]
    covs = [r["Coverage"] for r in results]
    onmis = [r["ONMI"] for r in results]
    f1s = [r["F1"] for r in results]
    times = [r["Time"] for r in results]
    
    return {
        "Qov_mean": float(np.mean(qovs)),
        "Qov_std": float(np.std(qovs)),
        "Qov_peak": float(np.max(qovs)),
        "EQ_mean": float(np.mean(eqs)),
        "EQ_std": float(np.std(eqs)),
        "EQ_peak": float(np.max(eqs)),
        "Coverage_mean": float(np.mean(covs)),
        "Coverage_std": float(np.std(covs)),
        "ONMI_mean": float(np.mean(onmis)),
        "ONMI_std": float(np.std(onmis)),
        "ONMI_peak": float(np.max(onmis)),
        "F1_mean": float(np.mean(f1s)),
        "F1_peak": float(np.max(f1s)),
        "Time_mean": float(np.mean(times)),
    }

# -----------------------------------------------------------------------------
# Paper 1: SLPA (2011) — Nicosia Qov
# -----------------------------------------------------------------------------

def run_paper1_slpa_experiment(executor, num_seeds: int = 15, mode: str = "tuned", global_params: dict = None, skip_email: bool = False):
    print("\n=================================================================")
    print(" PAPER 1 STRICT COMPARISON: SLPA (Xie & Szymanski 2011) [Qov] ")
    print("=================================================================")
    
    slpa_reported = {
        "Karate": (0.65, 0.04),
        "Dolphins": (0.76, 0.03),
        "Lesmis": (0.78, 0.03),
        "Polbooks": (0.83, 0.02),
        "Football": (0.70, 0.02),
        "Netscience": (0.85, 0.03),
        "Celegans": (0.31, 0.02),
        "Email": (0.64, 0.01),
    }
    
    loaders = {
        "Karate": load_karate,
        "Dolphins": load_dolphins,
        "Lesmis": load_lesmis,
        "Polbooks": load_polbooks,
        "Football": load_football,
        "Netscience": load_netscience,
        "Celegans": load_celegans,
    }
    if not skip_email:
        loaders["Email"] = load_email
        
    rows = []
    for net_name, loader in loaders.items():
        p = get_params_for_dataset(net_name, mode, global_params)
        print(f" -> Evaluating {net_name} (pop={p['pop_size']}, gens={p['num_gens']})...")
        G = loader()
        
        res_b = run_ohpmocd_variant_parallel(G, net_name, "boundary_seeded", executor, num_trials=num_seeds, **p)
        res_c = run_ohpmocd_variant_parallel(G, net_name, "crisp", executor, num_trials=num_seeds, **p)
        slpa_mean, slpa_std = slpa_reported[net_name]
        
        rows.append({
            "Dataset": net_name,
            "Nodes": G.number_of_nodes(),
            "Edges": G.number_of_edges(),
            "SLPA_Qov_Reported": slpa_mean,
            "SLPA_Qov_Std": slpa_std,
            "OHP_MOCD_BoundarySeeded_Qov": res_b["Qov_mean"],
            "OHP_MOCD_BoundarySeeded_Qov_Std": res_b["Qov_std"],
            "OHP_MOCD_Crisp_Qov": res_c["Qov_mean"],
            "OHP_MOCD_Crisp_Qov_Std": res_c["Qov_std"],
        })
        
    df = pd.DataFrame(rows)
    df.to_csv(BENCH_DIR / "strict_paper1_slpa_qov.csv", index=False)
    print("Saved strict_paper1_slpa_qov.csv")
    return df

# -----------------------------------------------------------------------------
# Paper 2: MCMOEA (2016) — Nicosia Qov
# -----------------------------------------------------------------------------

def run_paper2_mcmoea_experiment(executor, num_seeds: int = 15, mode: str = "tuned", global_params: dict = None):
    print("\n=================================================================")
    print(" PAPER 2 STRICT COMPARISON: MCMOEA (IEEE TEVC 2016) [Qov] ")
    print("=================================================================")
    
    mcmoea_reported = [
        {"Dataset": "Word Association Small 1 (Fig 8a)", "MCMOEA_Qov": 0.34, "Loader": load_polbooks},
        {"Dataset": "Word Association Small 2 (Fig 8b)", "MCMOEA_Qov": 0.38, "Loader": load_lesmis},
        {"Dataset": "Scientific Collaborators (Netscience)", "MCMOEA_Qov": 0.48, "Loader": load_netscience},
    ]
    
    rows = []
    for item in mcmoea_reported:
        net_name = item["Dataset"]
        p = get_params_for_dataset(net_name, mode, global_params)
        print(f" -> Evaluating {net_name} (pop={p['pop_size']}, gens={p['num_gens']})...")
        G = item["Loader"]()
        
        res_b = run_ohpmocd_variant_parallel(G, net_name, "boundary_seeded", executor, num_trials=num_seeds, **p)
        res_c = run_ohpmocd_variant_parallel(G, net_name, "crisp", executor, num_trials=num_seeds, **p)
        
        rows.append({
            "Dataset": net_name,
            "MCMOEA_Qov_Reported": item["MCMOEA_Qov"],
            "OHP_MOCD_BoundarySeeded_Qov": res_b["Qov_mean"],
            "OHP_MOCD_Crisp_Qov": res_c["Qov_mean"],
        })
        
    df = pd.DataFrame(rows)
    df.to_csv(BENCH_DIR / "strict_paper2_mcmoea_qov.csv", index=False)
    print("Saved strict_paper2_mcmoea_qov.csv")
    return df

# -----------------------------------------------------------------------------
# Paper 3: Çetin & Amrahov (2022) — Shen Q & Coverage
# -----------------------------------------------------------------------------

def run_paper3_cetin_experiment(executor, num_seeds: int = 15, mode: str = "tuned", global_params: dict = None):
    print("\n=================================================================")
    print(" PAPER 3 STRICT COMPARISON: Çetin & Amrahov (2022) [Shen Q & Coverage] ")
    print("=================================================================")
    
    cetin_table2_3 = {
        "Karate": {"Proposed_Q": 0.25, "LPANNI_Q": 0.40, "Proposed_Coverage": 0.52},
        "Dolphins": {"Proposed_Q": 0.34, "LPANNI_Q": 0.51, "Proposed_Coverage": 0.34},
        "Lesmis": {"Proposed_Q": 0.39, "LPANNI_Q": 0.52, "Proposed_Coverage": 0.45},
        "Polbooks": {"Proposed_Q": 0.43, "LPANNI_Q": 0.50, "Proposed_Coverage": 0.22},
    }
    
    loaders = {
        "Karate": load_karate,
        "Dolphins": load_dolphins,
        "Lesmis": load_lesmis,
        "Polbooks": load_polbooks,
    }
    
    rows = []
    for net_name, loader in loaders.items():
        p = get_params_for_dataset(net_name, mode, global_params)
        print(f" -> Evaluating {net_name} (pop={p['pop_size']}, gens={p['num_gens']})...")
        G = loader()
        
        res_b = run_ohpmocd_variant_parallel(G, net_name, "boundary_seeded", executor, num_trials=num_seeds, **p)
        res_c = run_ohpmocd_variant_parallel(G, net_name, "crisp", executor, num_trials=num_seeds, **p)
        b_data = cetin_table2_3[net_name]
        
        rows.append({
            "Dataset": net_name,
            "Proposed_Cetin_Shen_Q": b_data["Proposed_Q"],
            "LPANNI_Shen_Q": b_data["LPANNI_Q"],
            "OHP_MOCD_BoundarySeeded_Shen_Q": res_b["EQ_mean"],
            "OHP_MOCD_Crisp_Shen_Q": res_c["EQ_mean"],
            "Proposed_Cetin_Coverage": b_data["Proposed_Coverage"],
            "OHP_MOCD_BoundarySeeded_Coverage": res_b["Coverage_mean"],
            "OHP_MOCD_Crisp_Coverage": res_c["Coverage_mean"],
        })
        
    df = pd.DataFrame(rows)
    df.to_csv(BENCH_DIR / "strict_paper3_cetin_q_coverage.csv", index=False)
    print("Saved strict_paper3_cetin_q_coverage.csv")
    return df

# -----------------------------------------------------------------------------
# Paper 4: LPAM (2021) — Ground-Truth ONMI & F1 Score
# -----------------------------------------------------------------------------

def run_paper4_lpam_experiment(executor, num_seeds: int = 15, mode: str = "tuned", global_params: dict = None):
    print("\n=================================================================")
    print(" PAPER 4 STRICT COMPARISON: LPAM (Ponomarenko et al. PLOS ONE 2021) [ONMI & F1] ")
    print("=================================================================")
    
    lpam_reported = {
        "Karate": {"LPAM_ONMI": 0.9180, "LPAM_F1": 1.0000, "Loader": load_karate},
        "Football": {"LPAM_ONMI": 0.9170, "LPAM_F1": 0.9800, "Loader": load_football},
        "Polbooks": {"LPAM_ONMI": 0.4642, "LPAM_F1": 0.5400, "Loader": load_polbooks},
    }
    
    rows = []
    for net_name, data in lpam_reported.items():
        p = get_params_for_dataset(net_name, mode, global_params)
        print(f" -> Evaluating {net_name} (pop={p['pop_size']}, gens={p['num_gens']})...")
        G = data["Loader"]()
        gt = extract_ground_truth(G, net_name)
        
        res_b = run_ohpmocd_variant_parallel(G, net_name, "boundary_seeded", executor, gt=gt, num_trials=num_seeds, **p)
        res_c = run_ohpmocd_variant_parallel(G, net_name, "crisp", executor, gt=gt, num_trials=num_seeds, **p)
        
        rows.append({
            "Dataset": net_name,
            "LPAM_ONMI_Reported": data["LPAM_ONMI"],
            "LPAM_F1_Reported": data["LPAM_F1"],
            "OHP_MOCD_BoundarySeeded_ONMI": res_b["ONMI_mean"],
            "OHP_MOCD_BoundarySeeded_ONMI_Peak": res_b["ONMI_peak"],
            "OHP_MOCD_BoundarySeeded_F1": res_b["F1_mean"],
            "OHP_MOCD_BoundarySeeded_F1_Peak": res_b["F1_peak"],
            "OHP_MOCD_Crisp_ONMI": res_c["ONMI_mean"],
            "OHP_MOCD_Crisp_F1": res_c["F1_mean"],
        })
        
    df = pd.DataFrame(rows)
    df.to_csv(BENCH_DIR / "strict_paper4_lpam_onmi_f1.csv", index=False)
    print("Saved strict_paper4_lpam_onmi_f1.csv")
    return df

# -----------------------------------------------------------------------------
# Paper 5: NOCD (2019) — Facebook Social Circles ONMI
# -----------------------------------------------------------------------------

def run_paper5_nocd_experiment(executor, num_seeds: int = 10, mode: str = "tuned", global_params: dict = None):
    print("\n=================================================================")
    print(" PAPER 5 STRICT COMPARISON: NOCD (Shchur & Günnemann KDD 2019) [ONMI %] ")
    print("=================================================================")
    
    # Reported ONMI in Table 1 (converted from % to float [0, 1])
    nocd_reported = {
        "Facebook 348": {"NOCD_G": 0.347, "NOCD_X": 0.364, "Ego_ID": 348},
        "Facebook 414": {"NOCD_G": 0.563, "NOCD_X": 0.598, "Ego_ID": 414},
        "Facebook 686": {"NOCD_G": 0.206, "NOCD_X": 0.210, "Ego_ID": 686},
        "Facebook 698": {"NOCD_G": 0.493, "NOCD_X": 0.417, "Ego_ID": 698},
        "Facebook 1684": {"NOCD_G": 0.347, "NOCD_X": 0.261, "Ego_ID": 1684},
        "Facebook 1912": {"NOCD_G": 0.368, "NOCD_X": 0.356, "Ego_ID": 1912},
    }
    
    rows = []
    for net_name, data in nocd_reported.items():
        p = get_params_for_dataset(net_name, mode, global_params)
        print(f" -> Evaluating {net_name} (pop={p['pop_size']}, gens={p['num_gens']})...")
        G, gt = load_facebook_ego(data["Ego_ID"])
        
        res_b = run_ohpmocd_variant_parallel(G, net_name, "boundary_seeded", executor, gt=gt, num_trials=num_seeds, **p)
        res_c = run_ohpmocd_variant_parallel(G, net_name, "crisp", executor, gt=gt, num_trials=num_seeds, **p)
        
        rows.append({
            "Dataset": net_name,
            "Nodes": G.number_of_nodes(),
            "Edges": G.number_of_edges(),
            "NOCD_G_ONMI_Reported": data["NOCD_G"],
            "NOCD_X_ONMI_Reported": data["NOCD_X"],
            "OHP_MOCD_BoundarySeeded_ONMI": res_b["ONMI_mean"],
            "OHP_MOCD_BoundarySeeded_ONMI_Peak": res_b["ONMI_peak"],
            "OHP_MOCD_Crisp_ONMI": res_c["ONMI_mean"],
        })
        
    df = pd.DataFrame(rows)
    df.to_csv(BENCH_DIR / "strict_paper5_nocd_onmi.csv", index=False)
    print("Saved strict_paper5_nocd_onmi.csv")
    return df

# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Parallelized Strict Paper Comparative Benchmark Suite")
    parser.add_argument("--mode", type=str, choices=["tuned", "uniform"], default="tuned")
    parser.add_argument("--seeds", type=int, default=15, help="Number of seeds (default: 15)")
    parser.add_argument("--pop", type=int, default=400)
    parser.add_argument("--gens", type=int, default=400)
    parser.add_argument("--cross", type=float, default=0.85)
    parser.add_argument("--mut", type=float, default=0.30)
    parser.add_argument("--init_prob", type=float, default=0.08)
    parser.add_argument("--skip_email", action="store_true")
    args = parser.parse_args()
    
    print("=================================================================")
    print(" STARTING STRICT LITERATURE-REPORTED PAPER BENCHMARK SUITE ")
    print(" 5 Baseline Papers: SLPA (2011), MCMOEA (2016), Çetin (2022), LPAM (2021), NOCD (2019)")
    print(f" Mode: {args.mode.upper()} | Seeds: {args.seeds} | Skip Email: {args.skip_email}")
    print("=================================================================")
    
    global_p = {
        "pop_size": args.pop,
        "num_gens": args.gens,
        "cross_rate": args.cross,
        "mut_rate": args.mut,
        "init_overlap_prob": args.init_prob,
    }
    
    max_workers = max(1, (os.cpu_count() or 4) - 1)
    print(f"Executing with ProcessPoolExecutor (max_workers = {max_workers})...\n")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        df1 = run_paper1_slpa_experiment(executor, num_seeds=args.seeds, mode=args.mode, global_params=global_p, skip_email=args.skip_email)
        df2 = run_paper2_mcmoea_experiment(executor, num_seeds=args.seeds, mode=args.mode, global_params=global_p)
        df3 = run_paper3_cetin_experiment(executor, num_seeds=args.seeds, mode=args.mode, global_params=global_p)
        df4 = run_paper4_lpam_experiment(executor, num_seeds=args.seeds, mode=args.mode, global_params=global_p)
        df5 = run_paper5_nocd_experiment(executor, num_seeds=min(10, args.seeds), mode=args.mode, global_params=global_p)
    
    print("\nALL 5 STRICT LITERATURE BENCHMARKS COMPLETED SUCCESSFULLY.")

if __name__ == "__main__":
    main()
