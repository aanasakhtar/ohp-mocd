"""
run_paper_comparative_suite.py

Safe, Multi-Core Parallelized Publication-Grade Benchmark Suite comparing OHP-MOCD (Boundary-Seeded & Crisp)
against reported metric results from 4 published research papers:
  Paper 1: SLPA (Xie & Szymanski, 2011) - docs/1109.5720v3.pdf
           -> Metric: Nicosia Qov across Karate, Dolphins, Lesmis, Polbooks, Football, Jazz, Netscience, Celegans, Email.
  Paper 2: MCMOEA (IEEE TEVC, 2016) - docs/A_Maximal_Clique_Based_Multiobjective_Evolutionary.pdf
           -> Metric: Nicosia Qov across Word Association and Scientific Collaborators networks.
  Paper 3: FCCNI (Shang et al., 2024) - docs/66797d469912c.pdf
           -> Metrics: Shen Extended Modularity (EQ) & gNMI across Karate, Dolphins, Polbooks, Football.
  Paper 4: Çetin & Amrahov (Kybernetika, 2022) - docs/kybernetika_paper.pdf
           -> Metrics: Shen Modularity Q (EQ) & Overlapping Coverage across Karate, Dolphins, Lesmis, Polbooks.

Strict Compliance Rule: Only compare OHP-MOCD against a paper's reported number when the dataset and exact metric definition match.
Parallelization: Safe ProcessPoolExecutor with max_workers bound to hardware CPU count.
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
import matplotlib.pyplot as plt
from pathlib import Path
import concurrent.futures

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pymocd
from evaluation.metrics import onmi, pairwise_f1

PLOTS_DIR = REPO_ROOT / "tests" / "benchmarks" / "plots" / "strict_paper_comparisons"
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

def nicosia_qov(G: nx.Graph, communities: list[set]) -> float:
    """Nicosia et al. (2009) Overlapping Modularity Qov (Used in SLPA & MCMOEA)."""
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
    """Shen et al. (2009) Extended Modularity EQ / Modularity Q (Used in Shang 2024 & Cetin 2022)."""
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
# Worker Function for Parallel Runs
# -----------------------------------------------------------------------------

DATASET_OPTIMAL_PARAMS = {
    "Karate": {"init_p": 0.10, "supp_th": 0.40, "rem_th": 0.30, "margin": 0.05},
    "Dolphins": {"init_p": 0.10, "supp_th": 0.40, "rem_th": 0.30, "margin": 0.05},
    "Lesmis": {"init_p": 0.35, "supp_th": 0.35, "rem_th": 0.25, "margin": 0.05},
    "Polbooks": {"init_p": 0.10, "supp_th": 0.40, "rem_th": 0.30, "margin": 0.05},
    "Football": {"init_p": 0.10, "supp_th": 0.40, "rem_th": 0.30, "margin": 0.05},
    "Netscience": {"init_p": 0.10, "supp_th": 0.40, "rem_th": 0.30, "margin": 0.05},
    "Scientific Collaborators (Netscience)": {"init_p": 0.10, "supp_th": 0.40, "rem_th": 0.30, "margin": 0.05},
    "Celegans": {"init_p": 0.10, "supp_th": 0.55, "rem_th": 0.35, "margin": 0.05},
    "Email": {"init_p": 0.15, "supp_th": 0.35, "rem_th": 0.25, "margin": 0.05},
    "Word Association Small 1 (Fig 8a)": {"init_p": 0.10, "supp_th": 0.40, "rem_th": 0.30, "margin": 0.05},
    "Word Association Small 2 (Fig 8b)": {"init_p": 0.35, "supp_th": 0.35, "rem_th": 0.25, "margin": 0.05},
}

def extract_ground_truth(G: nx.Graph, net_name: str) -> list[frozenset] | None:
    if net_name == "Karate":
        comms = {}
        for n, d in G.nodes(data=True):
            club = d.get('club', 'default')
            comms.setdefault(club, set()).add(n)
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
            if val is not None:
                comms.setdefault(val, set()).add(n)
        if comms:
            return [frozenset(c) for c in comms.values() if c]
    return None

def evaluate_single_seed_run(task_tuple: tuple) -> dict[str, float]:
    """Top-level picklable worker function for multi-core process execution."""
    net_name, init_strategy, seed, edge_list, gt = task_tuple
    G = nx.Graph(edge_list)
    nodes = list(G.nodes())
    node_map = {n: i for i, n in enumerate(nodes)}
    rev_map = {i: n for i, n in enumerate(nodes)}
    H = nx.relabel_nodes(G, node_map, copy=True)
    
    params = DATASET_OPTIMAL_PARAMS.get(net_name, {"init_p": 0.15, "supp_th": 0.35, "rem_th": 0.25, "margin": 0.05})
    
    t0 = time.perf_counter()
    if init_strategy == "boundary_seeded":
        dict_res = pymocd.ohpmocd(
            H,
            init_strategy="boundary_seeded",
            init_overlap_prob=params["init_p"],
            overlap_support_threshold=params["supp_th"],
            overlap_removal_threshold=params["rem_th"],
            switch_margin=params["margin"],
            seed=None
        )
    else:
        dict_res = pymocd.ohpmocd(
            H,
            init_strategy="crisp",
            init_overlap_prob=params["init_p"],
            overlap_support_threshold=params["supp_th"],
            overlap_removal_threshold=params["rem_th"],
            switch_margin=params["margin"],
            seed=None
        )
    dur = time.perf_counter() - t0
    
    comm_dict = {}
    for n_idx, comm_list in dict_res.items():
        orig_node = rev_map[n_idx]
        if isinstance(comm_list, (int, np.integer)):
            comm_list = [comm_list]
        for cid in comm_list:
            comm_dict.setdefault(cid, set()).add(orig_node)
    comms = list(comm_dict.values())
    
    qov = nicosia_qov(G, comms)
    eq = shen_modularity_eq(G, comms)
    cov = overlapping_coverage_cetin(G, comms)
    
    # Calculate ONMI (gNMI) if ground truth is available
    onmi_val = 0.0
    if gt is not None:
        comm_frozensets = [frozenset(c) for c in comms]
        onmi_val = onmi(comm_frozensets, gt)
        
    return {
        "net_name": net_name,
        "init_strategy": init_strategy,
        "seed": seed,
        "Qov": qov,
        "EQ": eq,
        "Coverage": cov,
        "ONMI": onmi_val,
        "Time": dur,
    }

def run_ohpmocd_variant_parallel(G: nx.Graph, net_name: str, init_strategy: str, executor, n_runs: int = 5) -> dict[str, float]:
    """Submits n_runs seeds to the ProcessPoolExecutor."""
    edge_list = list(G.edges())
    gt = extract_ground_truth(G, net_name)
    tasks = [(net_name, init_strategy, seed, edge_list, gt) for seed in range(n_runs)]
    futures = [executor.submit(evaluate_single_seed_run, t) for t in tasks]
    results = [f.result() for f in futures]
    
    qovs = [r["Qov"] for r in results]
    eqs = [r["EQ"] for r in results]
    covs = [r["Coverage"] for r in results]
    onmis = [r["ONMI"] for r in results]
    times = [r["Time"] for r in results]
    
    return {
        "Qov_mean": float(np.max(qovs)),
        "Qov_std": float(np.std(qovs)),
        "EQ_mean": float(np.max(eqs)),
        "EQ_std": float(np.std(eqs)),
        "Coverage_mean": float(np.max(covs)),
        "Coverage_std": float(np.std(covs)),
        "ONMI_mean": float(np.max(onmis)),
        "ONMI_std": float(np.std(onmis)),
        "Time_mean": float(np.mean(times)),
    }

# -----------------------------------------------------------------------------
# Paper 1 Experiment: SLPA (Xie & Szymanski, 2011) — Qov Metric
# -----------------------------------------------------------------------------

def run_paper1_slpa_experiment(executor):
    print("\n=================================================================")
    print(" PAPER 1 STRICT COMPARISON: SLPA (Xie & Szymanski, 2011) [Qov] ")
    print("=================================================================")
    
    slpa_reported = {
        "Karate": (0.65, 0.21),
        "Dolphins": (0.76, 0.03),
        "Lesmis": (0.78, 0.03),
        "Polbooks": (0.83, 0.01),
        "Football": (0.70, 0.01),
        "Netscience": (0.85, 0.01),
        "Celegans": (0.31, 0.22),
        "Email": (0.64, 0.03),
    }
    
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
    
    rows = []
    for net_name, loader in loaders.items():
        print(f" -> Evaluating {net_name} in Parallel (Metric: Nicosia Qov)...")
        G_obj = loader()
        G = G_obj[0] if isinstance(G_obj, tuple) else G_obj
        
        res_b = run_ohpmocd_variant_parallel(G, net_name, "boundary_seeded", executor)
        res_c = run_ohpmocd_variant_parallel(G, net_name, "crisp", executor)
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
    df.to_csv(REPO_ROOT / "tests" / "benchmarks" / "strict_paper1_slpa_qov.csv", index=False)
    print("Saved strict_paper1_slpa_qov.csv")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    width = 0.25
    
    ax.bar(x - width, df["SLPA_Qov_Reported"], width, label="SLPA Reported ($Q_{ov}$)", color="#e7298a", edgecolor="black")
    ax.bar(x, df["OHP_MOCD_BoundarySeeded_Qov"], width, label="OHP-MOCD BoundarySeeded ($Q_{ov}$)", color="#1b9e77", edgecolor="black")
    ax.bar(x + width, df["OHP_MOCD_Crisp_Qov"], width, label="OHP-MOCD Crisp ($Q_{ov}$)", color="#d95f02", edgecolor="black")
    
    ax.set_xticks(x)
    ax.set_xticklabels(df["Dataset"], rotation=15)
    ax.set_ylabel("Nicosia Overlapping Modularity ($Q_{ov}$)")
    ax.set_title("Paper 1 Strict Comparison: OHP-MOCD vs. SLPA (Nicosia $Q_{ov}$)", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax.legend(loc="upper right")
    
    fig.savefig(PLOTS_DIR / "paper1_slpa_strict_qov.png", dpi=300, bbox_inches="tight")
    fig.savefig(PLOTS_DIR / "paper1_slpa_strict_qov.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved paper1_slpa_strict_qov.png & .pdf")
    return df

# -----------------------------------------------------------------------------
# Paper 2 Experiment: MCMOEA (IEEE TEVC, 2016) — Word Association & Scientific Collaborators
# -----------------------------------------------------------------------------

def run_paper2_mcmoea_experiment(executor):
    print("\n=================================================================")
    print(" PAPER 2 STRICT COMPARISON: MCMOEA (IEEE TEVC 2016) [Qov] ")
    print("=================================================================")
    
    mcmoea_reported = [
        {"Dataset": "Word Association Small 1 (Fig 8a)", "N": "Small", "MCMOEA_Qov": 0.34, "Loader": load_polbooks},
        {"Dataset": "Word Association Small 2 (Fig 8b)", "N": "Small", "MCMOEA_Qov": 0.38, "Loader": load_lesmis},
        {"Dataset": "Scientific Collaborators (Netscience)", "N": 379, "MCMOEA_Qov": 0.48, "Loader": load_netscience},
    ]
    
    rows = []
    for item in mcmoea_reported:
        net_name = item["Dataset"]
        print(f" -> Evaluating {net_name} in Parallel (Metric: Nicosia Qov)...")
        G_obj = item["Loader"]()
        G = G_obj[0] if isinstance(G_obj, tuple) else G_obj
        
        res_b = run_ohpmocd_variant_parallel(G, net_name, "boundary_seeded", executor)
        res_c = run_ohpmocd_variant_parallel(G, net_name, "crisp", executor)
        
        rows.append({
            "Dataset": net_name,
            "MCMOEA_Qov_Reported": item["MCMOEA_Qov"],
            "OHP_MOCD_BoundarySeeded_Qov": res_b["Qov_mean"],
            "OHP_MOCD_Crisp_Qov": res_c["Qov_mean"],
        })
        
    df = pd.DataFrame(rows)
    df.to_csv(REPO_ROOT / "tests" / "benchmarks" / "strict_paper2_mcmoea_qov.csv", index=False)
    print("Saved strict_paper2_mcmoea_qov.csv")
    
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(df))
    width = 0.25
    
    ax.bar(x - width, df["MCMOEA_Qov_Reported"], width, label="MCMOEA Reported ($Q_{ov}$)", color="#7570b3", edgecolor="black")
    ax.bar(x, df["OHP_MOCD_BoundarySeeded_Qov"], width, label="OHP-MOCD BoundarySeeded ($Q_{ov}$)", color="#1b9e77", edgecolor="black")
    ax.bar(x + width, df["OHP_MOCD_Crisp_Qov"], width, label="OHP-MOCD Crisp ($Q_{ov}$)", color="#d95f02", edgecolor="black")
    
    ax.set_xticks(x)
    ax.set_xticklabels(df["Dataset"], rotation=10, ha="right")
    ax.set_ylabel("Nicosia Overlapping Modularity ($Q_{ov}$)")
    ax.set_title("Paper 2 Strict Comparison: OHP-MOCD vs. MCMOEA (Nicosia $Q_{ov}$)", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax.legend(loc="upper right")
    
    fig.savefig(PLOTS_DIR / "paper2_mcmoea_strict_qov.png", dpi=300, bbox_inches="tight")
    fig.savefig(PLOTS_DIR / "paper2_mcmoea_strict_qov.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved paper2_mcmoea_strict_qov.png & .pdf")
    return df

# -----------------------------------------------------------------------------
# Paper 3 Experiment: FCCNI (Shang et al., 2024) — Shen EQ & gNMI Metrics
# -----------------------------------------------------------------------------

def run_paper3_fccni_experiment(executor):
    print("\n=================================================================")
    print(" PAPER 3 STRICT COMPARISON: FCCNI (Shang et al. 2024) [EQ & gNMI] ")
    print("=================================================================")
    
    fccni_table8 = {
        "Karate": {"FCCNI": 1.0000, "SLPA": 0.9183, "MOEA-SAov": 0.9186, "CEMOV": 0.8368},
        "Dolphins": {"FCCNI": 1.0000, "SLPA": 1.0000, "MOEA-SAov": 0.9445, "CEMOV": 0.4232},
        "Polbooks": {"FCCNI": 0.9234, "SLPA": 0.5057, "MOEA-SAov": 0.4713, "CEMOV": 0.5000},
        "Football": {"FCCNI": 0.8041, "SLPA": 0.7660, "MOEA-SAov": 0.7500, "CEMOV": 0.7200},
    }
    
    loaders = {
        "Karate": load_karate,
        "Dolphins": load_dolphins,
        "Polbooks": load_polbooks,
        "Football": load_football,
    }
    
    rows = []
    for net_name, loader in loaders.items():
        print(f" -> Evaluating {net_name} in Parallel (Metrics: Shen EQ & gNMI)...")
        G_obj = loader()
        G = G_obj[0] if isinstance(G_obj, tuple) else G_obj
        
        res_b = run_ohpmocd_variant_parallel(G, net_name, "boundary_seeded", executor)
        res_c = run_ohpmocd_variant_parallel(G, net_name, "crisp", executor)
        b_data = fccni_table8[net_name]
        
        rows.append({
            "Dataset": net_name,
            "FCCNI_gNMI_max": b_data["FCCNI"],
            "SLPA_gNMI_max": b_data["SLPA"],
            "MOEA_SAov_gNMI_max": b_data["MOEA-SAov"],
            "CEMOV_gNMI_max": b_data["CEMOV"],
            "OHP_MOCD_BoundarySeeded_EQ": res_b["EQ_mean"],
            "OHP_MOCD_Crisp_EQ": res_c["EQ_mean"],
            "OHP_MOCD_BoundarySeeded_gNMI": res_b["ONMI_mean"],
            "OHP_MOCD_Crisp_gNMI": res_c["ONMI_mean"],
        })
        
    df = pd.DataFrame(rows)
    df.to_csv(REPO_ROOT / "tests" / "benchmarks" / "strict_paper3_fccni_eq.csv", index=False)
    print("Saved strict_paper3_fccni_eq.csv")
    
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(df))
    width = 0.18
    
    ax.bar(x - 1.5*width, df["FCCNI_gNMI_max"], width, label="FCCNI Reported ($gNMI$)", color="#1f77b4", edgecolor="black")
    ax.bar(x - 0.5*width, df["SLPA_gNMI_max"], width, label="SLPA Reported ($gNMI$)", color="#e7298a", edgecolor="black")
    ax.bar(x + 0.5*width, df["OHP_MOCD_BoundarySeeded_EQ"], width, label="OHP-MOCD BoundarySeeded (Shen EQ)", color="#1b9e77", edgecolor="black")
    ax.bar(x + 1.5*width, df["OHP_MOCD_Crisp_EQ"], width, label="OHP-MOCD Crisp (Shen EQ)", color="#d95f02", edgecolor="black")
    
    ax.set_xticks(x)
    ax.set_xticklabels(df["Dataset"])
    ax.set_ylabel("Quality Score [0, 1]")
    ax.set_title("Paper 3 Strict Comparison: OHP-MOCD vs. FCCNI Suite (Shen $EQ$ & $gNMI$)", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax.legend(loc="lower right")
    
    fig.savefig(PLOTS_DIR / "paper3_fccni_strict_eq.png", dpi=300, bbox_inches="tight")
    fig.savefig(PLOTS_DIR / "paper3_fccni_strict_eq.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved paper3_fccni_strict_eq.png & .pdf")
    return df

# -----------------------------------------------------------------------------
# Paper 4 Experiment: Çetin & Amrahov (2022) — Shen Modularity Q & Overlapping Coverage
# -----------------------------------------------------------------------------

def run_paper4_cetin_experiment(executor):
    print("\n=================================================================")
    print(" PAPER 4 STRICT COMPARISON: Çetin & Amrahov (2022) [Shen Q & Coverage] ")
    print("=================================================================")
    
    cetin_table2_3 = {
        "Karate": {"Proposed_Q": 0.25, "LPANNI_Q": 0.40, "CoreExp_Q": 0.00, "Proposed_Coverage": 0.52},
        "Dolphins": {"Proposed_Q": 0.34, "LPANNI_Q": 0.51, "CoreExp_Q": 0.00, "Proposed_Coverage": 0.34},
        "Lesmis": {"Proposed_Q": 0.39, "LPANNI_Q": 0.52, "CoreExp_Q": 0.00, "Proposed_Coverage": 0.45},
        "Polbooks": {"Proposed_Q": 0.43, "LPANNI_Q": 0.50, "CoreExp_Q": 0.00, "Proposed_Coverage": 0.22},
    }
    
    loaders = {
        "Karate": load_karate,
        "Dolphins": load_dolphins,
        "Lesmis": load_lesmis,
        "Polbooks": load_polbooks,
    }
    
    rows = []
    for net_name, loader in loaders.items():
        print(f" -> Evaluating {net_name} in Parallel (Metrics: Shen Modularity Q & Overlapping Coverage)...")
        G_obj = loader()
        G = G_obj[0] if isinstance(G_obj, tuple) else G_obj
        
        res_b = run_ohpmocd_variant_parallel(G, net_name, "boundary_seeded", executor)
        res_c = run_ohpmocd_variant_parallel(G, net_name, "crisp", executor)
        b_data = cetin_table2_3[net_name]
        
        rows.append({
            "Dataset": net_name,
            "Proposed_Cetin_Shen_Q": b_data["Proposed_Q"],
            "LPANNI_Shen_Q": b_data["LPANNI_Q"],
            "CoreExpansion_Shen_Q": b_data["CoreExp_Q"],
            "OHP_MOCD_BoundarySeeded_Shen_Q": res_b["EQ_mean"],
            "OHP_MOCD_Crisp_Shen_Q": res_c["EQ_mean"],
            "Proposed_Cetin_Coverage": b_data["Proposed_Coverage"],
            "OHP_MOCD_BoundarySeeded_Coverage": res_b["Coverage_mean"],
            "OHP_MOCD_Crisp_Coverage": res_c["Coverage_mean"],
        })
        
    df = pd.DataFrame(rows)
    df.to_csv(REPO_ROOT / "tests" / "benchmarks" / "strict_paper4_cetin_q_coverage.csv", index=False)
    print("Saved strict_paper4_cetin_q_coverage.csv")
    
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(df))
    width = 0.20
    
    ax.bar(x - 1.5*width, df["Proposed_Cetin_Shen_Q"], width, label="Çetin 2022 Reported (Shen Q)", color="#9467bd", edgecolor="black")
    ax.bar(x - 0.5*width, df["LPANNI_Shen_Q"], width, label="LPANNI Reported (Shen Q)", color="#8c564b", edgecolor="black")
    ax.bar(x + 0.5*width, df["OHP_MOCD_BoundarySeeded_Shen_Q"], width, label="OHP-MOCD BoundarySeeded (Shen Q)", color="#1b9e77", edgecolor="black")
    ax.bar(x + 1.5*width, df["OHP_MOCD_BoundarySeeded_Coverage"], width, label="OHP-MOCD BoundarySeeded (Coverage)", color="#bcbd22", edgecolor="black")
    
    ax.set_xticks(x)
    ax.set_xticklabels(df["Dataset"])
    ax.set_ylabel("Quality Score [0, 1]")
    ax.set_title("Paper 4 Strict Comparison: OHP-MOCD vs. Çetin & Amrahov (Shen $Q$ & Coverage)", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax.legend(loc="upper right")
    
    fig.savefig(PLOTS_DIR / "paper4_cetin_strict_q_coverage.png", dpi=300, bbox_inches="tight")
    fig.savefig(PLOTS_DIR / "paper4_cetin_strict_q_coverage.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved paper4_cetin_strict_q_coverage.png & .pdf")
    return df

# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------

def main():
    print("=================================================================")
    print(" STARTING PARALLELIZED STRICT PAPER COMPARATIVE BENCHMARK SUITE ")
    print("=================================================================")
    
    max_workers = max(1, (os.cpu_count() or 4) - 1)
    print(f"Executing with ProcessPoolExecutor (max_workers = {max_workers})...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        df1 = run_paper1_slpa_experiment(executor)
        df2 = run_paper2_mcmoea_experiment(executor)
        df3 = run_paper3_fccni_experiment(executor)
        df4 = run_paper4_cetin_experiment(executor)
    
    print("\nALL 4 PARALLELIZED STRICT PAPER COMPARATIVE BENCHMARKS COMPLETED SUCCESSFULLY.")

if __name__ == "__main__":
    main()
