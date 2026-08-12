"""
tests/benchmarks/baselines.py

Standalone, high-performance implementations of baseline algorithms for direct,
head-to-head empirical comparison against OHP-MOCD:

1. SLPA (Xie & Szymanski, IEEE TKDE 2012)
2. MCMOEA (Wen et al., IEEE TEVC 2016) — via pymocd.mcmoea (Rust)
3. FCCNI (Shang et al., Applied Soft Computing 2024) — Intimacy-corrected LPA
4. Çetin 2022 (Çetin & Amrahov, Kybernetika 2022) — Core expansion LPA
"""

import random
import collections
import numpy as np
import networkx as nx
import pymocd

# -----------------------------------------------------------------------------
# 1. SLPA (Xie & Szymanski, 2012)
# -----------------------------------------------------------------------------
def run_slpa(G: nx.Graph, T: int = 100, r: float = 0.05, seed: int = None) -> list[frozenset]:
    """Speaker-Listener Label Propagation Algorithm."""
    rng = random.Random(seed)
    memory = {v: collections.Counter([v]) for v in G.nodes()}
    nodes = list(G.nodes())
    for _ in range(T):
        rng.shuffle(nodes)
        for listener in nodes:
            neighbors = list(G.neighbors(listener))
            if not neighbors:
                continue
            heard = collections.Counter()
            for speaker in neighbors:
                mem = memory[speaker]
                total = sum(mem.values())
                keys = list(mem.keys())
                weights = [mem[k] / total for k in keys]
                choice = rng.choices(keys, weights=weights)[0]
                heard[choice] += 1
            best_label = heard.most_common(1)[0][0]
            memory[listener][best_label] += 1

    communities = collections.defaultdict(set)
    for v, mem in memory.items():
        total = sum(mem.values())
        for label, count in mem.items():
            if count / total >= r:
                communities[label].add(v)
    return [frozenset(c) for c in communities.values() if c]

# -----------------------------------------------------------------------------
# 2. MCMOEA (Wen et al., 2016) — Rust Binding
# -----------------------------------------------------------------------------
def run_mcmoea(G: nx.Graph, pop_size: int = 100, num_gens: int = 100) -> list[frozenset]:
    """Maximal Clique-Based Multi-Objective Evolutionary Algorithm."""
    nodes = sorted(G.nodes())
    node_map = {n: i for i, n in enumerate(nodes)}
    rev_map = {i: n for i, n in enumerate(nodes)}
    H = nx.relabel_nodes(G, node_map, copy=True)
    res_dict = pymocd.mcmoea(H, pop_size=pop_size, num_gens=num_gens)
    
    comm_dict = collections.defaultdict(set)
    for n_idx, comm_list in res_dict.items():
        orig = rev_map.get(n_idx, n_idx)
        if isinstance(comm_list, (int, np.integer)):
            comm_list = [comm_list]
        for cid in comm_list:
            comm_dict[cid].add(orig)
    return [frozenset(c) for c in comm_dict.values() if c]

# -----------------------------------------------------------------------------
# 3. FCCNI (Shang et al., 2024) — Intimacy-Corrected LPA
# -----------------------------------------------------------------------------
def run_fccni(G: nx.Graph, max_iters: int = 50, tau: float = 0.35, seed: int = None) -> list[frozenset]:
    """Fusion of Connectivity & Intimacy Correction Algorithm (Pure Python)."""
    rng = random.Random(seed)
    deg = dict(G.degree())
    
    # Corrected Node Intimacy matrix CNI(u,v) = |N(u) ∩ N(v)| / min(d(u), d(v))
    cni = {}
    for u in G.nodes():
        u_nbrs = set(G.neighbors(u))
        for v in u_nbrs:
            v_nbrs = set(G.neighbors(v))
            common = len(u_nbrs & v_nbrs)
            cni[(u, v)] = common / max(1, min(deg[u], deg[v]))

    # Initial crisp assignment
    memberships = {u: [u] for u in G.nodes()}
    nodes = list(G.nodes())

    for _ in range(max_iters):
        rng.shuffle(nodes)
        changed = False
        for u in nodes:
            nbrs = list(G.neighbors(u))
            if not nbrs:
                continue
            
            # Score candidate communities using CNI and topological fitness d_in/d(u)
            comm_scores = collections.defaultdict(float)
            for v in nbrs:
                intimacy = cni.get((u, v), 0.1)
                for c in memberships[v]:
                    # OCCSA-style in-degree ratio + intimacy weight
                    in_deg = sum(1 for w in G.neighbors(u) if c in memberships[w])
                    fitness = in_deg / max(1, deg[u])
                    comm_scores[c] += fitness + intimacy

            if not comm_scores:
                continue

            max_score = max(comm_scores.values())
            best_comms = [c for c, sc in comm_scores.items() if sc >= tau * max_score]
            if not best_comms:
                best_comms = [max(comm_scores, key=comm_scores.get)]

            if set(memberships[u]) != set(best_comms):
                memberships[u] = best_comms
                changed = True

        if not changed:
            break

    comm_dict = collections.defaultdict(set)
    for u, comms in memberships.items():
        for c in comms:
            comm_dict[c].add(u)
    return [frozenset(c) for c in comm_dict.values() if c]

# -----------------------------------------------------------------------------
# 4. Çetin 2022 (Çetin & Amrahov, 2022) — Core Expansion & LPA
# -----------------------------------------------------------------------------
def run_cetin2022(G: nx.Graph, overlap_threshold: float = 0.30, seed: int = None) -> list[frozenset]:
    """Core Expansion & Label Propagation Algorithm (Çetin 2022)."""
    rng = random.Random(seed)
    deg = dict(G.degree())
    
    # Identify core nodes (top degree percentile)
    sorted_nodes = sorted(G.nodes(), key=lambda n: deg[n], reverse=True)
    n_cores = max(2, int(0.20 * len(sorted_nodes)))
    cores = set(sorted_nodes[:n_cores])
    
    memberships = {}
    for cid, core in enumerate(cores):
        memberships[core] = [cid]
    
    non_cores = [n for n in sorted_nodes if n not in cores]
    for n in non_cores:
        nbrs = list(G.neighbors(n))
        comm_counts = collections.Counter()
        for nbr in nbrs:
            if nbr in memberships:
                for c in memberships[nbr]:
                    comm_counts[c] += 1
        if comm_counts:
            max_c = comm_counts.most_common(1)[0][1]
            admitted = [c for c, cnt in comm_counts.items() if cnt / max_c >= (1.0 - overlap_threshold)]
            memberships[n] = admitted
        else:
            memberships[n] = [n]
            
    comm_dict = collections.defaultdict(set)
    for u, comms in memberships.items():
        for c in comms:
            comm_dict[c].add(u)
    return [frozenset(c) for c in comm_dict.values() if c]
