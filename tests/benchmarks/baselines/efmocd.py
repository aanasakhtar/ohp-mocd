"""
tests/benchmarks/baselines/efmocd.py

EF-MOCD: Evolutionary Multiobjective Optimization-Based Fuzzy Method for Overlapping Community Detection
Reference:
  Ye Tian, Shangshang Yang, Xingyi Zhang,
  "An Evolutionary Multiobjective Optimization Based Fuzzy Method for Overlapping Community Detection",
  IEEE Transactions on Fuzzy Systems, Vol. 28, No. 11, pp. 2841-2855, 2020.
  DOI: 10.1109/TFUZZ.2019.2941687

Vectorized C-Speed Implementation:
  - 100% Pure Topological / Structure-based Overlapping MOEA (Operates strictly on G=(V, E)).
  - Vectorized all-pairs shortest paths and NumPy broadcasting for instantaneous objective evaluation.
"""

import random
import collections
import numpy as np
import networkx as nx

def compute_all_pairs_distances(G: nx.Graph) -> np.ndarray:
    nodes = sorted(list(G.nodes()))
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    dist_matrix = np.full((n, n), fill_value=n + 1.0, dtype=np.float64)
    np.fill_diagonal(dist_matrix, 0.0)
    
    for u, paths in nx.all_pairs_shortest_path_length(G):
        u_idx = node_to_idx[u]
        for v, d in paths.items():
            dist_matrix[u_idx, node_to_idx[v]] = float(d)
            
    return dist_matrix

def compute_fuzzy_membership(dist_matrix: np.ndarray, center_indices: list[int], m_fuzzy: float = 2.0) -> np.ndarray:
    n = dist_matrix.shape[0]
    K = len(center_indices)
    
    dists = dist_matrix[:, center_indices]  # Shape (N, K)
    p = 2.0 / (m_fuzzy - 1.0)
    
    inv_dists = 1.0 / (dists + 1e-6) ** p
    denom = np.sum(inv_dists, axis=1, keepdims=True)
    U = inv_dists / denom
    
    # Handle zero distances (centers themselves)
    zero_mask = (dists == 0.0)
    if np.any(zero_mask):
        row_zeros = np.any(zero_mask, axis=1)
        zero_counts = np.sum(zero_mask, axis=1, keepdims=True)
        U[row_zeros, :] = np.where(zero_mask[row_zeros, :], 1.0 / np.maximum(1.0, zero_counts[row_zeros, :]), 0.0)
        
    return U

def evaluate_efmocd_objectives_vec(dist_matrix: np.ndarray, center_indices: list[int], degs: np.ndarray, edge_u: np.ndarray, edge_v: np.ndarray, two_m: float, n: int) -> tuple[float, float, np.ndarray]:
    U = compute_fuzzy_membership(dist_matrix, center_indices)
    
    # f1: Fuzzy topological compactness (minimize)
    center_dists = dist_matrix[:, center_indices]
    f1 = float(np.sum((U ** 2.0) * (center_dists ** 2.0))) / float(n)
    
    # f2: Negative fuzzy modularity (minimize -EQ)
    internal_fuzzy_edges = np.sum(U[edge_u, :] * U[edge_v, :], axis=0)  # Shape (K,)
    vol_k = np.sum(U * degs[:, None], axis=0)                           # Shape (K,)
    f2_mod = np.sum((2.0 * internal_fuzzy_edges / two_m) - ((vol_k / two_m) ** 2.0))
    
    return f1, -float(f2_mod), U

def decode_overlapping_communities(U: np.ndarray, alpha: float = 0.50) -> list[frozenset]:
    n, K = U.shape
    comm_dict = collections.defaultdict(set)
    
    max_vals = np.max(U, axis=1, keepdims=True)
    thresh = alpha * max_vals
    
    # Membership mask
    assigned_mask = (U >= thresh) & (U > 0.05)
    
    for i in range(n):
        c_assigned = np.where(assigned_mask[i, :])[0]
        if len(c_assigned) > 0:
            for k in c_assigned:
                comm_dict[k].add(i)
        else:
            best_k = int(np.argmax(U[i, :]))
            comm_dict[best_k].add(i)
            
    return [frozenset(c) for c in comm_dict.values() if len(c) > 0]

def run_efmocd(G: nx.Graph, pop_size: int = 100, num_gens: int = 100, alpha: float = 0.50, seed: int = 42) -> list[frozenset]:
    """Runs EF-MOCD (Tian et al. 2020) on graph G."""
    random.seed(seed)
    np.random.seed(seed)
    
    nodes = list(G.nodes())
    n = len(nodes)
    if n <= 2:
        return [frozenset(nodes)]
        
    node_map = {node: i for i, node in enumerate(nodes)}
    rev_map = {i: node for i, node in enumerate(nodes)}
    H = nx.relabel_nodes(G, node_map, copy=True)
    
    edge_list = list(H.edges())
    m_edges = len(edge_list)
    if m_edges == 0:
        return [frozenset([n]) for n in nodes]
        
    two_m = max(1.0, 2.0 * m_edges)
    degs = np.array([H.degree(i) for i in range(n)], dtype=np.float64)
    edge_u = np.array([u for u, v in edge_list], dtype=np.int32)
    edge_v = np.array([v for u, v in edge_list], dtype=np.int32)
    
    dist_matrix = compute_all_pairs_distances(H)
    
    k_min = 2
    k_max = max(2, min(25, int(np.ceil(np.sqrt(n) * 1.5))))
    
    population = []
    for _ in range(pop_size):
        K_rand = random.randint(k_min, k_max)
        centers = random.sample(range(n), K_rand)
        population.append(centers)
        
    best_ind = None
    best_mod = -1e9
    best_U = None
    
    for gen in range(num_gens):
        evals = []
        for ind in population:
            f1, f2, U = evaluate_efmocd_objectives_vec(dist_matrix, ind, degs, edge_u, edge_v, two_m, n)
            evals.append((f1, f2, ind, U))
            if -f2 > best_mod:
                best_mod = -f2
                best_ind = ind
                best_U = U
                
        evals.sort(key=lambda x: (x[1], x[0]))
        elite = [x[2] for x in evals[:max(2, pop_size // 5)]]
        
        next_pop = list(elite)
        while len(next_pop) < pop_size:
            parent = random.choice(elite)
            child = list(parent)
            mut_type = random.random()
            if mut_type < 0.4 and len(child) > 0:
                c_idx = random.randint(0, len(child) - 1)
                nbrs = list(H.neighbors(child[c_idx]))
                if nbrs:
                    child[c_idx] = random.choice(nbrs)
            elif mut_type < 0.7 and len(child) < k_max:
                new_c = random.randint(0, n - 1)
                if new_c not in child:
                    child.append(new_c)
            elif len(child) > k_min:
                child.pop(random.randint(0, len(child) - 1))
            next_pop.append(child)
        population = next_pop
        
    if best_U is None:
        _, _, best_U = evaluate_efmocd_objectives_vec(dist_matrix, population[0], degs, edge_u, edge_v, two_m, n)
        
    comms_idx = decode_overlapping_communities(best_U, alpha=alpha)
    return [frozenset(rev_map[i] for i in c) for c in comms_idx if c]
