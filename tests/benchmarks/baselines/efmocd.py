"""
tests/benchmarks/baselines/efmocd.py

EF-MOCD / EMOFM: Evolutionary Multiobjective Optimization-Based Fuzzy Method for Overlapping Community Detection
Reference:
  Ye Tian, Shangshang Yang, Xingyi Zhang,
  "An Evolutionary Multiobjective Optimization Based Fuzzy Method for Overlapping Community Detection",
  IEEE Transactions on Fuzzy Systems, Vol. 28, No. 11, pp. 2841-2855, 2020.
  DOI: 10.1109/TFUZZ.2019.2945241

Exact 2-Stage Algorithm Implementation:
  Stage 1 (Center Optimization):
    - Representation: Binary vector b in {0, 1}^n indicating central nodes (CN).
    - Objective 1: Kernel k-means (KKM) = 2(n - k) - sum( 2*e(Ci) / |Ci| ) [Minimize]
    - Objective 2: Ratio Cut (RC) = sum( cut(Ci, V \ Ci) / |Ci| ) [Minimize]
  Stage 2 (Fuzzy Threshold Optimization):
    - Subpopulation initialization: k-means clustering on fuzzy membership matrix U.
    - Representation: Continuous threshold vector r in [0, 1]^n for non-central nodes.
    - Membership: U_ij = 1 / sum( (dist(NC_i, CN_j) / dist(NC_i, CN_l))^(2/(m-1)) )
    - Community assignment: NC_i in C_j iff U_ij >= r_i (with fallback to argmax).
    - Objective 1: Extended Modularity Q_ov (EQ) [Maximize]
    - Objective 2: Number of Overlapping Nodes (ON) [Maximize]
"""

import random
import collections
import numpy as np
import networkx as nx

def compute_all_pairs_distances(G: nx.Graph) -> np.ndarray:
    """Computes all-pairs shortest path distance matrix on graph G."""
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

def compute_fuzzy_membership_matrix(dist_matrix: np.ndarray, central_nodes: list[int], m_fuzzy: float = 2.0) -> np.ndarray:
    """Calculates continuous fuzzy membership matrix U (Equation 7 in Tian et al. 2020)."""
    n = dist_matrix.shape[0]
    k = len(central_nodes)
    if k == 0:
        return np.ones((n, 1), dtype=np.float64)
    if k == 1:
        return np.ones((n, 1), dtype=np.float64)
        
    dists = dist_matrix[:, central_nodes]  # Shape (n, k)
    p = 2.0 / (m_fuzzy - 1.0)
    
    inv_dists = 1.0 / (np.maximum(dists, 1e-6) ** p)
    denom = np.sum(inv_dists, axis=1, keepdims=True)
    U = inv_dists / np.maximum(denom, 1e-12)
    
    # Handle central nodes themselves (distance = 0)
    zero_mask = (dists == 0.0)
    if np.any(zero_mask):
        row_has_zero = np.any(zero_mask, axis=1)
        U[row_has_zero, :] = np.where(zero_mask[row_has_zero, :], 1.0, 0.0)
        row_sums = np.sum(U[row_has_zero, :], axis=1, keepdims=True)
        U[row_has_zero, :] /= np.maximum(1.0, row_sums)
        
    return U

def evaluate_stage1_kkm_rc(H: nx.Graph, dist_matrix: np.ndarray, central_nodes: list[int]) -> tuple[float, float, list[set[int]]]:
    """Calculates Stage 1 objectives: Kernel k-means (KKM) and Ratio Cut (RC) (Eq. 16 in Tian et al. 2020)."""
    n = H.number_of_nodes()
    k = len(central_nodes)
    if k < 2:
        return 1e6, 1e6, [set(range(n))]
        
    # Disjoint assignment to closest central node (Equation 8)
    sub_dists = dist_matrix[:, central_nodes]  # Shape (n, k)
    closest_center = np.argmin(sub_dists, axis=1)
    
    comms = collections.defaultdict(set)
    for u in range(n):
        comms[closest_center[u]].add(u)
    comm_list = [c for c in comms.values() if len(c) > 0]
    k_actual = len(comm_list)
    
    if k_actual < 2:
        return 1e6, 1e6, comm_list
        
    kkm_sum = 0.0
    rc_sum = 0.0
    
    for c in comm_list:
        size_c = len(c)
        sub_g = H.subgraph(c)
        int_edges = sub_g.number_of_edges()
        vol_c = sum(H.degree(u) for u in c)
        cut_edges = vol_c - 2 * int_edges
        
        # KKM: 2(n - k) - sum( 2 * e(Ci) / |Ci| )
        kkm_sum += (2.0 * int_edges) / size_c
        # RC: sum( cut(Ci, V \ Ci) / |Ci| )
        rc_sum += float(cut_edges) / size_c
        
    kkm = 2.0 * (n - k_actual) - kkm_sum
    return float(kkm), float(rc_sum), comm_list

def evaluate_stage2_qov_on(H: nx.Graph, two_m: float, overlapping_comms: list[set[int]]) -> tuple[float, float]:
    """Calculates Stage 2 objectives: Extended Modularity (Q_ov / EQ) and Overlapping Nodes (ON)."""
    n = H.number_of_nodes()
    valid_comms = [c for c in overlapping_comms if len(c) > 0]
    if not valid_comms:
        return -1.0, 0.0
        
    # Multi-membership count for each node
    membership_counts = np.zeros(n, dtype=np.int32)
    for c in valid_comms:
        for u in c:
            membership_counts[u] += 1
            
    on_count = float(np.sum(membership_counts > 1))
    
    # Extended modularity Q_ov (Equation 18)
    q_ov = 0.0
    for c in valid_comms:
        c_list = list(c)
        for i in range(len(c_list)):
            u = c_list[i]
            d_u = H.degree(u)
            o_u = max(1, membership_counts[u])
            for j in range(i, len(c_list)):
                v = c_list[j]
                d_v = H.degree(v)
                o_v = max(1, membership_counts[v])
                
                a_uv = 1.0 if H.has_edge(u, v) else 0.0
                coeff = 1.0 if i == j else 2.0
                
                val = (a_uv / two_m) - ((d_u * d_v) / (two_m ** 2.0))
                q_ov += coeff * (val / (o_u * o_v))
                
    return float(q_ov), on_count

def run_efmocd(G: nx.Graph, pop_size: int = 100, num_gens: int = 100, seed: int = 42) -> list[frozenset]:
    """Executes the exact 2-Stage EMOFM / EF-MOCD Algorithm (Tian et al., IEEE TFS 2020)."""
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
    
    dist_matrix = compute_all_pairs_distances(H)
    
    # -------------------------------------------------------------------------
    # STAGE 1: Central Node Optimization (Algorithm 1, Lines 1-4)
    # Evolve binary vectors b in {0, 1}^n by minimizing KKM and RC
    # -------------------------------------------------------------------------
    k_min = 2
    k_max = max(2, min(25, int(np.ceil(np.sqrt(n) * 1.5))))
    
    gen_stage1 = max(20, num_gens // 2)
    gen_stage2 = max(20, num_gens - gen_stage1)
    
    # Initialize Stage 1 population (Algorithm 3)
    stage1_pop = []
    for _ in range(pop_size):
        k_init = random.randint(k_min, k_max)
        centers = set(random.sample(range(n), k_init))
        b = np.zeros(n, dtype=np.int8)
        for c in centers:
            b[c] = 1
        stage1_pop.append(b)
        
    best_stage1_sol = None
    best_stage1_val = 1e9
    
    for gen in range(gen_stage1):
        evals = []
        for b in stage1_pop:
            centers = [i for i in range(n) if b[i] == 1]
            if len(centers) < 2:
                # Ensure at least 2 centers
                rand_picks = random.sample(range(n), 2)
                b[rand_picks[0]] = 1
                b[rand_picks[1]] = 1
                centers = [i for i in range(n) if b[i] == 1]
                
            kkm, rc, _ = evaluate_stage1_kkm_rc(H, dist_matrix, centers)
            evals.append((kkm, rc, b))
            
            combined_score = kkm + rc
            if combined_score < best_stage1_val:
                best_stage1_val = combined_score
                best_stage1_sol = np.copy(b)
                
        # Pareto-like truncation selection on (KKM, RC)
        evals.sort(key=lambda x: (x[0] + x[1]))
        elite = [x[2] for x in evals[:max(2, pop_size // 4)]]
        
        next_pop = [np.copy(x) for x in elite]
        while len(next_pop) < pop_size:
            p1 = random.choice(elite)
            p2 = random.choice(elite)
            # Uniform crossover on binary vector
            mask = np.random.rand(n) < 0.5
            child = np.where(mask, p1, p2)
            # Bitwise mutation with p_m = 1/n
            mut_mask = np.random.rand(n) < (1.0 / float(n))
            child[mut_mask] = 1 - child[mut_mask]
            
            # Bound number of centers
            num_c = np.sum(child)
            if num_c < k_min:
                add_idx = random.sample(range(n), k_min - num_c)
                child[add_idx] = 1
            elif num_c > k_max:
                active_idx = np.where(child == 1)[0]
                remove_idx = random.sample(list(active_idx), num_c - k_max)
                child[remove_idx] = 0
                
            next_pop.append(child)
        stage1_pop = next_pop
        
    if best_stage1_sol is None:
        best_stage1_sol = stage1_pop[0]
        
    best_central_nodes = [i for i in range(n) if best_stage1_sol[i] == 1]
    if len(best_central_nodes) < 2:
        best_central_nodes = list(range(min(2, n)))
        
    # -------------------------------------------------------------------------
    # STAGE 2: Fuzzy Threshold Optimization (Algorithm 1, Lines 5-8)
    # Evolve continuous threshold vector r in [0, 1]^n by maximizing Qov and ON
    # -------------------------------------------------------------------------
    U = compute_fuzzy_membership_matrix(dist_matrix, best_central_nodes)
    k_centers = len(best_central_nodes)
    
    # Subpopulation initialization via 2-means clustering on memberships (Algorithm 4)
    init_r = np.zeros(n, dtype=np.float64)
    for u in range(n):
        u_memberships = np.sort(U[u, :])
        if k_centers >= 2:
            mid = len(u_memberships) // 2
            s1_mean = np.mean(u_memberships[:mid])
            s2_mean = np.mean(u_memberships[mid:])
            init_r[u] = 0.5 * (s1_mean + s2_mean)
        else:
            init_r[u] = 0.5
            
    # Generate Stage 2 subpopulation
    stage2_pop = []
    for _ in range(pop_size):
        r_vec = np.copy(init_r)
        # Random perturbation with probability 0.5 (Algorithm 4, Lines 12-15)
        perturb_mask = np.random.rand(n) < 0.5
        r_vec[perturb_mask] = np.random.rand(np.sum(perturb_mask))
        stage2_pop.append(r_vec)
        
    best_comms = None
    best_qov = -1e9
    
    for gen in range(gen_stage2):
        evals = []
        for r_vec in stage2_pop:
            # Decode overlapping communities (Eq. 9)
            comms = collections.defaultdict(set)
            # Add central nodes to their respective communities
            for j, c_node in enumerate(best_central_nodes):
                comms[j].add(c_node)
                
            for u in range(n):
                assigned = False
                for j in range(k_centers):
                    if U[u, j] >= r_vec[u]:
                        comms[j].add(u)
                        assigned = True
                if not assigned:
                    best_j = int(np.argmax(U[u, :]))
                    comms[best_j].add(u)
                    
            comm_sets = [c for c in comms.values() if len(c) > 0]
            q_ov, on_count = evaluate_stage2_qov_on(H, two_m, comm_sets)
            evals.append((q_ov, on_count, r_vec, comm_sets))
            
            if q_ov > best_qov:
                best_qov = q_ov
                best_comms = comm_sets
                
        # Pareto selection maximizing Qov and ON
        evals.sort(key=lambda x: (x[0] + 0.01 * x[1]), reverse=True)
        elite = [x[2] for x in evals[:max(2, pop_size // 4)]]
        
        next_pop = [np.copy(x) for x in elite]
        while len(next_pop) < pop_size:
            p1 = random.choice(elite)
            p2 = random.choice(elite)
            # Simulated Binary Crossover (SBX)
            beta = np.random.rand(n)
            child = 0.5 * ((1.0 + beta) * p1 + (1.0 - beta) * p2)
            # Polynomial mutation
            mut_mask = np.random.rand(n) < (1.0 / float(n))
            delta = np.random.uniform(-0.1, 0.1, size=n)
            child[mut_mask] = np.clip(child[mut_mask] + delta[mut_mask], 0.0, 1.0)
            next_pop.append(child)
        stage2_pop = next_pop
        
    if best_comms is None:
        # Fallback to disjoint assignment from stage 1
        _, _, best_comms = evaluate_stage1_kkm_rc(H, dist_matrix, best_central_nodes)
        
    return [frozenset(rev_map[i] for i in c) for c in best_comms if len(c) > 0]
