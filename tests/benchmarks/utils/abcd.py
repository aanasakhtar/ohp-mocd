"""
tests/benchmarks/utils/abcd.py

Native Python generator for ABCD+o^2 (Artificial Benchmark for Community Detection
with Outliers and Overlapping Communities).

Reference:
    Barrett, J., DeWolfe, R., Kamiński, B., Prałat, P., Smith, A., & Théberge, F. (2026).
    The artificial benchmark for community detection with outliers and overlapping communities (ABCD+o^2).
    Journal of Complex Networks, 14(4), cnag023. https://doi.org/10.1093/comnet/cnag023
    Preprint: arXiv:2506.05486.
"""

from typing import List, Set, Dict, Tuple
import numpy as np
import networkx as nx
from scipy.spatial import KDTree


def sample_trunc_powerlaw(
    tau: float,
    x_min: int,
    x_max: int,
    size: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Sample from a discrete truncated power-law distribution:
    P(x) proportional to x^(-tau) for x in [x_min, x_max].
    """
    xs = np.arange(x_min, x_max + 1)
    weights = xs.astype(float) ** (-tau)
    probs = weights / weights.sum()
    return rng.choice(xs, size=size, p=probs)


def generate_abcd_benchmark(
    n: int = 300,
    tau1: float = 2.5,
    tau2: float = 1.5,
    d_min: int = 5,
    d_max: int = 30,
    c_min: int = 20,
    c_max: int = 60,
    xi: float = 0.2,
    eta: float = 1.5,
    dim: int = 2,
    rho: float = 0.0,
    n_outliers: int = 0,
    seed: int = 42
) -> Tuple[nx.Graph, List[frozenset], Dict[int, List[int]]]:
    """
    Generate an ABCD+o^2 synthetic complex network with overlapping communities.

    Parameters:
        n: Number of vertices in the network.
        tau1: Power-law exponent for degree distribution (gamma).
        tau2: Power-law exponent for community size distribution (beta).
        d_min: Minimum vertex degree (delta).
        d_max: Maximum vertex degree (Delta).
        c_min: Minimum community size after growth (s).
        c_max: Maximum community size after growth (S).
        xi: Background mixing parameter in [0.0, 1.0], fraction of edges between communities.
        eta: Overlap multiplier (average number of community memberships per non-outlier node >= 1.0).
        dim: Dimensionality of the latent geometric reference layer.
        rho: Target Pearson correlation between degree and number of memberships.
        n_outliers: Number of outlier vertices belonging to no community (s_0).
        seed: Random number generator seed.

    Returns:
        (G, ground_truth_communities, node_to_communities_map)
    """
    rng = np.random.default_rng(seed)

    # --- Phase 1: Degree Sequence ---
    for _ in range(1000):
        degs = sample_trunc_powerlaw(tau1, d_min, d_max, n, rng)
        if degs.sum() % 2 == 0:
            break
    else:
        degs[np.argmax(degs)] -= 1
    degs = np.sort(degs)[::-1]

    # --- Phase 2: Outlier Assignment ---
    n_outliers = max(0, min(n - 1, n_outliers))
    n_non_outliers = n - n_outliers
    outlier_nodes = set(range(n_non_outliers, n))

    # --- Phase 3: Creating Overlapping Communities via Geometric Reference Layer ---
    target_c_min = max(2, int(np.ceil(c_min / eta)))
    target_c_max = max(target_c_min, int(np.floor(c_max / eta)))

    comm_sizes = []
    curr_sum = 0
    while curr_sum < n_non_outliers:
        s = sample_trunc_powerlaw(tau2, target_c_min, target_c_max, 1, rng)[0]
        if curr_sum + s <= n_non_outliers:
            comm_sizes.append(s)
            curr_sum += s
        else:
            diff = n_non_outliers - curr_sum
            if diff >= target_c_min:
                comm_sizes.append(diff)
            else:
                comm_sizes[-1] += diff
            break

    # Sample geometric coordinates in unit d-ball
    points = rng.standard_normal((n_non_outliers, dim))
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    radii = rng.uniform(0.0, 1.0, (n_non_outliers, 1)) ** (1.0 / dim)
    points = (points / norms) * radii

    kdtree = KDTree(points)

    # Form primary communities: anchor at furthest point, take nearest unassigned points
    assigned_primary: Set[int] = set()
    dist_from_origin = np.sum(points ** 2, axis=1)
    primary_comms: List[Set[int]] = []

    order = rng.permutation(len(comm_sizes))
    for c_idx in order:
        target_size = comm_sizes[c_idx]
        available = [i for i in range(n_non_outliers) if i not in assigned_primary]
        if not available:
            break
        furthest = available[int(np.argmax(dist_from_origin[available]))]
        _, nn_indices = kdtree.query(points[furthest], k=min(n_non_outliers, target_size * 5))
        chosen: List[int] = []
        for idx in nn_indices:
            if idx not in assigned_primary:
                chosen.append(int(idx))
                assigned_primary.add(int(idx))
                if len(chosen) == target_size:
                    break
        primary_comms.append(set(chosen))

    # Community growth: each community grows to round(s * eta) by adding nearest neighbors to centroid
    full_comms: List[Set[int]] = []
    for c_set in primary_comms:
        target_grown = int(np.round(len(c_set) * eta))
        if target_grown <= len(c_set):
            full_comms.append(set(c_set))
            continue
        c_pts = points[list(c_set)]
        centroid = np.mean(c_pts, axis=0)
        _, nn_indices = kdtree.query(centroid, k=min(n_non_outliers, target_grown * 4))
        grown = set(c_set)
        for idx in nn_indices:
            grown.add(int(idx))
            if len(grown) >= target_grown:
                break
        full_comms.append(grown)

    # Build node-to-communities dictionary
    node_to_comms: Dict[int, List[int]] = {i: [] for i in range(n)}
    for cid, c_nodes in enumerate(full_comms):
        for u in c_nodes:
            node_to_comms[u].append(cid)

    # --- Phase 4 & 5: Degrees & Edge Generation via Configuration Model ---
    G = nx.Graph()
    G.add_nodes_from(range(n))

    comm_stubs: Dict[int, List[int]] = {cid: [] for cid in range(len(full_comms))}
    bg_stubs: List[int] = []

    for u in range(n):
        d_u = int(degs[u])
        if u in outlier_nodes or len(node_to_comms[u]) == 0:
            bg_stubs.extend([u] * d_u)
            continue

        # Split into internal and background degrees
        internal_d = int(np.round(d_u * (1.0 - xi)))
        bg_d = d_u - internal_d
        bg_stubs.extend([u] * bg_d)

        # Distribute internal half-edges among the node's communities
        n_comms_u = len(node_to_comms[u])
        share = internal_d // n_comms_u
        rem = internal_d % n_comms_u
        shuffled_c = rng.permutation(node_to_comms[u])
        for idx, cid in enumerate(shuffled_c):
            count = share + (1 if idx < rem else 0)
            comm_stubs[int(cid)].extend([u] * count)

    # Generate intra-community edges
    for cid, stubs in comm_stubs.items():
        if len(stubs) < 2:
            continue
        stubs_arr = np.array(stubs)
        rng.shuffle(stubs_arr)
        if len(stubs_arr) % 2 != 0:
            bg_stubs.append(int(stubs_arr[-1]))
            stubs_arr = stubs_arr[:-1]
        for i in range(0, len(stubs_arr), 2):
            u, v = int(stubs_arr[i]), int(stubs_arr[i+1])
            if u != v:
                G.add_edge(u, v)

    # Generate background edges
    bg_arr = np.array(bg_stubs)
    rng.shuffle(bg_arr)
    if len(bg_arr) % 2 != 0:
        bg_arr = bg_arr[:-1]
    for i in range(0, len(bg_arr), 2):
        u, v = int(bg_arr[i]), int(bg_arr[i+1])
        if u != v:
            G.add_edge(u, v)

    # Remove self-loops
    G.remove_edges_from(list(nx.selfloop_edges(G)))

    ground_truth = [frozenset(c) for c in full_comms if len(c) > 0]
    return G, ground_truth, node_to_comms
