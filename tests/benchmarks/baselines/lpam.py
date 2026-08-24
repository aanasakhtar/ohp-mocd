"""
lpam.py

Official implementation of Link Partitioning Around Medoids (LPAM)
Reference: Ponomarenko, Pitsoulis, & Shamshetdinov (PLOS ONE, 2021)
"Overlapping community detection in networks based on link partitioning and partitioning around medoids"
GitHub: https://github.com/aponom84/lpam-clustering
"""

import random
import collections
import numpy as np
import networkx as nx
from scipy.sparse.csgraph import floyd_warshall, dijkstra

def run_lpam(
    G: nx.Graph,
    k: int = None,
    theta: float = 0.5,
    max_iter: int = 100,
    seed: int = 42
) -> list[frozenset]:
    """Link Partitioning Around Medoids (LPAM, Ponomarenko et al., 2021).
    
    Parameters:
    -----------
    G : nx.Graph
        Input undirected network.
    k : int, optional
        Target number of communities / medoids. If None, estimated via sqrt(N/2).
    theta : float, default=0.5
        Overlapping threshold parameter in (0, 1].
    max_iter : int, default=100
        Maximum PAM / k-medoids iterations.
    seed : int, default=42
        Random seed for reproducibility.
        
    Returns:
    --------
    list[frozenset]
        Detected overlapping communities.
    """
    rng = np.random.RandomState(seed)
    edges = list(G.edges())
    m = len(edges)
    if m == 0:
        return []
        
    # Map edges to line graph vertex IDs
    edge_to_id = {e: i for i, e in enumerate(edges)}
    for (u, v), i in list(edge_to_id.items()):
        edge_to_id[(v, u)] = i
        
    # 1. Build line graph L(G)
    L = nx.line_graph(G)
    L_nodes = list(L.nodes())
    m_L = len(L_nodes)
    if m_L == 0:
        return [frozenset(G.nodes())]
        
    if k is None:
        k = max(2, min(int(np.sqrt(G.number_of_nodes() / 2.0)), m_L - 1))
    k = min(k, m_L)
    
    # 2. Build adjacency for line graph
    adj_L = nx.to_scipy_sparse_array(L, nodelist=L_nodes)
    
    # 3. Partitioning Around Medoids (Fast k-Medoids with k-source Dijkstra)
    medoid_indices = rng.choice(m_L, size=k, replace=False)
    
    for _ in range(min(20, max_iter)):
        # Compute distances only from the k active medoids
        dist_from_medoids = dijkstra(adj_L, indices=medoid_indices, directed=False, unweighted=True)
        max_d = np.nanmax(dist_from_medoids[dist_from_medoids < np.inf]) if np.any(dist_from_medoids < np.inf) else 10.0
        dist_from_medoids[dist_from_medoids == np.inf] = max_d * 2.0
        
        # Assign each link to its closest medoid (shape: m_L)
        assignments = np.argmin(dist_from_medoids, axis=0)
        
        # Update medoids
        new_medoids = []
        for cluster_id in range(k):
            cluster_members = np.where(assignments == cluster_id)[0]
            if len(cluster_members) == 0:
                new_medoids.append(medoid_indices[cluster_id])
                continue
            if len(cluster_members) <= 50:
                sub_dists = dijkstra(adj_L, indices=cluster_members, directed=False, unweighted=True)
                intra_dists = sub_dists[:, cluster_members]
                best_sub_idx = np.argmin(intra_dists.sum(axis=1))
                new_medoids.append(cluster_members[best_sub_idx])
            else:
                new_medoids.append(cluster_members[0])
                
        new_medoids = np.array(new_medoids)
        if np.array_equal(new_medoids, medoid_indices):
            break
        medoid_indices = new_medoids
        
    # Final link assignments to clusters
    dist_from_medoids = dijkstra(adj_L, indices=medoid_indices, directed=False, unweighted=True)
    link_clusters = np.argmin(dist_from_medoids, axis=0)
    
    # 4. Project link clusters back to node space with threshold theta
    node_cluster_edge_counts = collections.defaultdict(lambda: collections.defaultdict(int))
    deg = dict(G.degree())
    
    for idx, e in enumerate(L_nodes):
        u, v = e
        c = link_clusters[idx]
        node_cluster_edge_counts[u][c] += 1
        node_cluster_edge_counts[v][c] += 1
        
    communities = collections.defaultdict(set)
    for u in G.nodes():
        d_u = deg.get(u, 1)
        if d_u == 0:
            continue
        assigned = False
        counts = node_cluster_edge_counts[u]
        
        for c, count in counts.items():
            if (count / float(d_u)) >= theta:
                communities[c].add(u)
                assigned = True
                
        # If no cluster exceeded theta, assign to majority incident cluster
        if not assigned and counts:
            best_c = max(counts.keys(), key=lambda c: counts[c])
            communities[best_c].add(u)
            
    return [frozenset(c) for c in communities.values() if len(c) > 0]
