"""
merge.py

Centralized, highly optimized post-hoc boundary modularity merge implementation for OHP-MOCD.
Uses O(m) inter-community edge precomputation, incremental O(1) ΔQ gain updates,
O(1) degree aggregation, and accurate multi-membership boundary edge filtering.
"""

import collections
import networkx as nx

def post_hoc_boundary_merge(G: nx.Graph, communities: list[set], merge_threshold: float | str | None = 0.50) -> list[set]:
    """Fast, optimized post-hoc boundary modularity merge operator.
    Supports parameter-free automatic peak modularity merge when merge_threshold is 'auto', None, or 0.0.
    
    Inter-Community Edge Definition for Overlapping Nodes:
      An edge (u, v) is counted as an inter-community edge between C_i and C_j (C_i != C_j)
      ONLY IF u and v share NO common community membership (set(u_comms) & set(v_comms) == set()).
      If u and v share a community, edge (u, v) is an intra-community edge for that shared community
      and does NOT inflate inter-community merge eagerness.
    """
    if not communities or len(communities) <= 1:
        return communities
    m = G.number_of_edges()
    if m == 0:
        return communities
    
    two_m = 2.0 * m
    two_m_sq = two_m * two_m
    deg = dict(G.degree())
    
    # Filter empty communities and assign integer IDs
    comm_sets = {i: set(c) for i, c in enumerate(communities) if c}
    if len(comm_sets) <= 1:
        return list(comm_sets.values())
        
    # Map node -> list of community IDs it belongs to
    node_to_comms = collections.defaultdict(list)
    for cid, cset in comm_sets.items():
        for u in cset:
            node_to_comms[u].append(cid)
            
    # Precompute inter-community edge counts: inter_edges[c1][c2]
    inter_edges = collections.defaultdict(lambda: collections.defaultdict(int))
    for u, v in G.edges():
        u_comms = node_to_comms.get(u, [])
        v_comms = node_to_comms.get(v, [])
        
        # If u and v share any common community, this is an intra-community edge for that shared community
        if set(u_comms) & set(v_comms):
            continue
            
        for c1 in u_comms:
            for c2 in v_comms:
                if c1 != c2:
                    inter_edges[c1][c2] += 1
                    inter_edges[c2][c1] += 1

    # Precompute total degree per community: sum(deg[u] for u in C)
    comm_degs = {cid: sum(deg.get(u, 0) for u in cset) for cid, cset in comm_sets.items()}
    
    is_auto = (merge_threshold == 'auto' or merge_threshold is None or merge_threshold == 0.0)
    thresh_val = 0.0 if is_auto else float(merge_threshold)

    while len(comm_sets) > 1:
        best_pair = None
        best_gain = 0.0
        
        # Scan only existing adjacent community pairs
        for c1, neighbors in list(inter_edges.items()):
            deg_c1 = comm_degs[c1]
            size_c1 = len(comm_sets[c1])
            
            for c2, e_inter in list(neighbors.items()):
                if c2 <= c1 or c2 not in comm_sets:
                    continue  # check each pair once
                
                deg_c2 = comm_degs[c2]
                delta_q = (2.0 * e_inter / two_m) - (2.0 * deg_c1 * deg_c2 / two_m_sq)
                
                if delta_q > 0.0:
                    if not is_auto:
                        size_c2 = len(comm_sets[c2])
                        min_size = min(size_c1, size_c2)
                        bound_ratio = e_inter / min_size if min_size > 0 else 0.0
                        if bound_ratio < thresh_val:
                            continue
                    
                    if delta_q > best_gain:
                        best_gain = delta_q
                        best_pair = (c1, c2)
                        
        if best_pair is None or best_gain <= 0.0:
            break
            
        c1, c2 = best_pair
        
        # Merge c2 into c1
        comm_sets[c1] |= comm_sets[c2]
        # O(1) degree aggregation update
        comm_degs[c1] += comm_degs[c2]
        
        # Update inter-community edge counts
        c2_neighbors = dict(inter_edges[c2])
        for k, w in c2_neighbors.items():
            if k == c1:
                continue
            inter_edges[c1][k] += w
            inter_edges[k][c1] += w
            if c2 in inter_edges[k]:
                del inter_edges[k][c2]
                
        if c2 in inter_edges[c1]:
            del inter_edges[c1][c2]
        if c1 in inter_edges[c2]:
            del inter_edges[c2][c1]
            
        del inter_edges[c2]
        del comm_sets[c2]
        del comm_degs[c2]

    return list(comm_sets.values())
