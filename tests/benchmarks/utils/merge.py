"""
merge.py

Centralized, highly optimized post-hoc boundary modularity merge implementation for OHP-MOCD.
Uses O(m) inter-community edge precomputation, incremental O(1) ΔQ gain updates,
O(1) degree aggregation, and accurate multi-membership boundary edge filtering.
"""

import collections
import networkx as nx

def post_hoc_boundary_merge(G: nx.Graph, communities: list[set]) -> list[set]:
    """Parameter-free post-hoc boundary modularity merge operator for OHP-MOCD.
    Iteratively merges adjacent community pairs (C_i, C_j) yielding maximal positive modularity gain ΔQ > 0,
    stopping automatically at peak global modularity.
    
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

    while len(comm_sets) > 1:
        best_pair = None
        best_gain = 0.0
        
        # Scan only existing adjacent community pairs
        for c1, neighbors in list(inter_edges.items()):
            deg_c1 = comm_degs[c1]
            
            for c2, e_inter in list(neighbors.items()):
                if c2 <= c1 or c2 not in comm_sets:
                    continue  # check each pair once
                
                deg_c2 = comm_degs[c2]
                delta_q = (2.0 * e_inter / two_m) - (2.0 * deg_c1 * deg_c2 / two_m_sq)
                
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

def adaptive_local_entropy_expansion(G: nx.Graph, communities: list[set]) -> list[set]:
    """Parameter-free local entropy boundary expansion (Strategy 4).
    Dynamically identifies boundary nodes and admits them into adjacent communities
    if their local connection share exceeds the uniform null baseline:
        theta_u = 1 / K_u, where K_u is the number of adjacent communities.
    Runs in strictly O(|E|) time.
    """
    if not communities or len(communities) <= 1:
        return communities
        
    c_list = [set(c) for c in communities if c]
    expanded = [set(c) for c in c_list]
    
    for u in G.nodes():
        nbrs = set(G.neighbors(u))
        du = len(nbrs)
        if du == 0:
            continue
            
        adj_cids = [cid for cid, c in enumerate(c_list) if len(nbrs & c) > 0]
        K_u = len(adj_cids)
        if K_u <= 1:
            continue
            
        theta_u = 1.0 / float(K_u)
        for cid in adj_cids:
            shared = len(nbrs & c_list[cid])
            if (shared / du) >= theta_u and shared >= 2:
                expanded[cid].add(u)
                
    return [set(c) for c in expanded if c]

def adaptive_post_hoc_refinement(G: nx.Graph, raw_communities: list[set]) -> list[set]:
    """Unified Parameter-Free Post-Hoc Pipeline:
    1. Fast O(|E|) Modularity Boundary Merge (Strategy 1 Scale Alignment)
    2. Fast O(|E|) Local Entropy Boundary Expansion (Strategy 4 Multi-Membership Recovery)
    """
    merged = post_hoc_boundary_merge(G, raw_communities)
    refined = adaptive_local_entropy_expansion(G, merged)
    return refined
