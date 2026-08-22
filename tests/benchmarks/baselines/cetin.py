"""
cetin.py

Official implementation of Core-Expansion Overlapping Community Detection
Reference: Çetin & Amrahov (Kybernetika, 2022)
"A new method for overlapping community detection based on core expansion"
"""

import collections
import networkx as nx

def run_cetin(
    G: nx.Graph,
    q_threshold: float = 0.001,
    seed: int = 42
) -> list[frozenset]:
    """Çetin & Amrahov (2022) Core-Expansion Algorithm.
    
    Parameters:
    -----------
    G : nx.Graph
        Input undirected network.
    q_threshold : float, default=0.001
        Modularity gain threshold for expanding boundary nodes.
    seed : int, default=42
        Random seed for reproducibility.
        
    Returns:
    --------
    list[frozenset]
        Detected overlapping communities.
    """
    nodes = list(G.nodes())
    if not nodes:
        return []
        
    degrees = dict(G.degree())
    total_edges = G.number_of_edges()
    if total_edges == 0:
        return [frozenset(nodes)]
        
    # Sort seeds by degree descending
    sorted_seeds = sorted(nodes, key=lambda u: degrees[u], reverse=True)
    comms = []
    
    for seed_node in sorted_seeds:
        cluster = {seed_node}
        candidates = set(G.neighbors(seed_node))
        improved = True
        
        while candidates and improved:
            improved = False
            best_node = None
            best_gain = 0.0
            
            for cand in list(candidates):
                test_comm = cluster | {cand}
                internal_edges = G.subgraph(test_comm).number_of_edges()
                comm_deg = sum(degrees[u] for u in test_comm)
                if comm_deg > 0:
                    eq_gain = (internal_edges / float(total_edges)) - (comm_deg / (2.0 * float(total_edges)))**2
                    if eq_gain > best_gain:
                        best_gain = eq_gain
                        best_node = cand
                        
            if best_node and best_gain > q_threshold:
                cluster.add(best_node)
                candidates.remove(best_node)
                candidates.update(set(G.neighbors(best_node)) - cluster)
                improved = True
                
        if len(cluster) > 1:
            comms.append(frozenset(cluster))
            
    # Deduplicate identical or fully subsumed subsets
    unique_comms = []
    for c in sorted(comms, key=lambda x: len(x), reverse=True):
        if not any(c == existing for existing in unique_comms):
            unique_comms.append(c)
            
    return unique_comms if unique_comms else [frozenset(nodes)]
