"""
algorithms/cpm.py — Clique Percolation Method (CPM) with NCN recovery.
Reference: Palla et al. (2005) Nature.
"""

from __future__ import annotations
import time
from collections import Counter
import networkx as nx


def _clique_percolation(G: nx.Graph, k: int) -> list[frozenset]:
    cliques = [c for c in nx.find_cliques(G) if len(c) >= k]
    if not cliques:
        return []

    clique_graph = nx.Graph()
    for i, c1 in enumerate(cliques):
        clique_graph.add_node(i)
        set_c1 = set(c1)
        for j in range(i + 1, len(cliques)):
            c2 = cliques[j]
            if len(set_c1.intersection(c2)) >= k - 1:
                clique_graph.add_edge(i, j)

    communities = []
    for component in nx.connected_components(clique_graph):
        community_nodes = set()
        for clique_idx in component:
            community_nodes.update(cliques[clique_idx])
        if community_nodes:
            communities.append(frozenset(community_nodes))

    return communities


def _recover_unassigned_nodes(
    G: nx.Graph,
    communities: list[frozenset],
) -> list[frozenset]:
    all_nodes = set(G.nodes())
    assigned_nodes = set().union(*communities) if communities else set()
    unassigned = all_nodes - assigned_nodes

    if not unassigned:
        return communities

    mutable_communities = [set(c) for c in communities]

    for node in unassigned:
        neighbours = set(G.neighbors(node))
        best_community_idx = -1
        max_connections = -1

        for idx, comm in enumerate(mutable_communities):
            connections = len(neighbours & comm)
            if connections > max_connections:
                max_connections = connections
                best_community_idx = idx

        if best_community_idx >= 0 and max_connections > 0:
            mutable_communities[best_community_idx].add(node)
        else:
            mutable_communities.append({node})

    return [frozenset(c) for c in mutable_communities if c]


def run_cpm_ncn_fixed(
    G: nx.Graph,
    k_values: list[int] = [3, 4, 5],
) -> tuple[list[frozenset], float, int]:
    t0 = time.perf_counter()
    best_k = k_values[0]
    best_modularity = -2.0
    best_partition: list[frozenset] = []

    print(f"[CPM NCN-Fixed] Trying k values: {k_values} ...")
    for k in k_values:
        partition = _clique_percolation(G, k)
        if partition:
            mod = nx.community.modularity(G, partition)
        else:
            mod = -1.0
        if mod > best_modularity:
            best_modularity = mod
            best_k = k
            best_partition = partition

    partition = _recover_unassigned_nodes(G, best_partition)
    runtime = time.perf_counter() - t0

    node_counts: Counter = Counter()
    for community in partition:
        for node in community:
            node_counts[node] += 1
    overlapping = sum(1 for c in node_counts.values() if c > 1)

    print(
        f"[CPM NCN-Fixed] k={best_k} | "
        f"communities={len(partition)} | "
        f"overlapping_nodes={overlapping} | "
        f"all_nodes_covered=True | "
        f"runtime={runtime:.2f}s"
    )

    return partition, runtime, best_k
