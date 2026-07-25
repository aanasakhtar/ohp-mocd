"""
data/load_lfr.py — Generate LFR benchmark graphs (disjoint & overlapping).

The project uses NetworkX's disjoint LFR generator as the base benchmark.
For the overlapping phase, we build a synthetic overlapping cover on top of
that same graph so the repo stays dependency-light and reproducible across
NetworkX versions.
"""

from collections import Counter
import random

import networkx as nx
from networkx.generators.community import LFR_benchmark_graph
from config import LFR_CONFIG


def _disjoint_communities_from_graph(G: nx.Graph) -> list[frozenset]:
    communities = list({frozenset(G.nodes[v]["community"]) for v in G})
    return communities


def load_lfr_disjoint(cfg: dict = LFR_CONFIG) -> tuple[nx.Graph, list[frozenset]]:
    """Generate a disjoint LFR graph."""
    G = LFR_benchmark_graph(
        n=cfg["n"],
        tau1=cfg["tau1"],
        tau2=cfg["tau2"],
        mu=cfg["mu"],
        average_degree=cfg["average_degree"],
        max_degree=cfg["max_degree"],
        min_community=cfg["min_community"],
        max_community=cfg["max_community"],
        seed=cfg["seed"],
    )

    communities = _disjoint_communities_from_graph(G)

    G_clean = nx.Graph(G)
    nx.set_node_attributes(
        G_clean,
        {v: G.nodes[v]["community"] for v in G},
        "community",
    )
    print(
        f"[LFR disjoint] nodes={G_clean.number_of_nodes()}, "
        f"edges={G_clean.number_of_edges()}, "
        f"communities={len(communities)}"
    )
    return G_clean, communities


def load_lfr_overlapping(cfg: dict = LFR_CONFIG) -> tuple[nx.Graph, list[frozenset]]:
    """Generate a synthetic overlapping LFR benchmark.

    We first generate a standard disjoint LFR graph, then assign a subset of
    nodes to a second community using neighborhood support. This preserves the
    benchmark's structure while giving us overlapping ground truth for Omega
    Index evaluation.
    """
    G, base_communities = load_lfr_disjoint(cfg)

    community_sets = [set(comm) for comm in base_communities]
    node_memberships: dict[int, list[int]] = {
        v: [cid]
        for cid, community in enumerate(community_sets)
        for v in community
    }

    nodes_by_degree = sorted(G.nodes(), key=lambda v: G.degree(v), reverse=True)
    overlap_n = min(int(cfg.get("overlap_n", 0)), len(nodes_by_degree))
    overlap_k = max(2, int(cfg.get("overlap_membership", 2)))
    rng = random.Random(cfg.get("seed", 42))

    for node in nodes_by_degree[:overlap_n]:
        current = set(node_memberships[node])
        extra_needed = overlap_k - len(current)
        if extra_needed <= 0:
            continue

        neighbour_support = Counter()
        for nbr in G.neighbors(node):
            neighbour_support.update(cid for cid in node_memberships[nbr] if cid not in current)

        while extra_needed > 0:
            target = None
            for cid, _count in neighbour_support.most_common():
                if cid not in current:
                    target = cid
                    break

            if target is None:
                available = [cid for cid in range(len(community_sets)) if cid not in current]
                if not available:
                    break
                target = rng.choice(available)

            current.add(target)
            node_memberships[node].append(target)
            community_sets[target].add(node)
            extra_needed -= 1

    G_clean = nx.Graph(G)
    nx.set_node_attributes(
        G_clean,
        {v: set(node_memberships[v]) for v in G_clean.nodes()},
        "community",
    )

    overlapping_nodes = [v for v in G_clean if len(G_clean.nodes[v]["community"]) > 1]
    communities = [frozenset(members) for members in community_sets if members]

    print(
        f"[LFR overlapping] nodes={G_clean.number_of_nodes()}, "
        f"edges={G_clean.number_of_edges()}, "
        f"communities={len(communities)}, "
        f"overlapping_nodes={len(overlapping_nodes)}"
    )
    return G_clean, communities


# Quick sanity check
if __name__ == "__main__":
    G_d, cmty_d = load_lfr_disjoint()
    G_o, cmty_o = load_lfr_overlapping()