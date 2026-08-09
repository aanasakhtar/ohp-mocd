"""Integration tests for OHP-MOCD (Top-K overlapping community detection)."""

import networkx as nx
import pymocd


def two_cliques_with_boundary_node():
    """Builds a graph with two 4-cliques connected via a boundary node 4."""
    G = nx.Graph()
    # Clique 1: {0, 1, 2, 3}
    for i in range(4):
        for j in range(i + 1, 4):
            G.add_edge(i, j)
    # Clique 2: {5, 6, 7, 8}
    for i in range(5, 9):
        for j in range(i + 1, 9):
            G.add_edge(i, j)
    # Boundary node 4 connected to {0, 1, 2} in Clique 1 and {5, 6, 7} in Clique 2
    for u in [0, 1, 2]:
        G.add_edge(4, u)
    for u in [5, 6, 7]:
        G.add_edge(4, u)
    return G


def test_ohpmocd_crisp_mode():
    G = two_cliques_with_boundary_node()
    part = pymocd.ohpmocd(G, init_strategy="crisp", seed=42)

    assert isinstance(part, dict)
    for node in G.nodes():
        assert node in part
        assert isinstance(part[node], list)
        assert len(part[node]) >= 1


def test_ohpmocd_overlapping_mode_dynamic():
    G = two_cliques_with_boundary_node()
    part = pymocd.ohpmocd(G, init_strategy="boundary_seeded", seed=42)

    assert isinstance(part, dict)
    for node in G.nodes():
        assert node in part
        assert isinstance(part[node], list)
        assert len(part[node]) >= 1
        for c in part[node]:
            assert isinstance(c, int)


def test_ohpmocd_overlapping_mode_three_cliques():
    G = two_cliques_with_boundary_node()
    # Add a 3rd clique connected to boundary node 4
    for i in range(10, 14):
        for j in range(i + 1, 14):
            G.add_edge(i, j)
    for u in [10, 11, 12]:
        G.add_edge(4, u)

    part = pymocd.ohpmocd(G, init_strategy="boundary_seeded", seed=42)
    assert isinstance(part, dict)
    for node in G.nodes():
        assert node in part
        assert isinstance(part[node], list)
        assert len(part[node]) >= 1
        for c in part[node]:
            assert isinstance(c, int)


def test_ohpmocd_class_pareto_front():
    G = two_cliques_with_boundary_node()
    alg = pymocd.OhpMocd(G, seed=42)
    front = alg.generate_pareto_front()

    assert isinstance(front, list)
    assert len(front) > 0
    for part, objs in front:
        assert isinstance(part, dict)
        assert len(objs) == 3
        for node in G.nodes():
            assert node in part
            assert isinstance(part[node], list)
            assert len(part[node]) >= 1


if __name__ == "__main__":
    test_ohpmocd_crisp_mode()
    test_ohpmocd_overlapping_mode_dynamic()
    test_ohpmocd_overlapping_mode_three_cliques()
    test_ohpmocd_class_pareto_front()
    print("All dynamic OHP-MOCD overlap python tests passed successfully!")
