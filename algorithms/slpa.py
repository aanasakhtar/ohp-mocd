"""
algorithms/slpa.py — Speaker-Listener Label Propagation Algorithm (SLPA).
Reference: Xie, J., Szymanski, B. K., & Liu, X. (2011). SLPA. IEEE ICDM.
"""

from __future__ import annotations
import random
import sys
import time
from collections import Counter
from pathlib import Path
import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import HPMOCD_CONFIG


def _slpa_core(
    G: nx.Graph,
    T: int,
    r: float,
    seed: int | None,
) -> list[frozenset]:
    rng = random.Random(seed)
    nodes = list(G.nodes())
    memory: dict = {v: Counter({v: 1}) for v in nodes}

    for _t in range(T):
        order = list(nodes)
        rng.shuffle(order)

        for listener in order:
            neighbours = list(G.neighbors(listener))
            if not neighbours:
                continue

            received: list = []
            for speaker in neighbours:
                mem = memory[speaker]
                total = sum(mem.values())
                labels = list(mem.keys())
                weights = [mem[l] / total for l in labels]
                chosen = rng.choices(labels, weights=weights, k=1)[0]
                received.append(chosen)

            counts = Counter(received)
            max_count = max(counts.values())
            candidates = [l for l, c in counts.items() if c == max_count]
            adopted = rng.choice(candidates)
            memory[listener][adopted] = memory[listener].get(adopted, 0) + 1

    node_labels: dict = {}
    for v in nodes:
        mem = memory[v]
        total = sum(mem.values())
        kept = {l for l, c in mem.items() if c / total > r}
        if not kept:
            kept = {max(mem, key=lambda l: mem[l])}
        node_labels[v] = kept

    label_to_nodes: dict = {}
    for v, labels in node_labels.items():
        for l in labels:
            label_to_nodes.setdefault(l, set()).add(v)

    communities = [
        frozenset(members)
        for members in label_to_nodes.values()
        if len(members) >= 2
    ]

    assigned = set().union(*communities) if communities else set()
    unassigned = set(nodes) - assigned
    if unassigned:
        for v in unassigned:
            best_cid = None
            best_count = -1
            nbrs = set(G.neighbors(v))
            for cid, comm in enumerate(communities):
                count = len(nbrs & comm)
                if count > best_count:
                    best_count = count
                    best_cid = cid
            if best_cid is not None:
                communities[best_cid] = frozenset(communities[best_cid] | {v})
            else:
                communities.append(frozenset({v}))

    return communities


def run_slpa(
    G: nx.Graph,
    T: int = 20,
    r: float = 0.1,
    seed: int | None = None,
    cfg: dict = HPMOCD_CONFIG,
) -> tuple[list[frozenset], float]:
    t0 = time.perf_counter()
    partition = _slpa_core(G, T=T, r=r, seed=seed)
    runtime = time.perf_counter() - t0

    node_counts: Counter = Counter()
    for community in partition:
        for node in community:
            node_counts[node] += 1
    overlapping = sum(1 for c in node_counts.values() if c > 1)
    assigned = len(node_counts)

    print(
        f"[SLPA] T={T} r={r} | "
        f"communities={len(partition)} | "
        f"overlapping_nodes={overlapping}/{G.number_of_nodes()} | "
        f"assigned={assigned}/{G.number_of_nodes()} | "
        f"runtime={runtime:.2f}s"
    )
    return partition, runtime
