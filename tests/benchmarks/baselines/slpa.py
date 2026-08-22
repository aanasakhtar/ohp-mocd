"""
slpa.py

Official implementation of Speaker-Listener Label Propagation Algorithm (SLPA)
Reference: Xie & Szymanski (IEEE TKDE 2011/2012)
"SLPA: Uncovering Overlapping Communities in Social Networks via A Speaker-listener Interaction Dynamic Process"
"""

import random
import collections
import networkx as nx

def run_slpa(
    G: nx.Graph,
    r: float = 0.45,
    t: int = 100,
    seed: int = 42
) -> list[frozenset]:
    """Speaker-Listener Label Propagation Algorithm (SLPA, Xie & Szymanski, 2011).
    
    Parameters:
    -----------
    G : nx.Graph
        Input undirected network.
    r : float, default=0.45
        Post-processing label probability threshold in (0, 1].
    t : int, default=100
        Number of propagation iterations.
    seed : int, default=42
        Random seed for reproducibility.
        
    Returns:
    --------
    list[frozenset]
        Detected overlapping communities.
    """
    rng = random.Random(seed)
    nodes = list(G.nodes())
    if not nodes:
        return []
        
    # Memory buffer: initialized with each node's own ID
    memory = {v: [v] for v in nodes}
    
    for _ in range(t):
        order = list(nodes)
        rng.shuffle(order)
        for listener in order:
            neighbors = list(G.neighbors(listener))
            if not neighbors:
                continue
                
            # Each speaker selects a random label from its memory
            speakers_labels = [rng.choice(memory[speaker]) for speaker in neighbors]
            
            # Listener rule: Count label frequencies and tie-break uniformly at random
            counts = collections.Counter(speakers_labels)
            max_c = max(counts.values())
            candidates = [l for l, c in counts.items() if c == max_c]
            chosen_label = rng.choice(candidates)
            
            memory[listener].append(chosen_label)
            
    # Post-processing stage: threshold r
    communities = collections.defaultdict(set)
    for v in nodes:
        total = len(memory[v])
        counts = collections.Counter(memory[v])
        for l, cnt in counts.items():
            if (cnt / float(total)) >= r:
                communities[l].add(v)
                
    return [frozenset(c) for c in communities.values() if len(c) > 0]
