use rustc_hash::FxHashSet;
use std::time::Instant;
use crate::core::graph::{Graph, NodeId};

pub type Clique = Vec<NodeId>;

pub fn find_maximal_cliques(graph: &Graph) -> Vec<Clique> {
    let start_time = Instant::now();
    println!("[MCMOEA Rust] Maximal clique extraction started (nodes={}, edges={})...", graph.num_nodes(), graph.num_edges());

    let mut cliques = Vec::new();
    let mut r = Vec::new();
    let mut p: FxHashSet<NodeId> = graph.nodes.iter().copied().collect();
    let mut x = FxHashSet::default();

    // Pre-build neighbor sets for O(1) set operations
    let nbr_sets: rustc_hash::FxHashMap<NodeId, FxHashSet<NodeId>> = graph.nodes.iter()
        .map(|&u| {
            let set: FxHashSet<NodeId> = graph.adjacency_list.get(&u)
                .map(|nbrs| nbrs.iter().copied().collect())
                .unwrap_or_default();
            (u, set)
        })
        .collect();

    bron_kerbosch_pivot_fast(graph, &nbr_sets, &mut r, &mut p, &mut x, &mut cliques);

    // Cap maximal cliques at 5,000 largest cliques to keep MCMOEA precomputed matrix under 100MB RAM
    if cliques.len() > 5000 {
        cliques.sort_by(|a, b| b.len().cmp(&a.len()));
        cliques.truncate(5000);
    }

    let mut covered_nodes = FxHashSet::default();
    for clique in &cliques {
        for &node in clique {
            covered_nodes.insert(node);
        }
    }

    for &node in &graph.nodes {
        if !covered_nodes.contains(&node) {
            cliques.push(vec![node]);
        }
    }

    let elapsed = start_time.elapsed();
    println!("[MCMOEA Rust] Maximal clique extraction completed: {} cliques retained in {:.3?} s", cliques.len(), elapsed);

    cliques
}

const MAX_CLIQUE_SIZE: usize = 15;

fn bron_kerbosch_pivot_fast(
    graph: &Graph,
    nbr_sets: &rustc_hash::FxHashMap<NodeId, FxHashSet<NodeId>>,
    r: &mut Vec<NodeId>,
    p: &mut FxHashSet<NodeId>,
    x: &mut FxHashSet<NodeId>,
    cliques: &mut Vec<Clique>,
) {
    if r.len() >= MAX_CLIQUE_SIZE || (p.is_empty() && x.is_empty()) {
        if r.len() >= 2 {
            cliques.push(r.clone());
        }
        return;
    }

    if p.is_empty() {
        return;
    }

    let pivot = p.union(x)
        .max_by_key(|&&u| {
            nbr_sets.get(&u).map(|s| s.intersection(p).count()).unwrap_or(0)
        })
        .copied()
        .unwrap();

    let empty_set = FxHashSet::default();
    let pivot_nbrs = nbr_sets.get(&pivot).unwrap_or(&empty_set);
    let candidates: Vec<NodeId> = p.difference(pivot_nbrs).copied().collect();

    for v in candidates {
        r.push(v);
        let v_nbrs = nbr_sets.get(&v).unwrap_or(&empty_set);

        let mut p_next: FxHashSet<NodeId> = p.intersection(v_nbrs).copied().collect();
        let mut x_next: FxHashSet<NodeId> = x.intersection(v_nbrs).copied().collect();

        bron_kerbosch_pivot_fast(graph, nbr_sets, r, &mut p_next, &mut x_next, cliques);

        r.pop();
        p.remove(&v);
        x.insert(v);
    }
}
