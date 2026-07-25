//! Objective evaluation for OHP-MOCD.
//! Decomposed modularity (intra, inter) extended to Top-K overlapping memberships
//! via fractional membership weights r_{v,c} = 1 / |M(v)|.
//! Reduces identically to Shi's crisp decomposed modularity when |M(v)| = 1.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos.

use crate::core::algorithms::ohpmocd::individual::{OhpIndividual, OhpPartition};
use crate::core::graph::{CommunityId, Graph, NodeId, Partition};
use crate::core::metaheuristics::helpers::individual::Individual;
use crate::core::metaheuristics::helpers::objectives::decomposed_modularity::calculate_objectives;
use rayon::prelude::*;
use rustc_hash::{FxBuildHasher, FxHashMap};
use std::collections::HashMap;

#[allow(dead_code)]
pub fn evaluate_crisp_population(
    individuals: &mut [Individual],
    graph: &Graph,
    degrees: &HashMap<i32, usize, FxBuildHasher>,
) {
    individuals.par_iter_mut().for_each(|ind| {
        let metrics = calculate_objectives(graph, &ind.partition, degrees, true);
        ind.objectives = vec![metrics.intra, metrics.inter];
    });
}

#[allow(dead_code)]
pub fn evaluate_crisp_partition(
    graph: &Graph,
    partition: &Partition,
    degrees: &HashMap<i32, usize, FxBuildHasher>,
) -> (f64, f64) {
    let metrics = calculate_objectives(graph, partition, degrees, false);
    (metrics.intra, metrics.inter)
}

/// Calculates Shi-style decomposed modularity (intra, inter) for Top-K overlapping partitions.
/// When all nodes have 1 membership, returns identical results to crisp decomposed modularity.
pub fn calculate_ohp_objectives(
    graph: &Graph,
    partition: &OhpPartition,
    degrees: &HashMap<NodeId, usize, FxBuildHasher>,
) -> (f64, f64) {
    let total_edges = graph.edges.len() as f64;
    if total_edges == 0.0 {
        return (0.0, 0.0);
    }

    let mut communities: FxHashMap<CommunityId, Vec<(NodeId, f64)>> = FxHashMap::default();
    for (&node, membership) in partition.iter() {
        let weight = 1.0 / membership.len() as f64;
        for &comm in &membership.communities {
            communities.entry(comm).or_default().push((node, weight));
        }
    }

    let total_edges_doubled = 2.0 * total_edges;
    let mut intra_sum = 0.0;
    let mut inter_sum = 0.0;

    for (&comm, nodes) in communities.iter() {
        let mut community_degree = 0.0;
        for &(node, weight) in nodes {
            let deg = *degrees.get(&node).unwrap_or(&0) as f64;
            community_degree += weight * deg;
        }

        let mut community_edges = 0.0;
        for &(node, w_u) in nodes {
            if let Some(neighbors) = graph.adjacency_list.get(&node) {
                for &neighbor in neighbors {
                    if node < neighbor {
                        if let Some(m_v) = partition.get(&neighbor) {
                            let w_v = if m_v.contains(comm) {
                                1.0 / m_v.len() as f64
                            } else {
                                0.0
                            };
                            if w_v > 0.0 {
                                community_edges += w_u * w_v;
                            }
                        }
                    }
                }
            }
        }

        intra_sum += community_edges;
        inter_sum += (community_degree / total_edges_doubled).powi(2);
    }

    let intra = 1.0 - (intra_sum / total_edges);
    let inter = inter_sum;
    (intra, inter)
}

pub fn evaluate_ohp_population(
    individuals: &mut [OhpIndividual],
    graph: &Graph,
    degrees: &HashMap<NodeId, usize, FxBuildHasher>,
) {
    individuals.par_iter_mut().for_each(|ind| {
        let (intra, inter) = calculate_ohp_objectives(graph, &ind.partition, degrees);
        ind.objectives = vec![intra, inter];
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::ohpmocd::individual::crisp_to_ohp;
    use crate::core::graph::Graph;

    fn test_graph() -> Graph {
        let mut g = Graph::new();
        for (a, b) in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)] {
            g.add_edge(a, b);
        }
        g.finalize();
        g
    }

    #[test]
    fn ohp_objectives_matches_crisp_when_max_memberships_is_1() {
        let g = test_graph();
        let degrees = g.precompute_degrees();
        let crisp_part: Partition = [(0, 0), (1, 0), (2, 0), (3, 1), (4, 1), (5, 1)]
            .into_iter()
            .collect();
        let ohp_part = crisp_to_ohp(&crisp_part);

        let (c_intra, c_inter) = evaluate_crisp_partition(&g, &crisp_part, &degrees);
        let (o_intra, o_inter) = calculate_ohp_objectives(&g, &ohp_part, &degrees);

        assert!((c_intra - o_intra).abs() < 1e-10);
        assert!((c_inter - o_inter).abs() < 1e-10);
    }
}
