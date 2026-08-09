//! Objective evaluation for OHP-MOCD.
//! Decomposed modularity (intra, inter) extended to Top-K overlapping memberships
//! via fractional membership weights r_{v,c} = 1 / |M(v)|.
//! Single-pass edge iteration for HP-MOCD level performance (~10s on 10k nodes).
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

/// Calculates Shi-style decomposed modularity (intra, inter) for Top-K overlapping partitions in 1 pass over edges.
pub fn calculate_ohp_objectives(
    graph: &Graph,
    partition: &OhpPartition,
    degrees: &HashMap<NodeId, usize, FxBuildHasher>,
) -> (f64, f64) {
    let total_edges = graph.edges.len() as f64;
    if total_edges == 0.0 {
        return (0.0, 0.0);
    }

    let mut community_degrees: FxHashMap<CommunityId, f64> = FxHashMap::default();
    for (&node, membership) in partition.iter() {
        let deg = *degrees.get(&node).unwrap_or(&0) as f64;
        let weight = deg / membership.len() as f64;
        for &comm in &membership.communities {
            *community_degrees.entry(comm).or_insert(0.0) += weight;
        }
    }

    let total_edges_doubled = 2.0 * total_edges;
    let mut inter_sum = 0.0;
    for &comm_deg in community_degrees.values() {
        inter_sum += (comm_deg / total_edges_doubled).powi(2);
    }

    let mut intra_sum = 0.0;
    for (u, v) in &graph.edges {
        if let (Some(m_u), Some(m_v)) = (partition.get(u), partition.get(v)) {
            let w_u = 1.0 / m_u.len() as f64;
            let w_v = 1.0 / m_v.len() as f64;
            for &c_u in &m_u.communities {
                if m_v.contains(c_u) {
                    intra_sum += w_u * w_v;
                }
            }
        }
    }

    let intra = 1.0 - (intra_sum / total_edges);
    let inter = inter_sum;
    (intra, inter)
}

/// Calculates the 3rd objective f3: Parameter-Free Intrinsic Overlap Cohesion Penalty
/// Unsupported Overlap Penalty = avg(1.0 - s_min / s_max) across overlapping nodes.
pub fn calculate_f3_objective(
    graph: &Graph,
    partition: &OhpPartition,
) -> f64 {
    let total_nodes = graph.nodes.len() as f64;
    if total_nodes == 0.0 {
        return 0.0;
    }

    let mut overlapping_nodes_count = 0.0;
    let mut total_unsupported_penalty = 0.0;

    for (&node, membership) in partition.iter() {
        if membership.len() > 1 {
            overlapping_nodes_count += 1.0;

            if let Some(neighbors) = graph.adjacency_list.get(&node) {
                let deg = neighbors.len() as f64;
                if deg > 0.0 {
                    let mut s_min = 1.0;
                    let mut s_max = 0.0;
                    for &comm in &membership.communities {
                        let count = neighbors.iter()
                            .filter(|&&nbr| {
                                partition.get(&nbr).map_or(false, |m| m.contains(comm))
                            })
                            .count() as f64;
                        let s = count / deg;
                        if s < s_min { s_min = s; }
                        if s > s_max { s_max = s; }
                    }
                    if s_max > 0.0 {
                        let support_ratio = s_min / s_max;
                        total_unsupported_penalty += 1.0 - support_ratio;
                    }
                }
            }
        }
    }

    if overlapping_nodes_count > 0.0 {
        total_unsupported_penalty / overlapping_nodes_count
    } else {
        0.0
    }
}

pub fn evaluate_ohp_population(
    individuals: &mut [OhpIndividual],
    graph: &Graph,
    degrees: &HashMap<NodeId, usize, FxBuildHasher>,
    enable_f3: bool,
    phase1_active: bool,
) {
    individuals.par_iter_mut().for_each(|ind| {
        let (intra, inter) = calculate_ohp_objectives(graph, &ind.partition, degrees);
        if enable_f3 {
            let f3 = if phase1_active {
                0.0
            } else {
                calculate_f3_objective(graph, &ind.partition)
            };
            ind.objectives = vec![intra, inter, f3];
        } else {
            ind.objectives = vec![intra, inter];
        }
    });
}
