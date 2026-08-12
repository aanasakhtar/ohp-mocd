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

/// Computes DWI-proportional soft membership weights r_{v,c} = DWI(v,c) / sum_{c'} DWI(v,c')
pub(crate) fn compute_dwi_membership_weights(
    node: NodeId,
    partition: &OhpPartition,
    graph: &Graph,
    degrees: &HashMap<NodeId, usize, FxBuildHasher>,
) -> FxHashMap<CommunityId, f64> {
    let mut weights = FxHashMap::default();
    let membership = match partition.get(&node) {
        Some(m) if !m.is_empty() => m,
        _ => return weights,
    };

    if membership.len() == 1 {
        weights.insert(membership.primary(), 1.0);
        return weights;
    }

    let mut raw_dwi = FxHashMap::default();
    let mut total_dwi_sum = 0.0;

    if let Some(neighbors) = graph.adjacency_list.get(&node) {
        let mut total_neighbor_deg = 0.0;
        for &neighbor in neighbors {
            let nbr_deg = *degrees.get(&neighbor).unwrap_or(&1) as f64;
            total_neighbor_deg += nbr_deg;
            if let Some(m) = partition.get(&neighbor) {
                for &c in &m.communities {
                    if membership.contains(c) {
                        *raw_dwi.entry(c).or_insert(0.0) += nbr_deg;
                    }
                }
            }
        }
        if total_neighbor_deg > 0.0 {
            for val in raw_dwi.values_mut() {
                *val /= total_neighbor_deg;
                total_dwi_sum += *val;
            }
        }
    }

    if total_dwi_sum > 0.0 {
        for &c in &membership.communities {
            let dwi_val = raw_dwi.get(&c).copied().unwrap_or(0.0);
            weights.insert(c, dwi_val / total_dwi_sum);
        }
    } else {
        let unif = 1.0 / membership.len() as f64;
        for &c in &membership.communities {
            weights.insert(c, unif);
        }
    }

    weights
}
/// Computes OCCSA-proportional soft membership weights r_{v,c} = d_in(v,c) / Σ_{c'} d_in(v,c')
/// where d_in(v,c) = |{u ∈ N(v) : c ∈ M(u)}| — the unweighted neighbor count in community c.
/// This matches the OCCSA membership assignment criterion (Shang et al. 2024) used in the operators,
/// ensuring geometric consistency between membership decisions and objective evaluation.
/// When all communities have zero in-degree (isolated node), falls back to uniform 1/|M(v)|.
pub(crate) fn compute_occsa_membership_weights(
    node: NodeId,
    partition: &OhpPartition,
    graph: &Graph,
) -> FxHashMap<CommunityId, f64> {
    let mut weights = FxHashMap::default();
    let membership = match partition.get(&node) {
        Some(m) if !m.is_empty() => m,
        _ => return weights,
    };

    if membership.len() == 1 {
        weights.insert(membership.primary(), 1.0);
        return weights;
    }

    // Count how many neighbors of v are in each of v's communities (unweighted).
    let mut raw_counts: FxHashMap<CommunityId, f64> = FxHashMap::default();
    if let Some(neighbors) = graph.adjacency_list.get(&node) {
        for &neighbor in neighbors {
            if let Some(m) = partition.get(&neighbor) {
                for &c in &m.communities {
                    if membership.contains(c) {
                        *raw_counts.entry(c).or_insert(0.0) += 1.0;
                    }
                }
            }
        }
    }

    let total_count: f64 = raw_counts.values().sum();
    if total_count > 0.0 {
        for &c in &membership.communities {
            let count = raw_counts.get(&c).copied().unwrap_or(0.0);
            weights.insert(c, count / total_count);
        }
    } else {
        // Fallback: uniform weights when node has no covered neighbors.
        let unif = 1.0 / membership.len() as f64;
        for &c in &membership.communities {
            weights.insert(c, unif);
        }
    }

    weights
}

/// Calculates Shi-style decomposed modularity (intra, inter) using OCCSA-proportional soft membership weights.
/// OCCSA weights r_{v,c} = d_in(v,c)/Σ_c' d_in(v,c') match the unweighted membership assignment operators,
/// ensuring f1/f2 are geometrically consistent with how NSGA-II assigns community memberships.
pub fn calculate_ohp_objectives(
    graph: &Graph,
    partition: &OhpPartition,
    degrees: &HashMap<NodeId, usize, FxBuildHasher>,
) -> (f64, f64) {
    let total_edges = graph.edges.len() as f64;
    if total_edges == 0.0 {
        return (0.0, 0.0);
    }

    // Use OCCSA weights for consistency with membership assignment operators.
    let mut node_weights: FxHashMap<NodeId, FxHashMap<CommunityId, f64>> = FxHashMap::default();
    for &node in partition.keys() {
        let w_map = compute_occsa_membership_weights(node, partition, graph);
        node_weights.insert(node, w_map);
    }

    let mut community_degrees: FxHashMap<CommunityId, f64> = FxHashMap::default();
    for (&node, w_map) in &node_weights {
        let deg = *degrees.get(&node).unwrap_or(&0) as f64;
        for (&comm, &r_vc) in w_map {
            *community_degrees.entry(comm).or_insert(0.0) += deg * r_vc;
        }
    }

    let total_edges_doubled = 2.0 * total_edges;

    // f2: Inter-community modularity penalty (squared degree sum — prevents large-volume communities).
    let mut inter_sum = 0.0;
    for &comm_deg in community_degrees.values() {
        inter_sum += (comm_deg / total_edges_doubled).powi(2);
    }

    // f1: Intra-community edge coverage weighted by soft DWI memberships.
    let mut intra_sum = 0.0;
    for (u, v) in &graph.edges {
        if let (Some(w_u_map), Some(w_v_map)) = (node_weights.get(u), node_weights.get(v)) {
            for (&c_u, &r_uc) in w_u_map {
                if let Some(&r_vc) = w_v_map.get(&c_u) {
                    intra_sum += r_uc * r_vc;
                }
            }
        }
    }

    let intra = 1.0 - (intra_sum / total_edges);
    let inter = inter_sum;
    (intra, inter)
}

/// Calculates the 3rd objective f3: Parameter-Free Intrinsic Overlap Cohesion Penalty.
/// Governs the quality of overlapping memberships: penalizes overlapping nodes whose community
/// support fractions are highly imbalanced (s_min / s_max << 1), pruning weakly-supported overlaps.
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
