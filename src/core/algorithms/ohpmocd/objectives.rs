//! Objective functions for OHP-MOCD (clean threshold-free architecture).
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos.

use std::collections::HashMap;

use rayon::prelude::*;
use rustc_hash::{FxBuildHasher, FxHashMap};

use crate::core::algorithms::ohpmocd::individual::{OhpIndividual, OhpPartition};
use crate::core::graph::{CommunityId, Graph, NodeId, Partition};
use crate::core::metaheuristics::helpers::individual::Individual;
use crate::core::metaheuristics::helpers::operators::get_fitness;

/// Objective function type for OHP-MOCD.
pub type OhpObjectiveFn =
    Box<dyn Fn(&Graph, &OhpPartition) -> Vec<f64> + Send + Sync>;

/// Objective function calculation for crisp partitions (HPMOCD compatibility).
pub fn calculate_crisp_objectives(
    graph: &Graph,
    partition: &Partition,
    degrees: &HashMap<i32, usize, FxBuildHasher>,
) -> (f64, f64) {
    let metrics = get_fitness(graph, partition, degrees, true);
    (metrics.intra, metrics.inter)
}

/// Evaluates crisp population for baseline/comparison runs.
pub fn evaluate_crisp_population(
    individuals: &mut [Individual],
    graph: &Graph,
    degrees: &HashMap<i32, usize, FxBuildHasher>,
) {
    individuals.par_iter_mut().for_each(|ind| {
        let (intra, inter) = calculate_crisp_objectives(graph, &ind.partition, degrees);
        ind.objectives = vec![intra, inter];
    });
}

/// Computes uniform soft membership weights: r_{v,c} = 1 / |M(v)|
pub(crate) fn compute_uniform_membership_weights(
    node: NodeId,
    partition: &OhpPartition,
) -> FxHashMap<CommunityId, f64> {
    let mut weights = FxHashMap::default();
    let membership = match partition.get(&node) {
        Some(m) if !m.is_empty() => m,
        _ => return weights,
    };

    let unif = 1.0 / membership.len() as f64;
    for &c in &membership.communities {
        weights.insert(c, unif);
    }
    weights
}

/// Calculates Shi-style decomposed modularity (intra, inter) using uniform soft membership weights.
pub fn calculate_ohp_objectives(
    graph: &Graph,
    partition: &OhpPartition,
    degrees: &HashMap<NodeId, usize, FxBuildHasher>,
) -> (f64, f64) {
    let total_edges = graph.edges.len() as f64;
    if total_edges == 0.0 {
        return (0.0, 0.0);
    }

    let mut node_weights: FxHashMap<NodeId, FxHashMap<CommunityId, f64>> = FxHashMap::default();
    for &node in partition.keys() {
        let w_map = compute_uniform_membership_weights(node, partition);
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

    // f2: Inter-community modularity penalty (squared degree sum)
    let mut inter_sum = 0.0;
    for &comm_deg in community_degrees.values() {
        inter_sum += (comm_deg / total_edges_doubled).powi(2);
    }

    // f1: Intra-community edge coverage weighted by uniform memberships
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

/// Calculates the 3rd objective f3: Parameter-Free Overlap Complexity Penalty.
/// f3 = (1/N) * sum_v max(0, |M(v)| - 1)
pub fn calculate_f3_objective(
    graph: &Graph,
    partition: &OhpPartition,
) -> f64 {
    let total_nodes = graph.nodes.len() as f64;
    if total_nodes == 0.0 {
        return 0.0;
    }

    let mut excess_memberships = 0.0;
    for membership in partition.values() {
        if membership.len() > 1 {
            excess_memberships += (membership.len() - 1) as f64;
        }
    }

    excess_memberships / total_nodes
}

pub fn evaluate_ohp_population(
    individuals: &mut [OhpIndividual],
    graph: &Graph,
    degrees: &HashMap<NodeId, usize, FxBuildHasher>,
    enable_f3: bool,
) {
    individuals.par_iter_mut().for_each(|ind| {
        let (intra, inter) = calculate_ohp_objectives(graph, &ind.partition, degrees);
        if enable_f3 {
            let f3 = calculate_f3_objective(graph, &ind.partition);
            ind.objectives = vec![intra, inter, f3];
        } else {
            ind.objectives = vec![intra, inter];
        }
    });
}
