//! Objective functions for OHP-MOCD (clean threshold-free architecture).
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos.

use std::collections::HashMap;

use rayon::prelude::*;
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};

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

/// Precomputes static asymmetric node intimacy F_uv = (|N(u) ∩ N(v)| + 1) / d_u (FCCNI Eq. 4)
pub fn precompute_intimacy(graph: &Graph) -> FxHashMap<NodeId, FxHashMap<NodeId, f64>> {
    let mut intimacy = FxHashMap::default();
    for (&u, u_neighbors) in &graph.adjacency_list {
        let d_u = u_neighbors.len() as f64;
        if d_u == 0.0 {
            continue;
        }
        let u_set: FxHashSet<NodeId> = u_neighbors.iter().copied().collect();
        let mut u_map = FxHashMap::default();
        for &v in u_neighbors {
            if let Some(v_neighbors) = graph.adjacency_list.get(&v) {
                let common = v_neighbors.iter().filter(|n| u_set.contains(n)).count() as f64;
                let f_uv = (common + 1.0) / d_u;
                u_map.insert(v, f_uv);
            }
        }
        intimacy.insert(u, u_map);
    }
    intimacy
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

/// Computes Direction 2 intimacy-informed soft membership weights dynamically
pub(crate) fn compute_intimacy_membership_weights(
    node: NodeId,
    partition: &OhpPartition,
    intimacy: &FxHashMap<NodeId, FxHashMap<NodeId, f64>>,
    graph: &Graph,
) -> FxHashMap<CommunityId, f64> {
    let mut weights = FxHashMap::default();
    let membership = match partition.get(&node) {
        Some(m) if !m.is_empty() => m,
        _ => return weights,
    };

    if membership.len() == 1 {
        weights.insert(membership.communities[0], 1.0);
        return weights;
    }

    let mut comm_intimacy_sum: FxHashMap<CommunityId, f64> = FxHashMap::default();
    for &c in &membership.communities {
        comm_intimacy_sum.insert(c, 0.0);
    }

    if let Some(u_intimacy_map) = intimacy.get(&node) {
        if let Some(neighbors) = graph.adjacency_list.get(&node) {
            for &v in neighbors {
                let f_uv = *u_intimacy_map.get(&v).unwrap_or(&0.0);
                if let Some(v_m) = partition.get(&v) {
                    for &c in &v_m.communities {
                        if let Some(sum_val) = comm_intimacy_sum.get_mut(&c) {
                            *sum_val += f_uv;
                        }
                    }
                }
            }
        }
    }

    let total_intimacy: f64 = comm_intimacy_sum.values().sum();
    if total_intimacy > 0.0 {
        for (c, score) in comm_intimacy_sum {
            weights.insert(c, score / total_intimacy);
        }
    } else {
        let unif = 1.0 / membership.len() as f64;
        for &c in &membership.communities {
            weights.insert(c, unif);
        }
    }

    weights
}

/// Calculates Shi-style decomposed modularity (intra, inter) using uniform or intimacy soft weights.
pub fn calculate_ohp_objectives(
    graph: &Graph,
    partition: &OhpPartition,
    degrees: &HashMap<NodeId, usize, FxBuildHasher>,
    intimacy: Option<&FxHashMap<NodeId, FxHashMap<NodeId, f64>>>,
) -> (f64, f64) {
    let total_edges = graph.edges.len() as f64;
    if total_edges == 0.0 {
        return (0.0, 0.0);
    }

    let mut node_weights: FxHashMap<NodeId, FxHashMap<CommunityId, f64>> = FxHashMap::default();
    for &node in partition.keys() {
        let w_map = match intimacy {
            Some(intimacy_map) => compute_intimacy_membership_weights(node, partition, intimacy_map, graph),
            None => compute_uniform_membership_weights(node, partition),
        };
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

    // f1: Intra-community edge coverage weighted by soft memberships
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

/// Calculates the 3rd objective f3 (Standard): Parameter-Free Overlap Complexity Count Penalty.
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

/// Calculates Direction 1 Structural Overlap Cohesion objective f3:
/// Penalizes unsupported/spurious overlapping memberships where a node has low internal degree support.
pub fn calculate_structural_cohesion_f3(
    graph: &Graph,
    partition: &OhpPartition,
) -> f64 {
    let total_nodes = graph.nodes.len() as f64;
    if total_nodes == 0.0 {
        return 0.0;
    }

    let mut total_penalty = 0.0;

    for (&u, membership) in partition.iter() {
        if membership.len() <= 1 {
            continue;
        }
        let deg_u = match graph.adjacency_list.get(&u) {
            Some(nbrs) => nbrs.len() as f64,
            None => 0.0,
        };
        if deg_u == 0.0 {
            continue;
        }

        let mut internal_deg: FxHashMap<CommunityId, usize> = FxHashMap::default();
        for &c in &membership.communities {
            internal_deg.insert(c, 0);
        }

        if let Some(neighbors) = graph.adjacency_list.get(&u) {
            for &v in neighbors {
                if let Some(v_m) = partition.get(&v) {
                    for &c in &v_m.communities {
                        if let Some(count) = internal_deg.get_mut(&c) {
                            *count += 1;
                        }
                    }
                }
            }
        }

        for (_c, &in_deg) in &internal_deg {
            let support_ratio = (2.0 * in_deg as f64) / deg_u;
            if support_ratio < 1.0 {
                total_penalty += 1.0 - support_ratio;
            }
        }
    }

    total_penalty / total_nodes
}

/// Calculates Ratio Cut & Ratio Association objectives (Pizzuti / MCMOEA / Shi formulation).
/// f1 = Intra-Community Ratio Association loss (to minimize)
/// f2 = Inter-Community Ratio Cut (to minimize)
pub fn calculate_ratio_cut_objectives(
    graph: &Graph,
    partition: &OhpPartition,
) -> (f64, f64) {
    let total_edges = graph.edges.len() as f64;
    if total_edges == 0.0 {
        return (0.0, 0.0);
    }

    let mut comm_sizes: FxHashMap<CommunityId, usize> = FxHashMap::default();
    for membership in partition.values() {
        for &c in &membership.communities {
            *comm_sizes.entry(c).or_insert(0) += 1;
        }
    }

    let mut comm_in_edges: FxHashMap<CommunityId, f64> = FxHashMap::default();
    let mut comm_out_edges: FxHashMap<CommunityId, f64> = FxHashMap::default();

    for (u, v) in &graph.edges {
        if let (Some(m_u), Some(m_v)) = (partition.get(u), partition.get(v)) {
            let u_set: FxHashSet<CommunityId> = m_u.communities.iter().copied().collect();
            let v_set: FxHashSet<CommunityId> = m_v.communities.iter().copied().collect();

            let common: Vec<CommunityId> = u_set.intersection(&v_set).copied().collect();
            if !common.is_empty() {
                let share = 1.0 / common.len() as f64;
                for c in common {
                    *comm_in_edges.entry(c).or_insert(0.0) += share;
                }
            } else {
                for &c_u in &u_set {
                    *comm_out_edges.entry(c_u).or_insert(0.0) += 0.5;
                }
                for &c_v in &v_set {
                    *comm_out_edges.entry(c_v).or_insert(0.0) += 0.5;
                }
            }
        }
    }

    let mut ra_sum = 0.0;
    let mut rc_sum = 0.0;

    for (&c, &size) in &comm_sizes {
        if size > 0 {
            let size_f = size as f64;
            let in_e = *comm_in_edges.get(&c).unwrap_or(&0.0);
            let out_e = *comm_out_edges.get(&c).unwrap_or(&0.0);
            ra_sum += (2.0 * in_e) / size_f;
            rc_sum += out_e / size_f;
        }
    }

    let f1 = 1.0 / (1.0 + ra_sum);
    let f2 = rc_sum / total_edges;
    (f1, f2)
}

pub fn evaluate_ohp_population(
    individuals: &mut [OhpIndividual],
    graph: &Graph,
    degrees: &HashMap<NodeId, usize, FxBuildHasher>,
    enable_f3: bool,
    objective_mode: &str,
    intimacy: Option<&FxHashMap<NodeId, FxHashMap<NodeId, f64>>>,
) {
    let is_cohesion = objective_mode.eq_ignore_ascii_case("cohesion_intimacy");
    let is_ratio_cut = objective_mode.eq_ignore_ascii_case("ratio_cut");

    individuals.par_iter_mut().for_each(|ind| {
        let (intra, inter) = if is_ratio_cut {
            calculate_ratio_cut_objectives(graph, &ind.partition)
        } else {
            calculate_ohp_objectives(graph, &ind.partition, degrees, if is_cohesion { intimacy } else { None })
        };
        if enable_f3 {
            let f3 = if is_cohesion {
                calculate_structural_cohesion_f3(graph, &ind.partition)
            } else {
                calculate_f3_objective(graph, &ind.partition)
            };
            ind.objectives = vec![intra, inter, f3];
        } else {
            ind.objectives = vec![intra, inter];
        }
    });
}
