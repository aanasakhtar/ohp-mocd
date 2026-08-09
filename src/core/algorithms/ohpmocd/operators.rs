//! Genetic operators for OHP-MOCD (Phases 4, 5, 6).
//! Implements random crisp initialization, Top-K overlap-aware ensemble crossover,
//! and Top-K topology-guided mutation.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos.

use crate::core::algorithms::ohpmocd::defaults::*;
use crate::core::algorithms::ohpmocd::individual::{
    crisp_to_ohp, ohp_to_crisp, OhpIndividual, OhpMembership, OhpPartition,
};
use crate::core::graph::{CommunityId, Graph, NodeId, Partition};
use crate::core::metaheuristics::helpers::individual::{Individual, TOURNAMENT_SIZE};
use rand::distr::{Bernoulli, Distribution};
use rand::rngs::StdRng;
use rand::seq::IndexedRandom;
use rand::{Rng, RngExt, SeedableRng};
use rayon::prelude::*;
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet as HashSet};

const ENSEMBLE_SIZE: usize = 4;

/// Calculates neighborhood support for community `c` at node `v`.
/// support(v, c) = |{u in N(v) : c in M(u)}| / |N(v)|
pub(crate) fn neighborhood_support_counts(
    node: NodeId,
    partition: &OhpPartition,
    graph: &Graph,
) -> FxHashMap<CommunityId, usize> {
    let mut counts = FxHashMap::default();

    if let Some(neighbors) = graph.adjacency_list.get(&node) {
        for &neighbor in neighbors {
            if let Some(m) = partition.get(&neighbor) {
                for &community in &m.communities {
                    *counts.entry(community).or_insert(0) += 1;
                }
            }
        }
    }

    counts
}

#[inline(always)]
pub(crate) fn support_ratio_from_counts(
    support_counts: &FxHashMap<CommunityId, usize>,
    degree: usize,
    community: CommunityId,
) -> f64 {
    if degree == 0 {
        return 0.0;
    }

    support_counts
        .get(&community)
        .copied()
        .unwrap_or(0) as f64
        / degree as f64
}

/// Calculates neighborhood support for community `c` at node `v`.
/// support(v, c) = |{u in N(v) : c in M(u)}| / |N(v)|
#[allow(dead_code)]
pub fn calculate_support(
    node: NodeId,
    community: CommunityId,
    partition: &OhpPartition,
    graph: &Graph,
) -> f64 {
    let neighbors = match graph.adjacency_list.get(&node) {
        Some(n) if !n.is_empty() => n,
        _ => return 0.0,
    };

    let support_counts = neighborhood_support_counts(node, partition, graph);
    support_ratio_from_counts(&support_counts, neighbors.len(), community)
}

/// Phase 4: Pluggable population initialization (Crisp, RandomOverlap, or BoundarySeeded).
pub fn generate_population_ohp_seeded(
    graph: &Graph,
    population_size: usize,
    strategy: &InitializationStrategy,
    rng: &mut StdRng,
) -> Vec<OhpPartition> {
    let mut node_ids: Vec<NodeId> = graph.nodes.iter().copied().collect();
    node_ids.sort_unstable();
    let num_communities = node_ids.len().max(1);

    (0..population_size)
        .map(|_| {
            let mut partition: OhpPartition = node_ids
                .iter()
                .map(|&node_id| {
                    let community = rng.random_range(0..num_communities) as CommunityId;
                    (node_id, OhpMembership::new(community, &[]))
                })
                .collect();

            match strategy {
                InitializationStrategy::Crisp => {}
                InitializationStrategy::RandomOverlap { overlap_probability } => {
                    if *overlap_probability > 0.0 {
                        let mut nodes: Vec<NodeId> = partition.keys().copied().collect();
                        nodes.sort_unstable();
                        for &node in &nodes {
                            if rng.random_bool(*overlap_probability) {
                                let primary = partition[&node].primary();
                                let mut candidate_comms = Vec::new();
                                if let Some(neighbors) = graph.adjacency_list.get(&node) {
                                    for &nbr in neighbors {
                                        if let Some(m) = partition.get(&nbr) {
                                            let nbr_p = m.primary();
                                            if nbr_p != primary && !candidate_comms.contains(&nbr_p) {
                                                candidate_comms.push(nbr_p);
                                            }
                                        }
                                    }
                                }
                                if candidate_comms.is_empty() {
                                    let rand_c = rng.random_range(0..num_communities) as CommunityId;
                                    if rand_c != primary {
                                        candidate_comms.push(rand_c);
                                    }
                                }
                                if let Some(&sec) = candidate_comms.choose(rng) {
                                    if let Some(m) = partition.get_mut(&node) {
                                        if !m.contains(sec) {
                                            m.communities.push(sec);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                InitializationStrategy::BoundarySeeded { overlap_probability } => {
                    if *overlap_probability > 0.0 {
                        let mut nodes: Vec<NodeId> = partition.keys().copied().collect();
                        nodes.sort_unstable();
                        for &node in &nodes {
                            let neighbors = match graph.adjacency_list.get(&node) {
                                Some(n) if !n.is_empty() => n,
                                _ => continue,
                            };

                            let primary = partition[&node].primary();
                            let mut nbr_comm_counts = FxHashMap::default();
                            for &nbr in neighbors {
                                if let Some(m) = partition.get(&nbr) {
                                    *nbr_comm_counts.entry(m.primary()).or_insert(0usize) += 1;
                                }
                            }

                            if nbr_comm_counts.len() > 1 && rng.random_bool(*overlap_probability) {
                                let mut sorted_nbr_comms: Vec<(CommunityId, usize)> =
                                    nbr_comm_counts.into_iter().collect();
                                sorted_nbr_comms.sort_unstable_by(|a, b| {
                                    b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0))
                                });

                                for (cand_c, _) in sorted_nbr_comms {
                                    if cand_c != primary {
                                        if let Some(m) = partition.get_mut(&node) {
                                            if !m.contains(cand_c) {
                                                m.communities.push(cand_c);
                                                break;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            partition
        })
        .collect()
}

/// Phase 5: Overlap-aware ensemble crossover over 4 parents with dynamic neighborhood support.
pub fn ensemble_crossover_ohp_with_rng<R: Rng + ?Sized>(
    parents: &[&OhpPartition],
    graph: &Graph,
    overlap_support_threshold: f64,
    rng: &mut R,
) -> OhpPartition {
    if parents.is_empty() {
        return FxHashMap::default();
    }

    let mut keys: Vec<NodeId> = parents[0].keys().copied().collect();
    keys.sort_unstable();
    let mut child = FxHashMap::with_capacity_and_hasher(keys.len(), FxBuildHasher);
    let mut community_counts = FxHashMap::with_capacity_and_hasher(8, FxBuildHasher);
    let mut runner_ups: FxHashMap<NodeId, Vec<CommunityId>> =
        FxHashMap::with_capacity_and_hasher(keys.len(), FxBuildHasher);

    // Step 1: Majority vote for primary label and collect runner-up candidates
    for &node in &keys {
        community_counts.clear();

        for parent in parents {
            if let Some(m) = parent.get(&node) {
                for &c in &m.communities {
                    *community_counts.entry(c).or_insert(0) += 1;
                }
            }
        }

        if community_counts.is_empty() {
            child.insert(node, OhpMembership::new(0, &[]));
            continue;
        }

        let mut counts_vec: Vec<(CommunityId, usize)> =
            community_counts.iter().map(|(&c, &cnt)| (c, cnt)).collect();
        counts_vec.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        let max_cnt = counts_vec[0].1;
        let tied_primaries: Vec<CommunityId> = counts_vec
            .iter()
            .filter(|(_, cnt)| *cnt == max_cnt)
            .map(|&(c, _)| c)
            .collect();
        let primary = *tied_primaries.choose(rng).unwrap();

        let runner_up_candidates: Vec<CommunityId> = counts_vec
            .iter()
            .filter(|&&(c, _)| c != primary)
            .map(|&(c, _)| c)
            .collect();

        child.insert(node, OhpMembership::new(primary, &[]));
        runner_ups.insert(node, runner_up_candidates);
    }

    // Step 2: Evaluate neighborhood support for runner-ups dynamically
    for &node in &keys {
        if let Some(cands) = runner_ups.get(&node) {
            let support_counts = neighborhood_support_counts(node, &child, graph);
            let degree = graph.adjacency_list.get(&node).map_or(0, Vec::len);
            for &cand in cands {
                let supp = support_ratio_from_counts(&support_counts, degree, cand);
                if supp >= overlap_support_threshold {
                    if let Some(m) = child.get_mut(&node) {
                        if !m.communities.contains(&cand) {
                            m.communities.push(cand);
                        }
                    }
                }
            }
        }
    }

    child
}

/// Phase 6: Topology-guided mutation for dynamic memberships (Add / Remove / Switch rules).
pub fn mutate_ohp_with_rng<R: Rng + ?Sized>(
    partition: &mut OhpPartition,
    graph: &Graph,
    mutation_rate: f64,
    overlap_support_threshold: f64,
    overlap_removal_threshold: f64,
    switch_margin: f64,
    rng: &mut R,
) {
    if mutation_rate == 0.0 || partition.is_empty() {
        return;
    }

    let mutation_dist = Bernoulli::new(mutation_rate).unwrap();
    let old_partition = partition.clone();
    let mut nodes: Vec<NodeId> = partition.keys().copied().collect();
    nodes.sort_unstable();

    for node in nodes {
        if !mutation_dist.sample(rng) {
            continue;
        }

        let neighbors = match graph.adjacency_list.get(&node) {
            Some(n) if !n.is_empty() => n,
            _ => continue,
        };

        let support_counts = neighborhood_support_counts(node, &old_partition, graph);
        let mut neighbor_comms: Vec<CommunityId> = support_counts.keys().copied().collect();
        neighbor_comms.sort_unstable();

        if neighbor_comms.is_empty() {
            continue;
        }

        let current_m = old_partition[&node].clone();
        let primary_comm = current_m.primary();
        let primary_supp = support_ratio_from_counts(&support_counts, neighbors.len(), primary_comm);

        // Rule 3: Switch primary if neighbor community has higher support by switch_margin
        let mut best_switch_comm = primary_comm;
        let mut max_switch_supp = primary_supp;

        for &c in &neighbor_comms {
            if c != primary_comm {
                let supp = support_ratio_from_counts(&support_counts, neighbors.len(), c);
                if supp - primary_supp >= switch_margin && supp > max_switch_supp {
                    max_switch_supp = supp;
                    best_switch_comm = c;
                }
            }
        }

        let mut updated_communities = current_m.communities.clone();
        if best_switch_comm != primary_comm {
            updated_communities.retain(|&c| c != best_switch_comm);
            updated_communities.insert(0, best_switch_comm);
        }

        // Rule 2: Remove unsupported additional memberships (indices 1..)
        let mut kept = vec![updated_communities[0]];
        for &c in &updated_communities[1..] {
            let supp = support_ratio_from_counts(&support_counts, neighbors.len(), c);
            if supp >= overlap_removal_threshold {
                if !kept.contains(&c) {
                    kept.push(c);
                }
            }
        }
        updated_communities = kept;

        // Rule 1: Add supported additional memberships dynamically based on support threshold
        let mut candidates: Vec<(CommunityId, f64)> = neighbor_comms
            .iter()
            .filter(|&&c| !updated_communities.contains(&c))
            .map(|&c| (c, support_ratio_from_counts(&support_counts, neighbors.len(), c)))
            .filter(|&(_, supp)| supp >= overlap_support_threshold)
            .collect();

        candidates.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });

        for (c, _) in candidates {
            if !updated_communities.contains(&c) {
                updated_communities.push(c);
            }
        }

        partition.insert(node, OhpMembership::from_vec(updated_communities));
    }
}

fn tournament_selection_index_ohp(
    population: &[OhpIndividual],
    tournament_size: usize,
    rng: &mut StdRng,
) -> usize {
    let mut best_idx = rng.random_range(0..population.len());
    let mut best = &population[best_idx];

    for _ in 1..tournament_size {
        let candidate_idx = rng.random_range(0..population.len());
        let candidate = &population[candidate_idx];

        if candidate.rank < best.rank
            || (candidate.rank == best.rank && candidate.crowding_distance > best.crowding_distance)
        {
            best = candidate;
            best_idx = candidate_idx;
        }
    }

    best_idx
}

pub fn create_offspring_ohp_seeded(
    population: &[OhpIndividual],
    graph: &Graph,
    crossover_rate: f64,
    mutation_rate: f64,
    tournament_size: usize,
    overlap_support_threshold: f64,
    overlap_removal_threshold: f64,
    switch_margin: f64,
    rng: &mut StdRng,
) -> Vec<OhpIndividual> {
    let pop_size = population.len();
    let mut plans: Vec<(Vec<usize>, u64)> = Vec::with_capacity(pop_size);

    for _ in 0..pop_size {
        let mut unique_parents = HashSet::with_capacity_and_hasher(ENSEMBLE_SIZE, FxBuildHasher);
        while unique_parents.len() < ENSEMBLE_SIZE {
            let parent_idx = tournament_selection_index_ohp(population, tournament_size, rng);
            unique_parents.insert(parent_idx);
        }
        let mut parent_indices: Vec<usize> = unique_parents.into_iter().collect();
        parent_indices.sort_unstable();
        plans.push((parent_indices, rng.random()));
    }

    plans
        .into_par_iter()
        .map(|(parent_indices, child_seed)| {
            let mut local_rng = StdRng::seed_from_u64(child_seed);
            let parent_partitions: Vec<&OhpPartition> = parent_indices
                .iter()
                .map(|&idx| &population[idx].partition)
                .collect();

            let mut child = if local_rng.random_bool(crossover_rate) {
                ensemble_crossover_ohp_with_rng(
                    &parent_partitions,
                    graph,
                    overlap_support_threshold,
                    &mut local_rng,
                )
            } else {
                parent_partitions[local_rng.random_range(0..parent_partitions.len())].clone()
            };

            mutate_ohp_with_rng(
                &mut child,
                graph,
                mutation_rate,
                overlap_support_threshold,
                overlap_removal_threshold,
                switch_margin,
                &mut local_rng,
            );

            OhpIndividual::new(child)
        })
        .collect()
}

// Crisp wrappers for backward compatibility & HP-MOCD reference testing
#[allow(dead_code)]
pub fn generate_population_seeded(
    graph: &Graph,
    population_size: usize,
    rng: &mut StdRng,
) -> Vec<Partition> {
    generate_population_ohp_seeded(
        graph,
        population_size,
        &InitializationStrategy::Crisp,
        rng,
    )
    .into_iter()
    .map(|ohp| ohp_to_crisp(&ohp))
    .collect()
}

#[allow(dead_code)]
pub fn mutate_with_rng(
    partition: &mut Partition,
    graph: &Graph,
    mutation_rate: f64,
    rng: &mut StdRng,
) {
    let mut ohp = crisp_to_ohp(partition);
    mutate_ohp_with_rng(
        &mut ohp,
        graph,
        mutation_rate,
        DEFAULT_OVERLAP_SUPPORT_THRESHOLD,
        DEFAULT_OVERLAP_REMOVAL_THRESHOLD,
        DEFAULT_SWITCH_MARGIN,
        rng,
    );
    *partition = ohp_to_crisp(&ohp);
}

#[allow(dead_code)]
pub fn create_offspring_seeded(
    population: &[Individual],
    graph: &Graph,
    crossover_rate: f64,
    mutation_rate: f64,
    tournament_size: usize,
    rng: &mut StdRng,
) -> Vec<Individual> {
    let ohp_pop: Vec<OhpIndividual> = population.iter().cloned().map(Into::into).collect();
    let ohp_offspring = create_offspring_ohp_seeded(
        &ohp_pop,
        graph,
        crossover_rate,
        mutation_rate,
        tournament_size,
        DEFAULT_OVERLAP_SUPPORT_THRESHOLD,
        DEFAULT_OVERLAP_REMOVAL_THRESHOLD,
        DEFAULT_SWITCH_MARGIN,
        rng,
    );
    ohp_offspring.into_iter().map(Into::into).collect()
}

#[allow(dead_code)]
pub fn hpmocd_reference_seeded(
    graph: &Graph,
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    seed: u64,
) -> Partition {
    use crate::core::algorithms::ohpmocd::utils::max_q_selection;
    use crate::core::metaheuristics::nsga2::select_survivors;
    use crate::core::utils::normalize_community_ids;

    let degrees = graph.precompute_degrees();
    let mut rng = StdRng::seed_from_u64(seed);

    let mut individuals: Vec<Individual> = generate_population_seeded(graph, pop_size, &mut rng)
        .into_iter()
        .map(Individual::new)
        .collect();

    for ind in individuals.iter_mut() {
        let metrics =
            crate::core::metaheuristics::helpers::operators::get_fitness(graph, &ind.partition, degrees, true);
        ind.objectives = vec![metrics.intra, metrics.inter];
    }

    for _ in 0..num_gens {
        select_survivors(&mut individuals, pop_size);
        let mut offspring =
            create_offspring_seeded(&individuals, graph, cross_rate, mut_rate, TOURNAMENT_SIZE, &mut rng);
        for ind in offspring.iter_mut() {
            let metrics =
                crate::core::metaheuristics::helpers::operators::get_fitness(graph, &ind.partition, degrees, true);
            ind.objectives = vec![metrics.intra, metrics.inter];
        }
        individuals.extend(offspring);
    }

    let first_front: Vec<Individual> = individuals
        .into_iter()
        .filter(|ind| ind.rank == 1)
        .collect();

    let best = max_q_selection(&first_front);
    normalize_community_ids(graph, best.partition.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::graph::Graph;

    fn boundary_node_graph() -> Graph {
        let mut g = Graph::new();
        for (a, b) in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)] {
            g.add_edge(a, b);
        }
        g.finalize();
        g
    }

    #[test]
    fn support_calculation_correctness() {
        let g = boundary_node_graph();
        let part: OhpPartition = [
            (0, OhpMembership::new(0, &[])),
            (1, OhpMembership::new(0, &[])),
            (2, OhpMembership::new(0, &[1])),
            (3, OhpMembership::new(1, &[])),
            (4, OhpMembership::new(1, &[])),
            (5, OhpMembership::new(1, &[])),
        ]
        .into_iter()
        .collect();

        assert!((calculate_support(2, 0, &part, &g) - 2.0 / 3.0).abs() < 1e-6);
        assert!((calculate_support(2, 1, &part, &g) - 1.0 / 3.0).abs() < 1e-6);
    }
}
