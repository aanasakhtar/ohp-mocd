//! Genetic operators for OHP-MOCD (Clean, threshold-free stochastic evolutionary architecture).
//! Implements random initialization, majority consensus crossover, and local-move mutation.
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

/// Pluggable population initialization (Crisp, RandomOverlap, or BoundarySeeded).
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

/// Phase 5: Overlap-aware ensemble crossover over 4 parents with majority vote and consensus overlap.
pub fn ensemble_crossover_ohp_with_rng<R: Rng + ?Sized>(
    parents: &[&OhpPartition],
    _graph: &Graph,
    rng: &mut R,
) -> OhpPartition {
    if parents.is_empty() {
        return FxHashMap::default();
    }

    let mut keys: Vec<NodeId> = parents[0].keys().copied().collect();
    keys.sort_unstable();
    let mut child = FxHashMap::with_capacity_and_hasher(keys.len(), FxBuildHasher);
    let mut community_counts = FxHashMap::with_capacity_and_hasher(8, FxBuildHasher);

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

        // Primary community: majority vote among parents (random tie break)
        let max_cnt = counts_vec[0].1;
        let tied_primaries: Vec<CommunityId> = counts_vec
            .iter()
            .filter(|(_, cnt)| *cnt == max_cnt)
            .map(|&(c, _)| c)
            .collect();
        let primary = *tied_primaries.choose(rng).unwrap();

        // Secondary communities: any community agreed upon by >= 2 parents (50% parent consensus)
        let mut secondaries = Vec::new();
        for &(c, count) in &counts_vec {
            if c != primary && count >= 2 {
                secondaries.push(c);
            }
        }

        child.insert(node, OhpMembership::new(primary, &secondaries));
    }

    child
}

/// Phase 6: Clean stochastic mutation (Add, Remove, or Switch Primary) without threshold policing.
pub fn mutate_ohp_with_rng<R: Rng + ?Sized>(
    partition: &mut OhpPartition,
    graph: &Graph,
    mutation_rate: f64,
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

        // Count community frequencies in node's neighborhood
        let mut comm_freq = FxHashMap::default();
        for &nbr in neighbors {
            if let Some(m) = old_partition.get(&nbr) {
                for &c in &m.communities {
                    *comm_freq.entry(c).or_insert(0usize) += 1;
                }
            }
        }

        if comm_freq.is_empty() {
            continue;
        }

        let mut sorted_comms: Vec<(CommunityId, usize)> = comm_freq.into_iter().collect();
        sorted_comms.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        // 1. Primary community: majority vote in local neighborhood
        let max_cnt = sorted_comms[0].1;
        let tied_primaries: Vec<CommunityId> = sorted_comms
            .iter()
            .filter(|(_, cnt)| *cnt == max_cnt)
            .map(|&(c, _)| c)
            .collect();
        let primary = *tied_primaries.choose(rng).unwrap();

        // 2. Secondary communities: any runner-up community with >= 2 connections in N(u)
        let mut secondaries = Vec::new();
        for &(c, count) in &sorted_comms {
            if c != primary && count >= 2 {
                secondaries.push(c);
            }
        }

        partition.insert(node, OhpMembership::new(primary, &secondaries));
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
                    &mut local_rng,
                )
            } else {
                parent_partitions[local_rng.random_range(0..parent_partitions.len())].clone()
            };

            mutate_ohp_with_rng(
                &mut child,
                graph,
                mutation_rate,
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

/// Parameter-Free Memetic Boundary Local Search Operator (LSO).
/// For every overlapping node u (|M(u)| > 1), prunes any community c where the node's internal degree
/// falls below the random expectation threshold: d_u^{in}(c) < d_u / |M(u)|.
pub fn memetic_boundary_refinement(
    graph: &Graph,
    partition: &mut OhpPartition,
) {
    let mut updates: Vec<(crate::core::graph::NodeId, Vec<crate::core::graph::CommunityId>)> = Vec::new();

    for (&u, membership) in partition.iter() {
        if membership.len() <= 1 {
            continue;
        }

        let neighbors = match graph.adjacency_list.get(&u) {
            Some(nbrs) if !nbrs.is_empty() => nbrs,
            _ => continue,
        };
        let deg_u = neighbors.len() as f64;
        let k = membership.len() as f64;
        let threshold = deg_u / k;

        let mut internal_deg: rustc_hash::FxHashMap<crate::core::graph::CommunityId, usize> = rustc_hash::FxHashMap::default();
        for &c in &membership.communities {
            internal_deg.insert(c, 0);
        }

        for &v in neighbors {
            if let Some(v_m) = partition.get(&v) {
                for &c in &v_m.communities {
                    if let Some(count) = internal_deg.get_mut(&c) {
                        *count += 1;
                    }
                }
            }
        }

        let mut kept_comms: Vec<crate::core::graph::CommunityId> = Vec::new();
        let mut best_comm = membership.communities[0];
        let mut max_in = 0;

        for &c in &membership.communities {
            let in_count = *internal_deg.get(&c).unwrap_or(&0);
            if in_count > max_in {
                max_in = in_count;
                best_comm = c;
            }
            if (in_count as f64) >= threshold {
                kept_comms.push(c);
            }
        }

        if kept_comms.is_empty() {
            kept_comms.push(best_comm);
        }

        if kept_comms.len() != membership.len() {
            updates.push((u, kept_comms));
        }
    }

    for (node, comms) in updates {
        if let Some(m) = partition.get_mut(&node) {
            *m = OhpMembership::from_vec(comms);
        }
    }
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
    fn test_ensemble_crossover_majority_and_secondary() {
        let p1: OhpPartition = [(0, OhpMembership::new(0, &[])), (1, OhpMembership::new(0, &[1]))].into_iter().collect();
        let p2: OhpPartition = [(0, OhpMembership::new(0, &[])), (1, OhpMembership::new(0, &[1]))].into_iter().collect();
        let p3: OhpPartition = [(0, OhpMembership::new(1, &[])), (1, OhpMembership::new(1, &[]))].into_iter().collect();
        let p4: OhpPartition = [(0, OhpMembership::new(0, &[])), (1, OhpMembership::new(1, &[0]))].into_iter().collect();

        let g = boundary_node_graph();
        let mut rng = StdRng::seed_from_u64(42);
        let child = ensemble_crossover_ohp_with_rng(&[&p1, &p2, &p3, &p4], &g, &mut rng);

        // Node 0: parents have [0, 0, 1, 0] -> majority primary is 0
        assert_eq!(child[&0].primary(), 0);
        // Node 1: parents have primaries [0, 0, 1, 1] (tie between 0 and 1)
        assert!(child[&1].primary() == 0 || child[&1].primary() == 1);
    }

    #[test]
    fn test_local_move_majority_mutation() {
        let g = boundary_node_graph();
        let mut part: OhpPartition = [
            (0, OhpMembership::new(0, &[])),
            (1, OhpMembership::new(0, &[])),
            (2, OhpMembership::new(1, &[])), // Node 2 in community 1, but connected to 0, 1 (comm 0) and 3 (comm 1)
            (3, OhpMembership::new(1, &[])),
            (4, OhpMembership::new(1, &[])),
            (5, OhpMembership::new(1, &[])),
        ]
        .into_iter()
        .collect();

        let mut rng = StdRng::seed_from_u64(42);
        // Mutate with mut_rate = 1.0
        mutate_ohp_with_rng(&mut part, &g, 1.0, &mut rng);

        // Node 2 has neighbors 0 (comm 0), 1 (comm 0), 3 (comm 1). Majority count is community 0 (2 votes).
        assert_eq!(part[&2].primary(), 0);
    }
}
