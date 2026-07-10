//! Genetic operators for OHP-MOCD.
//! Phase 2 (crisp mode): mirrors the shared HP-MOCD operators with optional seeded RNG
//! for deterministic regression tests.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos.

use crate::core::graph::{CommunityId, Graph, NodeId, Partition};
use crate::core::metaheuristics::helpers::individual::{Individual, TOURNAMENT_SIZE};
use crate::core::metaheuristics::helpers::operators;
use rand::distr::{Bernoulli, Distribution};
use rand::rngs::StdRng;
use rand::seq::IndexedRandom;
use rand::{Rng, RngExt, SeedableRng};
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet as HashSet};

const ENSEMBLE_SIZE: usize = 4;

fn ensemble_crossover_with_rng<R: Rng + ?Sized>(
    parents: &[&Partition],
    rng: &mut R,
) -> Partition {
    if parents.is_empty() {
        return FxHashMap::default();
    }

    let keys: Vec<NodeId> = parents[0].keys().copied().collect();
    let mut child = FxHashMap::with_capacity_and_hasher(keys.len(), FxBuildHasher);
    let mut community_counts = FxHashMap::with_capacity_and_hasher(parents.len(), FxBuildHasher);
    let mut candidates = Vec::with_capacity(parents.len());

    for &node in &keys {
        community_counts.clear();

        let majority_threshold = parents.len().div_ceil(2);
        let mut max_count = 0;
        let mut best_community = parents[0][&node];

        for parent in parents {
            if let Some(&community) = parent.get(&node) {
                let count = community_counts.entry(community).or_insert(0);
                *count += 1;

                if *count > max_count {
                    max_count = *count;
                    best_community = community;
                    if *count >= majority_threshold {
                        break;
                    }
                }
            }
        }

        let tie_count = community_counts
            .values()
            .filter(|&&count| count == max_count)
            .count();

        if tie_count > 1 {
            candidates.clear();
            candidates.extend(
                community_counts
                    .iter()
                    .filter(|(_, count)| **count == max_count)
                    .map(|(&comm, _)| comm),
            );
            best_community = *candidates.choose(rng).unwrap();
        }

        child.insert(node, best_community);
    }

    child
}

pub fn generate_population_seeded(
    graph: &Graph,
    population_size: usize,
    rng: &mut StdRng,
) -> Vec<Partition> {
    let node_ids: Vec<NodeId> = graph.nodes.iter().copied().collect();
    let num_communities = node_ids.len().max(1);
    (0..population_size)
        .map(|_| {
            node_ids
                .iter()
                .map(|&node_id| {
                    let community = rng.random_range(0..num_communities) as CommunityId;
                    (node_id, community)
                })
                .collect()
        })
        .collect()
}

fn tournament_selection_index(
    population: &[Individual],
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

pub fn mutate_with_rng(
    partition: &mut Partition,
    graph: &Graph,
    mutation_rate: f64,
    rng: &mut StdRng,
) {
    if mutation_rate == 0.0 || partition.is_empty() {
        return;
    }

    let mutation_dist = Bernoulli::new(mutation_rate).unwrap();
    let mut community_freq = FxHashMap::with_capacity_and_hasher(16, FxBuildHasher);

    for node in partition.keys().copied().collect::<Vec<_>>() {
        if !mutation_dist.sample(rng) {
            continue;
        }

        community_freq.clear();

        if let Some(neighbors) = graph.adjacency_list.get(&node) {
            let mut max_count = 0;
            let mut best_community = partition[&node];

            for &neighbor in neighbors {
                if let Some(&community) = partition.get(&neighbor) {
                    let count = community_freq.entry(community).or_insert(0);
                    *count += 1;

                    if *count > max_count {
                        max_count = *count;
                        best_community = community;
                    }
                }
            }

            if max_count > 0 && best_community != partition[&node] {
                partition.insert(node, best_community);
            }
        }
    }
}

pub fn create_offspring_seeded(
    population: &[Individual],
    graph: &Graph,
    crossover_rate: f64,
    mutation_rate: f64,
    tournament_size: usize,
    rng: &mut StdRng,
) -> Vec<Individual> {
    let pop_size = population.len();
    let crossover_dist = Bernoulli::new(crossover_rate).unwrap();
    let mut offspring = Vec::with_capacity(pop_size);

    for _ in 0..pop_size {
        let mut unique_parents = HashSet::with_capacity_and_hasher(ENSEMBLE_SIZE, FxBuildHasher);
        while unique_parents.len() < ENSEMBLE_SIZE {
            let parent_idx = tournament_selection_index(population, tournament_size, rng);
            unique_parents.insert(parent_idx);
        }

        let parent_partitions: Vec<&Partition> = unique_parents
            .iter()
            .map(|&idx| &population[idx].partition)
            .collect();

        let mut child = if crossover_dist.sample(rng) {
            ensemble_crossover_with_rng(&parent_partitions, rng)
        } else {
            parent_partitions[rng.random_range(0..parent_partitions.len())].clone()
        };

        mutate_with_rng(&mut child, graph, mutation_rate, rng);
        offspring.push(Individual::new(child));
    }

    offspring
}

/// Reference crisp pipeline used to verify HP-MOCD equivalence under a fixed seed.
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
        let metrics = operators::get_fitness(graph, &ind.partition, degrees, true);
        ind.objectives = vec![metrics.intra, metrics.inter];
    }

    for _ in 0..num_gens {
        select_survivors(&mut individuals, pop_size);
        let mut offspring =
            create_offspring_seeded(&individuals, graph, cross_rate, mut_rate, TOURNAMENT_SIZE, &mut rng);
        for ind in offspring.iter_mut() {
            let metrics = operators::get_fitness(graph, &ind.partition, degrees, true);
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

    fn two_community_graph() -> Graph {
        let mut g = Graph::new();
        for (a, b) in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)] {
            g.add_edge(a, b);
        }
        g.finalize();
        g
    }

    #[test]
    fn seeded_reference_is_deterministic() {
        let g = two_community_graph();
        let a = hpmocd_reference_seeded(&g, 20, 5, 0.7, 0.5, 42);
        let b = hpmocd_reference_seeded(&g, 20, 5, 0.7, 0.5, 42);
        assert_eq!(a, b);
    }
}
