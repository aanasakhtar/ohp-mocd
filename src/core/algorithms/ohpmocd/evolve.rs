//! NSGA-II evolution loops for OHP-MOCD (crisp and overlapping modes).
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos.

use crate::core::algorithms::ohpmocd::defaults::*;
use crate::core::algorithms::ohpmocd::individual::{ohp_to_crisp, OhpIndividual};
use crate::core::algorithms::ohpmocd::operators::{
    create_offspring_ohp_seeded, generate_population_ohp_seeded,
};
use crate::core::graph::{Graph, Partition};
use crate::core::metaheuristics::helpers::individual::{fast_non_dominated_sort, Individual, TOURNAMENT_SIZE};
use crate::core::metaheuristics::nsga2::{self, select_survivors};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rustc_hash::FxHashMap;
use std::cmp::Ordering;

fn fast_non_dominated_sort_ohp(population: &mut [OhpIndividual]) {
    let mut inds: Vec<Individual> = population
        .iter()
        .map(|ind| Individual {
            partition: FxHashMap::default(),
            objectives: ind.objectives.clone(),
            rank: ind.rank,
            crowding_distance: ind.crowding_distance,
        })
        .collect();
    fast_non_dominated_sort(&mut inds);
    for (ohp, ind) in population.iter_mut().zip(inds.iter()) {
        ohp.rank = ind.rank;
    }
}

fn calculate_crowding_distance_ohp(population: &mut [OhpIndividual]) {
    let mut inds: Vec<Individual> = population
        .iter()
        .map(|ind| Individual {
            partition: FxHashMap::default(),
            objectives: ind.objectives.clone(),
            rank: ind.rank,
            crowding_distance: ind.crowding_distance,
        })
        .collect();
    crate::core::metaheuristics::nsga2::calculate_crowding_distance(&mut inds);
    for (ohp, ind) in population.iter_mut().zip(inds.iter()) {
        ohp.crowding_distance = ind.crowding_distance;
    }
}

pub fn select_survivors_ohp(population: &mut Vec<OhpIndividual>, pop_size: usize) {
    fast_non_dominated_sort_ohp(population);
    calculate_crowding_distance_ohp(population);
    population.sort_unstable_by(|a, b| {
        a.rank.cmp(&b.rank).then_with(|| {
            b.crowding_distance
                .partial_cmp(&a.crowding_distance)
                .unwrap_or(Ordering::Equal)
        })
    });
    population.truncate(pop_size);
}

/// Crisp NSGA-II evolution. When `seed` is `None`, uses the same shared path as HP-MOCD.
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub fn evolve_crisp<E>(
    graph: &Graph,
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    seed: Option<u64>,
    evaluate: impl FnMut(&mut [Individual]) -> Result<(), E>,
    on_generation: impl FnMut(usize, usize, &[Individual]) -> Result<(), E>,
) -> Result<Vec<Individual>, E> {
    match seed {
        None => nsga2::evolve(
            graph,
            pop_size,
            num_gens,
            cross_rate,
            mut_rate,
            TOURNAMENT_SIZE,
            evaluate,
            on_generation,
        ),
        Some(seed) => evolve_crisp_seeded(
            graph,
            pop_size,
            num_gens,
            cross_rate,
            mut_rate,
            seed,
            evaluate,
            on_generation,
        ),
    }
}

#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn evolve_crisp_seeded<E>(
    graph: &Graph,
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    seed: u64,
    mut evaluate: impl FnMut(&mut [Individual]) -> Result<(), E>,
    mut on_generation: impl FnMut(usize, usize, &[Individual]) -> Result<(), E>,
) -> Result<Vec<Individual>, E> {
    let mut rng = StdRng::seed_from_u64(seed);

    let mut individuals: Vec<Individual> =
        generate_population_ohp_seeded(graph, pop_size, &InitializationStrategy::Crisp, &mut rng)
            .into_iter()
            .map(|ohp| Individual::new(ohp_to_crisp(&ohp)))
            .collect();
    evaluate(&mut individuals)?;

    for generation in 0..num_gens {
        select_survivors(&mut individuals, pop_size);

        let mut offspring = super::operators::create_offspring_seeded(
            &individuals,
            graph,
            cross_rate,
            mut_rate,
            TOURNAMENT_SIZE,
            &mut rng,
        );
        evaluate(&mut offspring)?;
        individuals.extend(offspring);

        on_generation(generation, num_gens, &individuals)?;
    }

    Ok(individuals)
}

/// OHP NSGA-II evolution (supports crisp and overlapping modes with pluggable initialization strategy).
#[allow(clippy::too_many_arguments)]
pub fn evolve_ohp<E>(
    graph: &Graph,
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    init_strategy: &InitializationStrategy,
    overlap_support_threshold: f64,
    overlap_removal_threshold: f64,
    switch_margin: f64,
    alpha: f64,
    seed: Option<u64>,
    mut evaluate: impl FnMut(usize, &mut [OhpIndividual]) -> Result<(), E>,
    mut on_generation: impl FnMut(usize, usize, &[OhpIndividual]) -> Result<(), E>,
) -> Result<Vec<OhpIndividual>, E> {
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::seed_from_u64(rand::random()),
    };

    let mut individuals: Vec<OhpIndividual> =
        generate_population_ohp_seeded(graph, pop_size, init_strategy, &mut rng)
            .into_iter()
            .map(OhpIndividual::new)
            .collect();

    evaluate(0, &mut individuals)?;

    for generation in 0..num_gens {
        select_survivors_ohp(&mut individuals, pop_size);

        let mut offspring = create_offspring_ohp_seeded(
            &individuals,
            graph,
            cross_rate,
            mut_rate,
            TOURNAMENT_SIZE,
            overlap_support_threshold,
            overlap_removal_threshold,
            switch_margin,
            alpha,
            &mut rng,
        );

        evaluate(generation, &mut offspring)?;
        individuals.extend(offspring);

        on_generation(generation, num_gens, &individuals)?;
    }

    select_survivors_ohp(&mut individuals, pop_size);
    Ok(individuals)
}

/// Run crisp OHP-MOCD end-to-end with a fixed seed.
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub fn run_crisp_seeded(
    graph: &Graph,
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    seed: u64,
) -> Partition {
    use crate::core::algorithms::ohpmocd::utils::max_q_selection;
    use crate::core::utils::normalize_community_ids;

    let degrees = graph.precompute_degrees();

    let population = evolve_crisp(
        graph,
        pop_size,
        num_gens,
        cross_rate,
        mut_rate,
        Some(seed),
        |inds| {
            crate::core::algorithms::ohpmocd::objectives::evaluate_crisp_population(
                inds, graph, degrees,
            );
            Ok::<(), ()>(())
        },
        |_, _, _| Ok::<(), ()>(()),
    )
    .expect("evolution failed");

    let first_front: Vec<Individual> = population
        .into_iter()
        .filter(|ind| ind.rank == 1)
        .collect();

    let best = max_q_selection(&first_front);
    normalize_community_ids(graph, best.partition.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::ohpmocd::operators::hpmocd_reference_seeded;
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
    fn crisp_seeded_matches_hpmocd_reference() {
        let g = two_community_graph();
        let params = (20_usize, 5_usize, 0.7_f64, 0.5_f64, 42_u64);

        let reference = hpmocd_reference_seeded(&g, params.0, params.1, params.2, params.3, params.4);
        let ohpmocd = run_crisp_seeded(&g, params.0, params.1, params.2, params.3, params.4);

        assert_eq!(
            ohpmocd, reference,
            "OHP-MOCD crisp mode must match HP-MOCD reference"
        );
    }
}
