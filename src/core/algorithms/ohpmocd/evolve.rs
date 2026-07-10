//! NSGA-II evolution loops for OHP-MOCD.
//! Phase 2: crisp mode delegates to the shared HP-MOCD engine when no seed is set,
//! and uses a sequential seeded path for deterministic regression tests.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos.

use crate::core::graph::{Graph, Partition};
use crate::core::metaheuristics::helpers::individual::{Individual, TOURNAMENT_SIZE};
use crate::core::metaheuristics::nsga2::{self, select_survivors};

use super::operators::{create_offspring_seeded, generate_population_seeded};
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Crisp NSGA-II evolution. When `seed` is `None`, uses the same shared path as HP-MOCD.
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

    let mut individuals: Vec<Individual> = generate_population_seeded(graph, pop_size, &mut rng)
        .into_iter()
        .map(Individual::new)
        .collect();
    evaluate(&mut individuals)?;

    for generation in 0..num_gens {
        select_survivors(&mut individuals, pop_size);

        let mut offspring = create_offspring_seeded(
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

/// Run crisp OHP-MOCD end-to-end with a fixed seed (for tests and reproducibility).
#[allow(clippy::too_many_arguments)]
pub fn run_crisp_seeded(
    graph: &Graph,
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    seed: u64,
) -> Partition {
    use crate::core::algorithms::ohpmocd::objectives::evaluate_crisp_population;
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
            evaluate_crisp_population(inds, graph, degrees);
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
    use crate::core::algorithms::ohpmocd::objectives::evaluate_crisp_population;
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

    #[test]
    fn crisp_seeded_evolve_is_deterministic() {
        let g = two_community_graph();
        let degrees = g.precompute_degrees();
        let seed = 99_u64;

        let run = || {
            evolve_crisp(
                &g,
                20,
                5,
                0.7,
                0.5,
                Some(seed),
                |inds| {
                    evaluate_crisp_population(inds, &g, degrees);
                    Ok::<(), ()>(())
                },
                |_, _, _| Ok::<(), ()>(()),
            )
            .unwrap()
        };

        let a = run();
        let b = run();
        assert_eq!(a.len(), b.len());
        for (ia, ib) in a.iter().zip(b.iter()) {
            assert_eq!(ia.partition, ib.partition);
            assert_eq!(ia.objectives, ib.objectives);
        }
    }
}
