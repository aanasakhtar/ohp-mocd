use rand::RngExt;
use rustc_hash::FxHashMap;
use std::time::Instant;
use crate::core::graph::{CommunityId, Graph, NodeId};
use crate::core::algorithms::mcmoea::cliques::find_maximal_cliques;
use crate::core::algorithms::mcmoea::individual::McmoeaIndividual;
use crate::core::algorithms::mcmoea::objectives::McmoeaEvaluator;
use crate::core::algorithms::mcmoea::operators::{crossover_mcmoea, generate_population_mcmoea, mutate_mcmoea};

pub fn evolve_mcmoea<R: RngExt>(
    graph: &Graph,
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    rng: &mut R,
) -> FxHashMap<NodeId, Vec<CommunityId>> {
    let start_evolve = Instant::now();

    let cliques = find_maximal_cliques(graph);
    let num_cliques = cliques.len();

    let eval_start = Instant::now();
    println!("[MCMOEA Rust] Precomputing clique edge evaluation matrix...");
    let evaluator = McmoeaEvaluator::new(graph, &cliques);
    println!("[MCMOEA Rust] Evaluator matrix initialized in {:.3?} s", eval_start.elapsed());

    println!("[MCMOEA Rust] Starting Evolutionary Loop (pop={}, gen={})...", pop_size, num_gens);
    let gen_start_time = Instant::now();

    let mut population = generate_population_mcmoea(num_cliques, pop_size, rng);
    for ind in population.iter_mut() {
        evaluator.calculate_objectives(ind, &cliques);
    }

    for g in 0..num_gens {
        fast_non_dominated_sort_mcmoea(&mut population);
        calculate_crowding_distance_mcmoea(&mut population);

        let mut offspring = Vec::with_capacity(pop_size);
        while offspring.len() < pop_size {
            let p1 = tournament_select(&population, rng);
            let p2 = tournament_select(&population, rng);
            let (mut c1, mut c2) = crossover_mcmoea(p1, p2, cross_rate, rng);
            mutate_mcmoea(&mut c1, mut_rate, rng);
            mutate_mcmoea(&mut c2, mut_rate, rng);
            evaluator.calculate_objectives(&mut c1, &cliques);
            evaluator.calculate_objectives(&mut c2, &cliques);
            offspring.push(c1);
            if offspring.len() < pop_size {
                offspring.push(c2);
            }
        }

        population.extend(offspring);
        fast_non_dominated_sort_mcmoea(&mut population);
        calculate_crowding_distance_mcmoea(&mut population);
        population.sort_by(|a, b| a.rank.cmp(&b.rank).then_with(|| b.crowding_distance.partial_cmp(&a.crowding_distance).unwrap_or(std::cmp::Ordering::Equal)));
        population.truncate(pop_size);

        if (g + 1) % 10 == 0 || g == num_gens - 1 {
            let best_f1 = population[0].objectives[0];
            let best_f2 = population[0].objectives[1];
            println!(
                "  [MCMOEA Gen {:3}/{}] Elapsed: {:.3?} s | Front1 Size: {} | Best f1: {:.4}, f2: {:.4}",
                g + 1,
                num_gens,
                gen_start_time.elapsed(),
                population.iter().filter(|ind| ind.rank == 1).count(),
                best_f1,
                best_f2,
            );
        }
    }

    println!("[MCMOEA Rust] Evolutionary loop completed in {:.3?} s total", start_evolve.elapsed());

    let best_ind = population.iter()
        .filter(|ind| ind.rank == 1)
        .min_by(|a, b| (a.objectives[0] + a.objectives[1]).partial_cmp(&(b.objectives[0] + b.objectives[1])).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(&population[0]);

    best_ind.decode_node_memberships(&cliques)
}

fn tournament_select<'a, R: RngExt>(pop: &'a [McmoeaIndividual], rng: &mut R) -> &'a McmoeaIndividual {
    let i1 = rng.random_range(0..pop.len());
    let i2 = rng.random_range(0..pop.len());
    let ind1 = &pop[i1];
    let ind2 = &pop[i2];

    if ind1.rank < ind2.rank {
        ind1
    } else if ind2.rank < ind1.rank {
        ind2
    } else if ind1.crowding_distance > ind2.crowding_distance {
        ind1
    } else {
        ind2
    }
}

fn fast_non_dominated_sort_mcmoea(population: &mut [McmoeaIndividual]) {
    let n = population.len();
    let mut domination_counts = vec![0; n];
    let mut dominated_solutions = vec![Vec::new(); n];
    let mut front1 = Vec::new();

    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            if dominates(&population[i].objectives, &population[j].objectives) {
                dominated_solutions[i].push(j);
            } else if dominates(&population[j].objectives, &population[i].objectives) {
                domination_counts[i] += 1;
            }
        }
        if domination_counts[i] == 0 {
            population[i].rank = 1;
            front1.push(i);
        }
    }

    let mut current_front = front1;
    let mut current_rank = 1;

    while !current_front.is_empty() {
        let mut next_front = Vec::new();
        for &i in &current_front {
            for &j in &dominated_solutions[i] {
                domination_counts[j] -= 1;
                if domination_counts[j] == 0 {
                    population[j].rank = current_rank + 1;
                    next_front.push(j);
                }
            }
        }
        current_rank += 1;
        current_front = next_front;
    }
}

fn dominates(obj_a: &[f64], obj_b: &[f64]) -> bool {
    let mut better_in_any = false;
    for i in 0..obj_a.len() {
        if obj_a[i] > obj_b[i] {
            return false;
        }
        if obj_a[i] < obj_b[i] {
            better_in_any = true;
        }
    }
    better_in_any
}

fn calculate_crowding_distance_mcmoea(population: &mut [McmoeaIndividual]) {
    let n = population.len();
    if n == 0 {
        return;
    }
    for ind in population.iter_mut() {
        ind.crowding_distance = 0.0;
    }

    let num_objectives = 3;
    for m in 0..num_objectives {
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            population[a].objectives[m]
                .partial_cmp(&population[b].objectives[m])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        population[indices[0]].crowding_distance = f64::INFINITY;
        population[indices[n - 1]].crowding_distance = f64::INFINITY;

        let obj_min = population[indices[0]].objectives[m];
        let obj_max = population[indices[n - 1]].objectives[m];
        let distance_spread = obj_max - obj_min;

        if distance_spread > 0.0 {
            for i in 1..n - 1 {
                if population[indices[i]].crowding_distance.is_finite() {
                    let diff = population[indices[i + 1]].objectives[m]
                        - population[indices[i - 1]].objectives[m];
                    population[indices[i]].crowding_distance += diff / distance_spread;
                }
            }
        }
    }
}
