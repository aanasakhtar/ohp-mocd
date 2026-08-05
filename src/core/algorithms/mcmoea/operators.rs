use rand::RngExt;
use crate::core::algorithms::mcmoea::individual::McmoeaIndividual;
use crate::core::graph::CommunityId;

pub fn generate_population_mcmoea<R: RngExt>(
    num_cliques: usize,
    pop_size: usize,
    rng: &mut R,
) -> Vec<McmoeaIndividual> {
    let mut population = Vec::with_capacity(pop_size);
    for _ in 0..pop_size {
        let mut labels = Vec::with_capacity(num_cliques);
        for _ in 0..num_cliques {
            labels.push(rng.random_range(0..num_cliques as CommunityId));
        }
        population.push(McmoeaIndividual::new(labels));
    }
    population
}

pub fn crossover_mcmoea<R: RngExt>(
    parent1: &McmoeaIndividual,
    parent2: &McmoeaIndividual,
    cross_rate: f64,
    rng: &mut R,
) -> (McmoeaIndividual, McmoeaIndividual) {
    let n = parent1.clique_labels.len();
    if n == 0 || rng.random::<f64>() > cross_rate {
        return (parent1.clone(), parent2.clone());
    }

    let mut child1_labels = vec![0; n];
    let mut child2_labels = vec![0; n];

    let point = if n > 1 { rng.random_range(1..n) } else { 0 };
    for i in 0..n {
        if i < point {
            child1_labels[i] = parent1.clique_labels[i];
            child2_labels[i] = parent2.clique_labels[i];
        } else {
            child1_labels[i] = parent2.clique_labels[i];
            child2_labels[i] = parent1.clique_labels[i];
        }
    }

    (McmoeaIndividual::new(child1_labels), McmoeaIndividual::new(child2_labels))
}

pub fn mutate_mcmoea<R: RngExt>(
    individual: &mut McmoeaIndividual,
    mut_rate: f64,
    rng: &mut R,
) {
    let num_cliques = individual.clique_labels.len();
    if num_cliques == 0 {
        return;
    }

    for i in 0..num_cliques {
        if rng.random::<f64>() < mut_rate {
            individual.clique_labels[i] = rng.random_range(0..num_cliques as CommunityId);
        }
    }
}
