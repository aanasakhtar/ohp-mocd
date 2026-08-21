//! Max-modularity selector for OHP-MOCD (crisp and overlapping modes).
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos.

use crate::core::algorithms::ohpmocd::individual::OhpIndividual;
use crate::core::metaheuristics::helpers::individual::Individual;
use std::cmp::Ordering;

#[allow(dead_code)]
#[inline(always)]
pub fn q(ind: &Individual) -> f64 {
    let n: f64 = ind.objectives.len() as f64;
    n - ind.objectives.iter().sum::<f64>()
}

#[allow(dead_code)]
#[inline(always)]
pub fn max_q_selection<'a>(population: &'a [Individual]) -> &'a Individual {
    population
        .iter()
        .max_by(|a, b| q(a).partial_cmp(&q(b)).unwrap_or(Ordering::Equal))
        .expect("Empty population")
}

#[inline(always)]
pub fn q_ohp(ind: &OhpIndividual) -> f64 {
    let f1 = ind.objectives.get(0).copied().unwrap_or(0.0);
    let f2 = ind.objectives.get(1).copied().unwrap_or(0.0);
    1.0 - f1 - f2
}

#[inline(always)]
pub fn max_q_selection_ohp<'a>(population: &'a [OhpIndividual]) -> &'a OhpIndividual {
    population
        .iter()
        .max_by(|a, b| q_ohp(a).partial_cmp(&q_ohp(b)).unwrap_or(Ordering::Equal))
        .expect("Empty population")
}

/// Utopia-Point Knee Selection on the Rank-1 Pareto Front.
/// Normalizes all objectives across the front and selects the solution minimizing Euclidean distance to the Ideal Utopia point (0, 0, 0).
pub fn knee_selection_ohp<'a>(population: &'a [OhpIndividual]) -> &'a OhpIndividual {
    let rank1: Vec<&OhpIndividual> = population.iter().filter(|ind| ind.rank == 1).collect();
    if rank1.is_empty() {
        return max_q_selection_ohp(population);
    }
    if rank1.len() == 1 {
        return rank1[0];
    }

    let num_objs = rank1[0].objectives.len();
    let mut min_objs = vec![f64::INFINITY; num_objs];
    let mut max_objs = vec![f64::NEG_INFINITY; num_objs];

    for ind in &rank1 {
        for (k, &val) in ind.objectives.iter().enumerate() {
            if val < min_objs[k] {
                min_objs[k] = val;
            }
            if val > max_objs[k] {
                max_objs[k] = val;
            }
        }
    }

    let mut best_ind = rank1[0];
    let mut min_dist = f64::INFINITY;

    for ind in &rank1 {
        let mut dist_sq = 0.0;
        for (k, &val) in ind.objectives.iter().enumerate() {
            let range = max_objs[k] - min_objs[k];
            let norm_val = if range > 1e-9 {
                (val - min_objs[k]) / range
            } else {
                0.0
            };
            dist_sq += norm_val.powi(2);
        }
        let dist = dist_sq.sqrt();
        if dist < min_dist {
            min_dist = dist;
            best_ind = ind;
        }
    }

    best_ind
}

pub fn select_ohp_solution<'a>(population: &'a [OhpIndividual], mode: &str) -> &'a OhpIndividual {
    match mode.to_lowercase().as_str() {
        "knee" | "knee_point" | "utopia" => knee_selection_ohp(population),
        _ => max_q_selection_ohp(population),
    }
}
