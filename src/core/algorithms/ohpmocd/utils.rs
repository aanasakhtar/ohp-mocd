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
