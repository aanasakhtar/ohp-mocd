//! Objective evaluation for OHP-MOCD.
//! Phase 2: delegates to Shi's decomposed modularity for crisp partitions.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos.

use crate::core::graph::{Graph, Partition};
use crate::core::metaheuristics::helpers::individual::Individual;
use crate::core::metaheuristics::helpers::objectives::decomposed_modularity::calculate_objectives;
use rayon::prelude::*;
use rustc_hash::FxBuildHasher;
use std::collections::HashMap;

pub fn evaluate_crisp_population(
    individuals: &mut [Individual],
    graph: &Graph,
    degrees: &HashMap<i32, usize, FxBuildHasher>,
) {
    individuals.par_iter_mut().for_each(|ind| {
        let metrics = calculate_objectives(graph, &ind.partition, degrees, true);
        ind.objectives = vec![metrics.intra, metrics.inter];
    });
}

pub fn evaluate_crisp_partition(
    graph: &Graph,
    partition: &Partition,
    degrees: &HashMap<i32, usize, FxBuildHasher>,
) -> (f64, f64) {
    let metrics = calculate_objectives(graph, partition, degrees, false);
    (metrics.intra, metrics.inter)
}
