//! Overlapping High-Performance Multi-Objective Community Detection (OHP-MOCD).
//! Phase 2: crisp-compatible scaffold (`max_memberships_per_node = 1`) reuses the
//! HP-MOCD NSGA-II pipeline; overlapping behaviour is added in later phases.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos.

mod defaults;
mod evolve;
mod individual;
mod objectives;
mod operators;
mod utils;

pub use defaults::*;

use crate::core::graph::{Graph, Partition};
use crate::core::metaheuristics::helpers::individual::Individual;
use crate::core::utils::normalize_community_ids;
use evolve::evolve_crisp;
use objectives::evaluate_crisp_population;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use rustc_hash::FxBuildHasher;
use std::collections::HashMap;
use utils::max_q_selection;

/// NSGA-II overlapping community detection (experimental).
///
/// With ``max_memberships_per_node=1``, behaves identically to HP-MOCD.
///
/// Args:
///     graph: networkx.Graph or DiGraph.
///     debug_level: 0 silent, 1+ logs every 10 generations.
///     pop_size: NSGA-II population size.
///     num_gens: number of generations.
///     cross_rate: crossover probability.
///     mut_rate: mutation probability.
///     max_memberships_per_node: 1 for crisp (HP-MOCD-compatible), 2 for overlap (Phase 3+).
///     seed: optional RNG seed for reproducible runs.
#[gen_stub_pyclass]
#[pyclass]
pub struct OhpMocd {
    graph: Graph,
    debug_level: i8,
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    max_memberships_per_node: usize,
    seed: Option<u64>,
    py_graph: Option<Py<PyAny>>,
    py_objectives: Vec<Py<PyAny>>,
    on_generation: Option<Py<PyAny>>,
}

impl OhpMocd {
    fn ensure_crisp_mode(&self) -> PyResult<()> {
        if self.max_memberships_per_node != 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "overlapping mode (max_memberships_per_node > 1) is not yet implemented",
            ));
        }
        Ok(())
    }

    fn evaluate_population(
        &self,
        py: Option<Python<'_>>,
        individuals: &mut [Individual],
        graph: &Graph,
        degrees: &HashMap<i32, usize, FxBuildHasher>,
    ) -> PyResult<()> {
        if self.py_objectives.is_empty() {
            evaluate_crisp_population(individuals, graph, degrees);
            Ok(())
        } else {
            let py = py.expect("Python token required when py_objectives are set");
            let py_graph = self
                .py_graph
                .as_ref()
                .expect("py_graph must be set when py_objectives are used");
            let py_objs = &self.py_objectives;

            let partition_dict = PyDict::new(py);
            let graph_ref = py_graph.bind(py);
            for ind in individuals.iter_mut() {
                partition_dict.clear();
                for (&node, &comm) in ind.partition.iter() {
                    partition_dict.set_item(node, comm)?;
                }
                let mut objectives = Vec::with_capacity(py_objs.len());
                for obj in py_objs.iter() {
                    let value = obj
                        .bind(py)
                        .call1((graph_ref, &partition_dict))?
                        .extract::<f64>()?;
                    objectives.push(value);
                }
                ind.objectives = objectives;
            }
            Ok(())
        }
    }

    fn envolve(&self, py: Option<Python<'_>>) -> PyResult<Vec<Individual>> {
        self.ensure_crisp_mode()?;

        let degrees = self.graph.precompute_degrees();

        let individuals = evolve_crisp(
            &self.graph,
            self.pop_size,
            self.num_gens,
            self.cross_rate,
            self.mut_rate,
            self.seed,
            |inds| self.evaluate_population(py, inds, &self.graph, degrees),
            |generation, num_gens, pop| {
                let first_front_size = pop.iter().filter(|ind| ind.rank == 1).count();

                if self.debug_level >= 1 && (generation % 10 == 0 || generation == num_gens - 1) {
                    debug!(
                        debug,
                        "OHP-MOCD NSGA-II: Gen {} | 1st Front/Pop: {}/{}",
                        generation,
                        first_front_size,
                        pop.len()
                    );
                }

                if let Some(cb) = &self.on_generation
                    && let Some(py) = py
                {
                    cb.bind(py)
                        .call1((generation, num_gens, first_front_size))?;
                }
                Ok(())
            },
        )?;

        Ok(individuals
            .into_iter()
            .filter(|ind| ind.rank == 1)
            .collect())
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl OhpMocd {
    #[new]
    #[pyo3(signature = (graph,
        debug_level = DEFAULT_DEBUG_LEVEL,
        pop_size = DEFAULT_POP_SIZE,
        num_gens = DEFAULT_NUM_GENS,
        cross_rate = DEFAULT_CROSS_RATE,
        mut_rate = DEFAULT_MUT_RATE,
        max_memberships_per_node = DEFAULT_MAX_MEMBERSHIPS_PER_NODE,
        seed = None,
        objectives = None
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        _py: Python<'_>,
        graph: &Bound<'_, PyAny>,
        debug_level: i8,
        pop_size: usize,
        num_gens: usize,
        cross_rate: f64,
        mut_rate: f64,
        max_memberships_per_node: usize,
        seed: Option<u64>,
        objectives: Option<&Bound<'_, PyList>>,
    ) -> PyResult<Self> {
        let rust_graph = Graph::from_python(graph);

        if debug_level >= 1 {
            debug!(
                debug,
                "Debug: {} | Level: {}",
                debug_level >= 1,
                debug_level
            );
            rust_graph.print();
        }

        let py_graph = Some(graph.clone().unbind());
        let py_objectives: Vec<Py<PyAny>> = objectives
            .map(|obj_list| obj_list.iter().map(|item| item.unbind()).collect())
            .unwrap_or_default();

        Ok(OhpMocd {
            graph: rust_graph,
            debug_level,
            pop_size,
            num_gens,
            cross_rate,
            mut_rate,
            max_memberships_per_node,
            seed,
            py_graph,
            py_objectives,
            on_generation: None,
        })
    }

    #[pyo3(signature = (objectives))]
    pub fn set_objectives(&mut self, objectives: &Bound<'_, PyList>) -> PyResult<()> {
        self.py_objectives = objectives.iter().map(|item| item.unbind()).collect();
        Ok(())
    }

    #[pyo3(signature = (callback))]
    pub fn set_on_generation(&mut self, callback: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        self.on_generation = callback.map(|cb| cb.clone().unbind());
        Ok(())
    }

    #[getter]
    pub fn num_gens(&self) -> usize {
        self.num_gens
    }

    #[getter]
    pub fn max_memberships_per_node(&self) -> usize {
        self.max_memberships_per_node
    }

    #[pyo3(signature = ())]
    pub fn generate_pareto_front(&self, py: Python<'_>) -> PyResult<Vec<(Partition, Vec<f64>)>> {
        let first_front = self.envolve(Some(py))?;

        Ok(first_front
            .into_iter()
            .map(|ind| {
                (
                    normalize_community_ids(&self.graph, ind.partition),
                    ind.objectives,
                )
            })
            .collect())
    }

    /// Run and return the best crisp partition (max-Q from the Pareto front).
    /// With ``max_memberships_per_node=1``, equivalent to HP-MOCD.
    /// Isolated nodes get community ``-1``.
    #[pyo3(signature = ())]
    pub fn run(&self, py: Python<'_>) -> PyResult<Partition> {
        let first_front: Vec<Individual> = self.envolve(Some(py))?;
        let best_solution: &Individual = max_q_selection(&first_front);

        Ok(normalize_community_ids(
            &self.graph,
            best_solution.partition.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::ohpmocd::evolve::run_crisp_seeded;
    use crate::core::algorithms::ohpmocd::operators::hpmocd_reference_seeded;
    use crate::core::graph::{CommunityId, NodeId, Partition};

    fn two_community_graph() -> Graph {
        let mut g = Graph::new();
        for (a, b) in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)] {
            g.add_edge(a, b);
        }
        g.finalize();
        g
    }

    fn part(pairs: &[(NodeId, CommunityId)]) -> Partition {
        pairs.iter().copied().collect()
    }

    #[test]
    fn crisp_mode_matches_hpmocd_reference_under_seed() {
        let g = two_community_graph();
        let reference = hpmocd_reference_seeded(&g, 20, 5, 0.7, 0.5, 42);
        let ohpmocd = run_crisp_seeded(&g, 20, 5, 0.7, 0.5, 42);
        assert_eq!(ohpmocd, reference);
    }

    #[test]
    fn membership_validity_crisp_partition_covers_all_nodes() {
        let g = two_community_graph();
        let result = run_crisp_seeded(&g, 20, 5, 0.7, 0.5, 7);
        for &node in g.nodes.iter() {
            assert!(result.contains_key(&node), "missing node {node}");
            let comm = result[&node];
            assert!(comm >= -1, "invalid community id for node {node}");
        }
    }

    #[test]
    fn normalize_produces_contiguous_ids() {
        let g = two_community_graph();
        let raw = part(&[(0, 10), (1, 10), (2, 10), (3, 20), (4, 20), (5, 20)]);
        let norm = normalize_community_ids(&g, raw);
        let ids: Vec<CommunityId> = norm.values().copied().collect();
        assert!(ids.iter().all(|&c| c == -1 || (0..=1).contains(&c)));
    }
}
