//! Overlapping High-Performance Multi-Objective Community Detection (OHP-MOCD).
//! Supports crisp (`max_memberships_per_node = 1`), Top-K overlapping (`max_memberships_per_node >= 2`),
//! and pluggable population initialization strategies (Crisp, RandomOverlap, BoundarySeeded).
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos.

mod defaults;
mod evolve;
mod individual;
mod objectives;
mod operators;
mod utils;

pub use defaults::*;
pub use individual::{OhpIndividual, OhpMembership, OhpPartition};

use crate::core::graph::{CommunityId, Graph};
use evolve::evolve_ohp;
use objectives::evaluate_ohp_population;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use rustc_hash::{FxBuildHasher, FxHashMap};
use std::collections::HashMap;
use std::sync::Mutex;
use utils::max_q_selection_ohp;

pub fn normalize_ohp_community_ids(graph: &Graph, partition: OhpPartition) -> OhpPartition {
    let mut new_partition: OhpPartition = FxHashMap::default();
    let mut id_mapping: FxHashMap<CommunityId, CommunityId> = FxHashMap::default();
    let mut next_id: CommunityId = 0;

    for &node in graph.nodes.iter() {
        let is_isolated = match graph.adjacency_list.get(&node) {
            Some(neighbors) => neighbors.is_empty(),
            None => true,
        };

        if is_isolated {
            new_partition.insert(node, OhpMembership::new(-1, &[]));
        } else if let Some(m) = partition.get(&node) {
            let mut norm_communities = Vec::with_capacity(m.communities.len());
            for &orig in &m.communities {
                if orig == -1 {
                    if !norm_communities.contains(&-1) {
                        norm_communities.push(-1);
                    }
                } else {
                    let mapped = *id_mapping.entry(orig).or_insert_with(|| {
                        let id = next_id;
                        next_id += 1;
                        id
                    });
                    if !norm_communities.contains(&mapped) {
                        norm_communities.push(mapped);
                    }
                }
            }
            new_partition.insert(node, OhpMembership::from_vec(norm_communities));
        } else {
            new_partition.insert(node, OhpMembership::new(-1, &[]));
        }
    }

    new_partition
}

pub fn ohp_partition_to_py(
    py: Python<'_>,
    max_memberships_per_node: usize,
    partition: &OhpPartition,
) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    if max_memberships_per_node == 1 {
        for (&node, m) in partition.iter() {
            dict.set_item(node, m.primary())?;
        }
    } else {
        for (&node, m) in partition.iter() {
            let list = PyList::new(py, m.to_vec())?;
            dict.set_item(node, list)?;
        }
    }
    Ok(dict.into_any().unbind())
}

/// NSGA-II Top-K overlapping community detection (OHP-MOCD).
///
/// With ``max_memberships_per_node=1``, behaves identically to HP-MOCD.
/// With ``max_memberships_per_node >= 2``, detects 1 to K overlapping community memberships per node.
/// Supports pluggable initialization strategies: ``"crisp"``, ``"random_overlap"``, and ``"boundary_seeded"``.
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
    init_strategy: InitializationStrategy,
    seed: Option<u64>,
    py_graph: Option<Py<PyAny>>,
    py_objectives: Vec<Py<PyAny>>,
    on_generation: Option<Py<PyAny>>,
    convergence_history: Mutex<Vec<f64>>,
}

impl OhpMocd {
    fn evaluate_population(
        &self,
        py: Option<Python<'_>>,
        individuals: &mut [OhpIndividual],
        graph: &Graph,
        degrees: &HashMap<i32, usize, FxBuildHasher>,
    ) -> PyResult<()> {
        if self.py_objectives.is_empty() {
            evaluate_ohp_population(individuals, graph, degrees);
            Ok(())
        } else {
            let py = py.expect("Python token required when py_objectives are set");
            let py_graph = self
                .py_graph
                .as_ref()
                .expect("py_graph must be set when py_objectives are used");
            let py_objs = &self.py_objectives;

            for ind in individuals.iter_mut() {
                let partition_py =
                    ohp_partition_to_py(py, self.max_memberships_per_node, &ind.partition)?;
                let graph_ref = py_graph.bind(py);
                let mut objectives = Vec::with_capacity(py_objs.len());
                for obj in py_objs.iter() {
                    let value = obj
                        .bind(py)
                        .call1((graph_ref, &partition_py))?
                        .extract::<f64>()?;
                    objectives.push(value);
                }
                ind.objectives = objectives;
            }
            Ok(())
        }
    }

    fn envolve(&self, py: Option<Python<'_>>) -> PyResult<Vec<OhpIndividual>> {
        let degrees = self.graph.precompute_degrees();
        if let Ok(mut hist) = self.convergence_history.lock() {
            hist.clear();
        }

        let individuals = evolve_ohp(
            &self.graph,
            self.pop_size,
            self.num_gens,
            self.cross_rate,
            self.mut_rate,
            self.max_memberships_per_node,
            &self.init_strategy,
            self.seed,
            |inds| self.evaluate_population(py, inds, &self.graph, degrees),
            |generation, num_gens, pop| {
                let rank1_solutions: Vec<&OhpIndividual> =
                    pop.iter().filter(|ind| ind.rank == 1).collect();
                let first_front_size = rank1_solutions.len();

                let best_q = rank1_solutions
                    .iter()
                    .map(|ind| 1.0 - ind.objectives[0] - ind.objectives[1])
                    .fold(f64::NEG_INFINITY, f64::max);

                if let Ok(mut hist) = self.convergence_history.lock() {
                    hist.push(best_q);
                }

                if self.debug_level >= 1 && (generation % 10 == 0 || generation == num_gens - 1) {
                    debug!(
                        debug,
                        "OHP-MOCD NSGA-II: Gen {} | 1st Front/Pop: {}/{} | Best Q: {:.4}",
                        generation,
                        first_front_size,
                        pop.len(),
                        best_q
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
        init_strategy = "crisp",
        init_overlap_prob = 0.2,
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
        init_strategy: &str,
        init_overlap_prob: f64,
        seed: Option<u64>,
        objectives: Option<&Bound<'_, PyList>>,
    ) -> PyResult<Self> {
        let rust_graph = Graph::from_python(graph);

        let strat = match init_strategy.to_lowercase().as_str() {
            "random_overlap" | "random" => InitializationStrategy::RandomOverlap {
                overlap_probability: init_overlap_prob,
            },
            "boundary_seeded" | "boundary" => InitializationStrategy::BoundarySeeded {
                overlap_probability: init_overlap_prob,
            },
            _ => InitializationStrategy::Crisp,
        };

        if debug_level >= 1 {
            debug!(
                debug,
                "Debug: {} | Level: {} | Strategy: {:?}",
                debug_level >= 1,
                debug_level,
                strat
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
            init_strategy: strat,
            seed,
            py_graph,
            py_objectives,
            on_generation: None,
            convergence_history: Mutex::new(Vec::new()),
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
    pub fn get_convergence_history(&self) -> Vec<f64> {
        self.convergence_history.lock().unwrap().clone()
    }

    #[pyo3(signature = ())]
    pub fn generate_pareto_front(&self, py: Python<'_>) -> PyResult<Vec<(Py<PyAny>, Vec<f64>)>> {
        let first_front = self.envolve(Some(py))?;

        let mut result = Vec::with_capacity(first_front.len());
        for ind in first_front {
            let norm = normalize_ohp_community_ids(&self.graph, ind.partition);
            let py_part = ohp_partition_to_py(py, self.max_memberships_per_node, &norm)?;
            result.push((py_part, ind.objectives));
        }
        Ok(result)
    }

    /// Run and return the best partition (max-Q from the Pareto front).
    /// If ``max_memberships_per_node=1``, returns ``dict[node, community]``.
    /// If ``max_memberships_per_node >= 2``, returns ``dict[node, list[community]]``.
    /// Isolated nodes get community ``-1``.
    #[pyo3(signature = ())]
    pub fn run(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let first_front: Vec<OhpIndividual> = self.envolve(Some(py))?;
        let best_solution: &OhpIndividual = max_q_selection_ohp(&first_front);
        let norm = normalize_ohp_community_ids(&self.graph, best_solution.partition.clone());

        ohp_partition_to_py(py, self.max_memberships_per_node, &norm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::algorithms::ohpmocd::evolve::{evolve_ohp, run_crisp_seeded};
    use crate::core::algorithms::ohpmocd::operators::{
        ensemble_crossover_ohp_with_rng, hpmocd_reference_seeded, mutate_ohp_with_rng,
    };
    use rand::SeedableRng;

    fn two_community_graph() -> Graph {
        let mut g = Graph::new();
        for (a, b) in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)] {
            g.add_edge(a, b);
        }
        g.finalize();
        g
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
    fn membership_validity_top_k_overlapping_mode() {
        let g = two_community_graph();
        let degrees = g.precompute_degrees();

        let pop = evolve_ohp(
            &g,
            30,
            10,
            0.7,
            0.5,
            3,
            &InitializationStrategy::Crisp,
            Some(42),
            |inds| {
                objectives::evaluate_ohp_population(inds, &g, &degrees);
                Ok::<(), ()>(())
            },
            |_, _, _| Ok::<(), ()>(()),
        )
        .expect("evolution failed");

        assert!(!pop.is_empty());
        for ind in &pop {
            for &node in g.nodes.iter() {
                assert!(ind.partition.contains_key(&node), "missing node {node}");
                let m = &ind.partition[&node];
                assert!(m.len() >= 1 && m.len() <= 3, "invalid membership length");
            }
        }
    }

    #[test]
    fn initialization_strategies_run_successfully() {
        let g = two_community_graph();
        let degrees = g.precompute_degrees();

        for strat in [
            InitializationStrategy::Crisp,
            InitializationStrategy::RandomOverlap {
                overlap_probability: 0.5,
            },
            InitializationStrategy::BoundarySeeded {
                overlap_probability: 0.5,
            },
        ] {
            let pop = evolve_ohp(
                &g,
                20,
                5,
                0.7,
                0.5,
                2,
                &strat,
                Some(42),
                |inds| {
                    objectives::evaluate_ohp_population(inds, &g, &degrees);
                    Ok::<(), ()>(())
                },
                |_, _, _| Ok::<(), ()>(()),
            )
            .unwrap();

            assert_eq!(pop.len(), 20);
        }
    }
}
