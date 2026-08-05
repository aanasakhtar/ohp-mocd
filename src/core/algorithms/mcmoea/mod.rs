pub mod cliques;
pub mod evolve;
pub mod individual;
pub mod objectives;
pub mod operators;

use pyo3::prelude::*;
use pyo3::types::PyAny;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;

use crate::core::graph::{CommunityId, Graph, NodeId};
use crate::core::algorithms::mcmoea::evolve::evolve_mcmoea;

#[pyclass]
pub struct Mcmoea {
    graph: Graph,
    pop_size: usize,
    num_gens: usize,
    cross_rate: f64,
    mut_rate: f64,
    seed: Option<u64>,
}

#[pymethods]
impl Mcmoea {
    #[new]
    #[pyo3(signature = (graph, pop_size=100, num_gens=100, cross_rate=0.8, mut_rate=0.2, seed=None))]
    pub fn new(
        _py: Python<'_>,
        graph: &Bound<'_, PyAny>,
        pop_size: usize,
        num_gens: usize,
        cross_rate: f64,
        mut_rate: f64,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let rust_graph = Graph::from_python(graph);
        Ok(Self {
            graph: rust_graph,
            pop_size,
            num_gens,
            cross_rate,
            mut_rate,
            seed,
        })
    }

    pub fn run(&mut self) -> PyResult<HashMap<NodeId, Vec<CommunityId>>> {
        let mut rng = match self.seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::seed_from_u64(rand::random()),
        };

        let raw_partition = evolve_mcmoea(
            &self.graph,
            self.pop_size,
            self.num_gens,
            self.cross_rate,
            self.mut_rate,
            &mut rng,
        );

        let result: HashMap<NodeId, Vec<CommunityId>> = raw_partition.into_iter().collect();
        Ok(result)
    }
}
