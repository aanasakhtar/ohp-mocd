//! OHP-MOCD individual representation.
//! Phase 2: crisp mode stores a single label per node in `partition` (same encoding
//! as HP-MOCD). Overlapping memberships will extend this type in Phase 3.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos.

use crate::core::graph::Partition;
use crate::core::metaheuristics::helpers::individual::Individual;

#[derive(Clone, Debug)]
pub struct OhpIndividual {
    pub partition: Partition,
    pub objectives: Vec<f64>,
    pub rank: usize,
    pub crowding_distance: f64,
}

impl OhpIndividual {
    pub fn new(partition: Partition) -> Self {
        OhpIndividual {
            partition,
            objectives: vec![0.0, 0.0],
            rank: usize::MAX,
            crowding_distance: f64::MAX,
        }
    }

    pub fn to_individual(&self) -> Individual {
        Individual {
            partition: self.partition.clone(),
            objectives: self.objectives.clone(),
            rank: self.rank,
            crowding_distance: self.crowding_distance,
        }
    }
}

impl From<Individual> for OhpIndividual {
    fn from(ind: Individual) -> Self {
        OhpIndividual {
            partition: ind.partition,
            objectives: ind.objectives,
            rank: ind.rank,
            crowding_distance: ind.crowding_distance,
        }
    }
}

impl From<OhpIndividual> for Individual {
    fn from(ind: OhpIndividual) -> Self {
        ind.to_individual()
    }
}
