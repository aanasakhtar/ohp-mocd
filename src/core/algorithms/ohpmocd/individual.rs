//! OHP-MOCD individual representation.
//! Supports 1 to K community memberships per node in `OhpPartition`.
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos.

use crate::core::graph::{CommunityId, NodeId, Partition};
use crate::core::metaheuristics::helpers::individual::Individual;
use rustc_hash::FxHashMap;

/// Represents 1 to K community memberships for a single node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OhpMembership {
    pub communities: Vec<CommunityId>,
}

impl OhpMembership {
    pub fn new(primary: CommunityId, additional: &[CommunityId]) -> Self {
        let mut communities = Vec::with_capacity(1 + additional.len());
        communities.push(primary);
        for &c in additional {
            if !communities.contains(&c) {
                communities.push(c);
            }
        }
        OhpMembership { communities }
    }

    pub fn from_vec(vec: Vec<CommunityId>) -> Self {
        if vec.is_empty() {
            return OhpMembership { communities: vec![0] };
        }
        let mut communities = Vec::with_capacity(vec.len());
        for &c in &vec {
            if !communities.contains(&c) {
                communities.push(c);
            }
        }
        OhpMembership { communities }
    }

    #[inline]
    pub fn primary(&self) -> CommunityId {
        self.communities[0]
    }

    #[inline]
    pub fn secondary(&self) -> Option<CommunityId> {
        self.communities.get(1).copied()
    }

    #[inline]
    pub fn contains(&self, community: CommunityId) -> bool {
        self.communities.contains(&community)
    }

    #[inline]
    pub fn to_vec(&self) -> Vec<CommunityId> {
        self.communities.clone()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.communities.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.communities.is_empty()
    }
}

pub type OhpPartition = FxHashMap<NodeId, OhpMembership>;

pub fn crisp_to_ohp(partition: &Partition) -> OhpPartition {
    partition
        .iter()
        .map(|(&n, &c)| (n, OhpMembership::new(c, &[])))
        .collect()
}

pub fn ohp_to_crisp(ohp: &OhpPartition) -> Partition {
    ohp.iter().map(|(&n, m)| (n, m.primary())).collect()
}

#[derive(Clone, Debug)]
pub struct OhpIndividual {
    pub partition: OhpPartition,
    pub objectives: Vec<f64>,
    pub rank: usize,
    pub crowding_distance: f64,
}

impl OhpIndividual {
    pub fn new(partition: OhpPartition) -> Self {
        OhpIndividual {
            partition,
            objectives: vec![0.0, 0.0],
            rank: usize::MAX,
            crowding_distance: f64::MAX,
        }
    }

    pub fn to_individual(&self) -> Individual {
        Individual {
            partition: ohp_to_crisp(&self.partition),
            objectives: self.objectives.clone(),
            rank: self.rank,
            crowding_distance: self.crowding_distance,
        }
    }
}

impl From<Individual> for OhpIndividual {
    fn from(ind: Individual) -> Self {
        OhpIndividual {
            partition: crisp_to_ohp(&ind.partition),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn membership_validity() {
        let m1 = OhpMembership::new(5, &[]);
        assert_eq!(m1.len(), 1);
        assert_eq!(m1.to_vec(), vec![5]);
        assert!(m1.contains(5));
        assert!(!m1.contains(10));

        let m2 = OhpMembership::new(5, &[10, 15]);
        assert_eq!(m2.len(), 3);
        assert_eq!(m2.to_vec(), vec![5, 10, 15]);
        assert!(m2.contains(5));
        assert!(m2.contains(10));
        assert!(m2.contains(15));

        // Duplicate secondary should collapse
        let m3 = OhpMembership::new(5, &[5, 10, 10]);
        assert_eq!(m3.len(), 2);
        assert_eq!(m3.to_vec(), vec![5, 10]);
    }
}
