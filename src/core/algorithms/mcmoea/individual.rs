use rustc_hash::FxHashMap;
use crate::core::graph::{CommunityId, NodeId};

pub type CliqueLabelMap = Vec<CommunityId>;

#[derive(Clone, Debug)]
pub struct McmoeaIndividual {
    pub clique_labels: CliqueLabelMap,
    pub objectives: Vec<f64>,
    pub rank: usize,
    pub crowding_distance: f64,
}

impl McmoeaIndividual {
    pub fn new(clique_labels: CliqueLabelMap) -> Self {
        Self {
            clique_labels,
            objectives: vec![0.0, 0.0, 0.0],
            rank: 0,
            crowding_distance: 0.0,
        }
    }

    pub fn decode_node_memberships(&self, cliques: &[Vec<NodeId>]) -> FxHashMap<NodeId, Vec<CommunityId>> {
        let mut node_to_comms: FxHashMap<NodeId, Vec<CommunityId>> = FxHashMap::default();
        for (i, clique) in cliques.iter().enumerate() {
            let label = self.clique_labels[i];
            for &node in clique {
                let entry = node_to_comms.entry(node).or_default();
                if !entry.contains(&label) {
                    entry.push(label);
                }
            }
        }
        node_to_comms
    }
}
