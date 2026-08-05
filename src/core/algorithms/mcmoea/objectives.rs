use rustc_hash::FxHashMap;
use crate::core::graph::{CommunityId, Graph, NodeId};
use crate::core::algorithms::mcmoea::cliques::Clique;
use crate::core::algorithms::mcmoea::individual::McmoeaIndividual;

pub struct McmoeaEvaluator {
    pub total_edges: f64,
    pub total_nodes: f64,
    pub clique_internal_edges: Vec<f64>,
    pub clique_pair_shared_edges: FxHashMap<(usize, usize), f64>,
}

impl McmoeaEvaluator {
    pub fn new(graph: &Graph, cliques: &[Clique]) -> Self {
        let total_edges = graph.num_edges() as f64;
        let total_nodes = graph.num_nodes() as f64;
        let num_cliques = cliques.len();

        let mut clique_internal_edges = vec![0.0; num_cliques];
        let mut clique_pair_shared_edges: FxHashMap<(usize, usize), f64> = FxHashMap::default();

        // Build node -> cliques mapping
        let mut node_cliques: FxHashMap<NodeId, Vec<usize>> = FxHashMap::default();
        for (cid, clique) in cliques.iter().enumerate() {
            for &v in clique {
                node_cliques.entry(v).or_default().push(cid);
            }
        }

        // Count edges inside and between cliques
        for (&u, nbrs) in &graph.adjacency_list {
            for &v in nbrs {
                if u >= v {
                    continue;
                }
                if let (Some(c_u), Some(c_v)) = (node_cliques.get(&u), node_cliques.get(&v)) {
                    for &cu in c_u {
                        for &cv in c_v {
                            if cu == cv {
                                clique_internal_edges[cu] += 1.0;
                            } else {
                                let pair = if cu < cv { (cu, cv) } else { (cv, cu) };
                                *clique_pair_shared_edges.entry(pair).or_insert(0.0) += 0.5;
                            }
                        }
                    }
                }
            }
        }

        Self {
            total_edges,
            total_nodes,
            clique_internal_edges,
            clique_pair_shared_edges,
        }
    }

    pub fn calculate_objectives(
        &self,
        individual: &mut McmoeaIndividual,
        cliques: &[Clique],
    ) {
        if self.total_edges == 0.0 || self.total_nodes == 0.0 {
            individual.objectives = vec![1.0, 1.0, 1.0];
            return;
        }

        let num_cliques = individual.clique_labels.len();

        // 1. Calculate intra-community edges
        let mut intra_edges = 0.0;
        for i in 0..num_cliques {
            intra_edges += self.clique_internal_edges[i];
        }

        for (&(i, j), &w) in &self.clique_pair_shared_edges {
            if individual.clique_labels[i] == individual.clique_labels[j] {
                intra_edges += w;
            }
        }

        intra_edges = intra_edges.min(self.total_edges);

        let f1 = 1.0 - (intra_edges / self.total_edges);
        let f2 = (self.total_edges - intra_edges) / self.total_edges;

        // 2. Overlap complexity
        let node_memberships = individual.decode_node_memberships(cliques);
        let mut overlapping_nodes = 0.0;
        for comms in node_memberships.values() {
            if comms.len() > 1 {
                overlapping_nodes += 1.0;
            }
        }
        let f3 = overlapping_nodes / self.total_nodes;

        individual.objectives = vec![f1, f2, f3];
    }
}
