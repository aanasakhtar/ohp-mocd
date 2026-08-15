//! Default parameters for OHP-MOCD (clean, threshold-free evolutionary architecture).
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos.

pub const DEFAULT_DEBUG_LEVEL: i8 = 0;
pub const DEFAULT_POP_SIZE: usize = 100;
pub const DEFAULT_NUM_GENS: usize = 100;
pub const DEFAULT_CROSS_RATE: f64 = 0.8;
pub const DEFAULT_MUT_RATE: f64 = 0.2;

/// Default settings for 3rd objective (f3: overlap complexity cost)
pub const DEFAULT_ENABLE_F3: bool = true;
pub const DEFAULT_PHASE1_RATIO: f64 = 0.0;

/// Population initialization strategies for OHP-MOCD.
#[derive(Clone, Debug, PartialEq)]
pub enum InitializationStrategy {
    /// Crisp initialization: random primary community per node, 0 additional communities.
    Crisp,
    /// Random overlap: assigns additional membership to randomly selected nodes.
    RandomOverlap { overlap_probability: f64 },
    /// Boundary seeded: identifies candidate boundary nodes and seeds additional membership.
    BoundarySeeded { overlap_probability: f64 },
}

impl Default for InitializationStrategy {
    fn default() -> Self {
        InitializationStrategy::BoundarySeeded {
            overlap_probability: 0.10,
        }
    }
}

