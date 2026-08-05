//! Default parameters for OHP-MOCD (overlapping extension of HP-MOCD).
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos.

pub const DEFAULT_DEBUG_LEVEL: i8 = 0;
pub const DEFAULT_POP_SIZE: usize = 100;
pub const DEFAULT_NUM_GENS: usize = 100;
pub const DEFAULT_CROSS_RATE: f64 = 0.7;
pub const DEFAULT_MUT_RATE: f64 = 0.5;

/// Maximum community memberships per node (3 = Top-K overlapping).
pub const DEFAULT_MAX_MEMBERSHIPS_PER_NODE: usize = 3;

/// Neighbourhood support threshold for overlap crossover/mutation.
pub const DEFAULT_OVERLAP_SUPPORT_THRESHOLD: f64 = 0.15;

/// Support below which a secondary membership is removed.
pub const DEFAULT_OVERLAP_REMOVAL_THRESHOLD: f64 = 0.08;

/// Margin required to switch primary membership.
pub const DEFAULT_SWITCH_MARGIN: f64 = 0.05;

/// Default settings for 3rd objective (f3) and 2-phase evolution schedule (from paper Eq. 6)
pub const DEFAULT_ENABLE_F3: bool = true;
pub const DEFAULT_TARGET_OVERLAP_RATE: f64 = 0.75;
pub const DEFAULT_ALPHA: f64 = 1.0;
pub const DEFAULT_PHASE1_RATIO: f64 = 0.25;

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
            overlap_probability: 0.40,
        }
    }
}
