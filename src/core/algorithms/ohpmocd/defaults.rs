//! Default parameters for OHP-MOCD (overlapping extension of HP-MOCD).
//! This Source Code Form is subject to the terms of The GNU General Public License v3.0
//! Copyright 2025 - Guilherme Santos.

pub const DEFAULT_DEBUG_LEVEL: i8 = 0;
pub const DEFAULT_POP_SIZE: usize = 100;
pub const DEFAULT_NUM_GENS: usize = 100;
pub const DEFAULT_CROSS_RATE: f64 = 0.7;
pub const DEFAULT_MUT_RATE: f64 = 0.5;

/// Maximum community memberships per node (1 = crisp, compatible with HP-MOCD).
pub const DEFAULT_MAX_MEMBERSHIPS_PER_NODE: usize = 1;

/// Neighbourhood support threshold for overlap crossover/mutation (Phase 5+).
pub const DEFAULT_OVERLAP_SUPPORT_THRESHOLD: f64 = 0.25;

/// Support below which a secondary membership is removed (Phase 6+).
pub const DEFAULT_OVERLAP_REMOVAL_THRESHOLD: f64 = 0.15;

/// Margin required to switch primary membership (Phase 6+).
pub const DEFAULT_SWITCH_MARGIN: f64 = 0.2;
