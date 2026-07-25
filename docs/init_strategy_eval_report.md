# OHP-MOCD Population Initialization Strategy Benchmark Report

This document presents the experimental evaluation of pluggable population initialization strategies for **OHP-MOCD** (Overlapping High-Performance Multi-Objective Community Detection).

---

## 1. Executive Summary & Key Findings

| Strategy | DBLP Max Q (mean) | DBLP NMI | LFR Overlap Max Q | LFR Overlap NMI | Runtime (DBLP) | Recommendation |
|----------|-------------------|----------|-------------------|-----------------|----------------|----------------|
| **`BoundarySeeded`** | **0.5862** | **0.6353** | **0.5182** | 0.6969 | 1904.6 ms | **RECOMMENDED** |
| **`Crisp`** | 0.5244 | 0.6203 | 0.5151 | **0.6994** | 1742.7 ms | Baseline |
| **`RandomOverlap`** | 0.5774 | 0.6176 | 0.4902 | 0.6827 | 1701.8 ms | Not Recommended |

> [!TIP]
> **Key Insight:** Seeding 2nd memberships exclusively for **topology-identified boundary nodes** (`BoundarySeeded`) provides substantial performance gains on real-world graphs like DBLP ($\Delta Q = +0.0618$, $+11.8\%$ modularity gain over crisp initialization), while random overlap seeding introduces noisy labels that slow down evolution.

---

## 2. Pluggable Initialization Architecture

The initialization pipeline in Rust (`src/core/algorithms/ohpmocd/`) was refactored with the pluggable `InitializationStrategy` enum:

```rust
#[derive(Clone, Debug, PartialEq)]
pub enum InitializationStrategy {
    /// Crisp initialization (reproduces default HP-MOCD behavior): 1 primary community, 0 additional.
    Crisp,
    /// Random overlap: assigns additional membership to randomly selected nodes with probability p.
    RandomOverlap { overlap_probability: f64 },
    /// Boundary seeded: identifies candidate boundary nodes (neighbors in > 1 primary communities)
    /// and seeds additional membership with probability p.
    BoundarySeeded { overlap_probability: f64 },
}
```

### Strategy Mechanics:
1. **`Crisp`**: Assigns random primary communities to all nodes ($M(v) = [c_{prim}]$).
2. **`RandomOverlap`**: Assigns random primary communities, then with probability $p$, adds a 2nd membership from neighbor/random communities for any node.
3. **`BoundarySeeded`**: Identifies candidate boundary nodes whose neighbors span $> 1$ primary communities, then with probability $p$, seeds a 2nd membership using the top runner-up neighbor community.

---

## 3. Dataset-Level Empirical Results (20 Random Seeds)

### 3.1. DBLP Co-Authorship Network (500 Subsampled Nodes, 1,634 Edges)
| Strategy | Intra Modularity | Inter Modularity | Max Q ($1 - \text{intra} - \text{inter}$) | NMI | AMI | ARI | Overlapping Nodes |
|----------|------------------|------------------|-------------------------------------------|-----|-----|-----|-------------------|
| **BoundarySeeded** | 0.3120 | 0.1018 | **0.5862 ± 0.0319** | **0.6353** | **0.2799** | **0.0941** | 202.55 |
| **Crisp** | 0.3705 | 0.1051 | **0.5244 ± 0.0586** | 0.6203 | 0.2501 | 0.0675 | 198.55 |
| **RandomOverlap** | 0.3208 | 0.1018 | **0.5774 ± 0.0550** | 0.6176 | 0.2667 | 0.0786 | 196.75 |

### 3.2. Synthetic LFR Overlapping Benchmark (250 Nodes, 478 Edges, 50 Overlapping Ground-Truth)
| Strategy | Max Q | NMI | AMI | ARI | Overlapping Nodes | Avg Memberships |
|----------|-------|-----|-----|-----|-------------------|-----------------|
| **BoundarySeeded** | **0.5182 ± 0.0598** | 0.6969 | 0.6138 | 0.4389 | 153.25 | 1.6130 |
| **Crisp** | 0.5151 ± 0.0565 | **0.6994** | **0.6143** | **0.4409** | 155.10 | 1.6204 |
| **RandomOverlap** | 0.4902 ± 0.0688 | 0.6827 | 0.5833 | 0.3936 | 161.55 | 1.6462 |

---

## 4. Exported Result Files

All raw run-by-run metrics and per-generation convergence logs are saved in CSV format for analysis:

1. **Summary CSV:** [init_strategy_summary.csv](file:///D:/Research/ohp-mocd/src/core/algorithms/data/init_strategy_summary.csv)
   - Columns: `dataset`, `strategy`, `overlap_prob`, `seed`, `runtime_ms`, `intra`, `inter`, `max_Q`, `nmi`, `ami`, `ari`, `num_overlapping_nodes`, `avg_memberships`, `max_memberships`.
2. **Convergence CSV:** [init_strategy_convergence.csv](file:///D:/Research/ohp-mocd/src/core/algorithms/data/init_strategy_convergence.csv)
   - Columns: `dataset`, `strategy`, `seed`, `generation`, `best_Q`.

---

## 5. Usage in Python & Rust

```python
import pymocd

# Run OHP-MOCD with BoundarySeeded initialization strategy
part = pymocd.ohpmocd(
    G,
    max_memberships_per_node=2,
    init_strategy="boundary_seeded",
    init_overlap_prob=0.2,
    seed=42
)
```
