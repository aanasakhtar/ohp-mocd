# OHP-MOCD: Overlapping High-Performance Multi-Objective Community Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/Rust-2021_Edition-orange.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Paper](https://img.shields.io/badge/Paper-Springer_LNCS-green.svg)](paper/main.tex)

**OHP-MOCD** is a high-performance evolutionary multi-objective framework for **overlapping community detection** in complex networks. It extends the modularity-decomposition paradigm of [HP-MOCD (Santos et al., 2025)](https://doi.org/10.1007/s13278-025-01519-7) to multi-membership covers through a direct chromosome encoding, a tri-objective Pareto formulation with overlap parsimony regularization, and an analytically proven, parameter-free statistical boundary bound based on the Central Limit Theorem.

The algorithm is implemented with a multi-threaded **Rust** core and exposed via **PyO3** Python bindings, achieving up to **27.5x computational speedups** over state-of-the-art evolutionary baselines.

---

## Key Features

* **Direct Node-to-Memberships Chromosome**: Eliminates locus-based decoding bottlenecks. Vertex membership lookup is O(1) and local genetic modifications require O(k) time.
* **Tri-Objective NSGA-II Optimization**: Jointly optimizes intra-community edge density (f1), null-model volume penalty (f2), and an explicit multi-membership parsimony regularizer (f3).
* **Topology-Aware Genetic Operators**: Features a 4-parent ensemble crossover mechanism and local-neighborhood majority mutation that naturally discover and preserve boundary multi-memberships.
* **Parameter-Free Statistical Null Boundary Bound**: Derives the secondary membership admission cutoff directly from the Central Limit Theorem and Chebyshev's inequality, eliminating manual heuristic threshold tuning.
* **High-Throughput Parallel Rust Core**: Implements lock-free parallel fitness evaluation and crossover via Rayon, scaling to large social ego-networks (30,000+ edges) in seconds.

---

## Repository Structure

```
ohp-mocd/
├── Cargo.toml                  # Rust workspace configuration
├── README.md                   # Public repository documentation
├── paper/                      # Publication manuscript and LaTeX source
│   ├── main.tex                # Springer svproc/LNCS manuscript
│   └── plots/                  # High-resolution benchmark figures (fig1-fig7)
├── src/                        # High-performance Rust core
│   ├── api.rs                  # Python PyO3 bindings and entrypoints
│   ├── lib.rs                  # Crate root
│   └── core/
│       ├── algorithms/         # Evolutionary algorithms (OHP-MOCD, HP-MOCD, etc.)
│       └── graph/              # Compressed graph representations and metrics
├── data/                       # Ground-truth network loaders and raw graphs
├── tests/benchmarks/           # Comprehensive benchmark suite & baselines
│   ├── baselines/              # Implementations of MCMOEA, EF-MOCD, MO-EE, SLPA, LPAM, NOCD
│   ├── utils/                  # Plotting, metrics (ONMI, Pairwise F1, EQ), and LFR generators
│   ├── run_paper_comparative_suite.py  # Master 12 real-world networks benchmark runner
│   ├── run_lfr_sweeps_and_plots.py     # Synthetic LFR benchmark sweep runner
│   └── master_overlapping_publication_table.csv # Aggregated real-world results
└── docs/                       # Theoretical and algorithmic references
```

---

## Installation

### Prerequisites
* [Rust](https://rustup.rs/) (v1.75+ recommended)
* [Python](https://www.python.org/) (v3.10+)
* [Maturin](https://github.com/PyO3/maturin) for building PyO3 bindings

### Building from Source

```bash
# Clone repository
git clone https://github.com/aanasakhtar/ohp-mocd.git
cd ohp-mocd

# Create and activate Python virtual environment
python -m venv .venv
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install Python build dependencies
pip install maturin networkx numpy scipy pandas scikit-learn matplotlib

# Build and install the Rust extension in release mode
maturin develop --release
```

---

## Quickstart & Python API

```python
import networkx as nx
import pymocd

# Load any NetworkX graph (with integer node IDs)
G = nx.karate_club_graph()

# Run OHP-MOCD overlapping community detection
# Returns a dictionary mapping node -> list of community IDs
communities = pymocd.ohpmocd(
    G,
    max_memberships_per_node=3,  # Maximum communities per node (1 = crisp, >1 = overlapping)
    pop_size=100,                # NSGA-II population size
    num_gens=100,                # Evolutionary generations
    seed=42                      # Random seed for reproducibility
)

print("Detected Community Memberships:")
for node, comms in sorted(communities.items())[:10]:
    print(f"  Node {node}: Communities {comms}")
```

---

## Benchmark Reproduction

To reproduce all 12 real-world ground-truth network evaluations and synthetic LFR sweeps reported in the paper:

### 1. Run Real-World Network Suite (12 Ground-Truth Datasets)
Evaluates OHP-MOCD against 6 baselines (MCMOEA, EF-MOCD, MO-EE, SLPA, LPAM, NOCD) across 15 independent seeds:
```bash
python tests/benchmarks/run_paper_comparative_suite.py
```
*Outputs: `tests/benchmarks/master_overlapping_publication_table.csv` and high-resolution figures `fig1`-`fig4`.*

### 2. Run Synthetic LFR Sweeps (mu in [0.1, 0.6])
Evaluates robustness against topological mixing noise:
```bash
python tests/benchmarks/run_lfr_sweeps_and_plots.py
```
*Outputs: `tests/benchmarks/lfr_synthetic_sweep_raw.csv` and degradation curve figures `fig5`-`fig7`.*

---

## Summary of Results

| Paradigm | Method | Mean ONMI (FB 414) | Mean F1 (FB 348) | Mean EQ (Polbooks) | Mean Runtime (Eu-core) |
|---|---|:---:|:---:|:---:|:---:|
| **Evolutionary (Ours)** | **OHP-MOCD** | **0.5146** | **0.8712** | **0.4942** | **14.08s** |
| Evolutionary | EF-MOCD (2020) | 0.3459 | 0.8390 | 0.3458 | 387.46s |
| Evolutionary | MO-EE (2018) | 0.3161 | 0.8249 | 0.0307 | 79.80s |
| Evolutionary | MCMOEA (2016) | 0.3129 | 0.2486 | 0.0474 | 61.95s |
| Label Propagation | SLPA (2011) | 0.4964 | 0.8304 | 0.4829 | 6.49s |
| Link Medoids | LPAM (2021) | 0.3743 | 0.5728 | 0.3459 | 92.78s |
| Graph Neural Net | NOCD (2019) | 0.3966 | 0.7768 | 0.1299 | 0.15s |

---

## Citation

If you use OHP-MOCD or its benchmark implementations in your research, please cite our paper:

```bibtex
@article{Anas2026OHPMOCD,
  author    = {Muhammad Anas and Manahil Jamil and Aneeza Khan},
  title     = {OHP-MOCD: Overlapping Community Detection via an Evolutionary Multi-Objective Extension of HP-MOCD},
  journal   = {Springer Lecture Notes in Computer Science (LNCS)},
  year      = {2026},
  url       = {https://github.com/aanasakhtar/ohp-mocd}
}
```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
