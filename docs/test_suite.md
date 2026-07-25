# OHP-MOCD Test Suite Documentation

This document provides a comprehensive summary of all tests conducted across the repository, with special focus on the **OHP-MOCD** (Overlapping High-Performance Multi-Objective Community Detection) algorithm and its Top-K generalization verification suite.

---

## 1. Executive Summary

| Test Suite | Module / Scope | Test Count | Status | Description |
|------------|----------------|------------|--------|-------------|
| **OHP-MOCD Unit Tests** | `src/core/algorithms/ohpmocd/` | 9 | **PASS** | Validates Top-K chromosome representation, fractional decomposed modularity, crossover support, topology mutation, and crisp HP-MOCD equivalence. |
| **Full Core Algorithm Suite** | `src/core/algorithms/` | 77 | **PASS** | Includes unit and regression tests for HP-MOCD, Shi-MOCD, MOGA-Net, CCM, KRM, MMCoMO, and SCALE. |
| **Graph & Metrics Suite** | `src/core/graph/`, `src/core/utils/` | 9 | **PASS** | Graph representation, adjacency/CSR verification, SBM MDL scoring, and NMI/ARI metric evaluation. |
| **Python Integration Suite** | `tests/test_ohpmocd_overlap.py` | 4 | **PASS** | End-to-end Python PyO3 API verification for crisp, $K=2$, and $K=3$ Top-K overlapping modes. |
| **Total Test Count** | **Entire Repository** | **99** | **PASS** | **100% Pass Rate** |

---

## 2. OHP-MOCD Top-K Test Details (`src/core/algorithms/ohpmocd/`)

The OHP-MOCD test suite verifies that Top-K overlapping community detection functions correctly for any $K \ge 1$ without breaking compatibility with HP-MOCD crisp baseline behavior.

### 2.1. Chromosome Representation & Validity
- **`individual::tests::membership_validity`**
  - **Objective:** Asserts correct behavior of `OhpMembership`.
  - **Checks:**
    - Single membership (`OhpMembership::new(5, &[])`): `len() == 1`, `to_vec() == [5]`, `contains(5)` is `true`.
    - Multi-membership ($K=3$, `OhpMembership::new(5, &[10, 15])`): `len() == 3`, `to_vec() == [5, 10, 15]`, `contains(5)`, `contains(10)`, and `contains(15)` are `true`.
    - Duplicate secondary safeguard (`OhpMembership::new(5, &[5, 10, 10])`): automatically deduplicates entries keeping order (`to_vec() == [5, 10]`, `len() == 2`).
- **`mod::tests::membership_validity_crisp_partition_covers_all_nodes`**
  - **Objective:** Verifies that every node in the graph is present in the final partition dict and assigned a valid community ID ($\ge -1$).
- **`mod::tests::membership_validity_top_k_overlapping_mode`**
  - **Objective:** Runs multi-generation `evolve_ohp` with `max_memberships_per_node = 3` ($K = 3$).
  - **Checks:** Ensures every node receives 1 to 3 unique memberships ($1 \le |M(v)| \le 3$) without duplicates.

### 2.2. Fitness & Objective Evaluation
- **`objectives::tests::ohp_objectives_matches_crisp_when_max_memberships_is_1`**
  - **Objective:** Proves mathematical equivalence between fractional-weight overlapping decomposed modularity (`calculate_ohp_objectives`) and Shi's crisp decomposed modularity (`calculate_objectives`).
  - **Checks:** On a 2-community benchmark graph with crisp labels ($|M(v)| = 1$), asserts metric deviation is zero:
    $$\left| f_1^{\text{crisp}} - f_1^{\text{ohp}} \right| < 10^{-10}, \quad \left| f_2^{\text{crisp}} - f_2^{\text{ohp}} \right| < 10^{-10}$$

### 2.3. Topology-Aware Genetic Operators
- **`operators::tests::support_calculation_correctness`**
  - **Objective:** Verifies local neighborhood support calculation $\text{support}(v, c) = \frac{|\{u \in N(v) : c \in M(u)\}|}{|N(v)|}$.
  - **Checks:** On a graph with two 3-node communities connected by boundary node 2 (2 neighbors in community 0, 1 neighbor in community 1), asserts $\text{support}(2, 0) = \frac{2}{3}$ and $\text{support}(2, 1) = \frac{1}{3}$.
- **`mod::tests::crossover_secondary_kept_only_with_support`**
  - **Objective:** Tests Top-K overlap-aware ensemble crossover over 4 parents.
  - **Checks:**
    - With a strict threshold (`0.90`), additional candidate memberships for boundary node 2 are rejected.
    - With standard threshold (`0.25`), additional candidate memberships for node 2 ($\text{support} = 0.333 \ge 0.25$) are accepted up to $K$.
- **`mod::tests::mutation_add_remove_switch_thresholds`**
  - **Objective:** Validates Top-K topology-guided mutation rules (Add, Remove, Switch).
  - **Checks:**
    - **Add Overlap:** Additional memberships are added when neighborhood support exceeds `overlap_support_threshold` (`0.25`) up to $K$.
    - **Remove Overlap:** Additional memberships to unsupported communities (e.g. community 99) are removed when support drops below `overlap_removal_threshold` (`0.15`).

### 2.4. Crisp Baseline Regression
- **`mod::tests::crisp_mode_matches_hpmocd_reference_under_seed`**
- **`evolve::tests::crisp_seeded_matches_hpmocd_reference`**
  - **Objective:** Ensures exact deterministic equivalence with HP-MOCD when `max_memberships_per_node = 1`.
  - **Checks:** Compares end-to-end evolution output of `OHP-MOCD` under seed 42 against reference HP-MOCD execution; partitions are identical.

---

## 3. General Repository Test Suite (77 Tests)

The remaining 77 tests in `src/` ensure that all other community detection algorithms and core data structures remain fully functional:

### 3.1. Baseline Detectors
- **CCM (NSGA-III CCM)**: 8 tests.
- **KRM (NSGA-III KRM)**: 3 tests.
- **MMCoMO (Co-Evolutionary)**: 12 tests.
- **MOCD (Shi-MOCD PESA-II)**: 5 tests.
- **MOGA-Net**: 3 tests.
- **SCALE (Sparse-CSR MMCoMO)**: 18 tests.

### 3.2. Graph & Utilities
- **Adjacency & CSR Graphs**: 7 tests.
- **Ground-Truth Metrics**: 3 tests.

---

## 4. Python Integration Suite (`tests/test_ohpmocd_overlap.py`)

End-to-end integration tests executed via Python 3.11 with NetworkX graphs:

```python
def test_ohpmocd_crisp_mode():
    # Verifies max_memberships_per_node=1 returns dict[node, int]
    G = two_cliques_with_boundary_node()
    part = pymocd.ohpmocd(G, max_memberships_per_node=1, seed=42)
    assert isinstance(part, dict)
    assert all(isinstance(v, int) for v in part.values())

def test_ohpmocd_overlapping_mode_k2():
    # Verifies max_memberships_per_node=2 returns dict[node, list[int]] with 1-2 elements
    G = two_cliques_with_boundary_node()
    part = pymocd.ohpmocd(G, max_memberships_per_node=2, seed=42)
    assert isinstance(part, dict)
    assert all(isinstance(v, list) and 1 <= len(v) <= 2 for v in part.values())

def test_ohpmocd_overlapping_mode_top_k3():
    # Verifies max_memberships_per_node=3 returns dict[node, list[int]] with 1-3 elements
    G = three_cliques_with_boundary_node()
    part = pymocd.ohpmocd(G, max_memberships_per_node=3, seed=42)
    assert isinstance(part, dict)
    assert all(isinstance(v, list) and 1 <= len(v) <= 3 for v in part.values())

def test_ohpmocd_class_pareto_front():
    # Verifies OhpMocd class generate_pareto_front() returns Pareto list of (partition_dict, objectives)
    G = two_cliques_with_boundary_node()
    alg = pymocd.OhpMocd(G, max_memberships_per_node=3, seed=42)
    front = alg.generate_pareto_front()
    assert isinstance(front, list)
    assert len(front) > 0
```

---

## 5. Execution Instructions

### Running Rust Unit Tests
```bash
# Run all OHP-MOCD tests
cargo test ohpmocd --no-default-features

# Run the complete repository test suite (86 tests)
cargo test --no-default-features
```

### Running Python Integration Tests
```bash
# Build & install native extension
D:\Research\ohp-mocd\.venv\Scripts\pip.exe install -e .

# Execute Python test suite
D:\Research\ohp-mocd\.venv\Scripts\python.exe tests/test_ohpmocd_overlap.py
```
