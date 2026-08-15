# OHP-MOCD Development Notes

**Status:** Phase 1–8 complete & Generalized to Top-K Overlapping Memberships  
**Baseline:** Fork of [pymocd](https://github.com/oliveira-sh/pymocd) — HP-MOCD remains unchanged  
**Goal:** Extend HP-MOCD into **OHP-MOCD** (overlapping communities: 1 to K memberships per node)

---

## Phase 1 — Repository analysis (done)

Documented the real HP-MOCD execution path before coding:

```text
Python API (hpmocd / HpMocd)
  → Graph::from_python
  → nsga2::evolve
      → generate_initial_population (random crisp labels)
      → evaluate (decomposed modularity: intra, inter)
      → tournament + ensemble crossover (4 parents) + neighbor-majority mutation
      → NSGA-II survivor selection (rank + crowding)
  → rank-1 filter → max_q_selection → normalize_community_ids
  → dict[node, community]
```

### Key findings

| Question            | Answer                                                                         |
| ------------------- | ------------------------------------------------------------------------------ |
| Chromosome          | `Partition = FxHashMap<NodeId, CommunityId>` — one label per node              |
| Shared `Individual` | Crisp only; do **not** change it for overlap                                   |
| NSGA-II objectives  | Generic over `objectives.len()` (2 or 3 OK)                                    |
| Crossover           | Majority vote over parent labels → one child label                             |
| Mutation            | Node adopts neighbors’ majority community                                      |
| Fitness             | Shi decomposed modularity (`intra`, `inter`); assumes **disjoint** communities |
| Final pick          | `max_q_selection`: maximize `n_obj − Σ objectives` (≈ max Q for 2 objs)        |
| API surface         | `api.rs` + `lib.rs` (+ stubs via `stub_gen`)                                   |

**Implication:** Overlap needs an OHP-specific individual, operators, and objectives. Reuse NSGA-II ranking/survival; do not plug overlap into shared `create_offspring`.

---

## Phase 2 — Crisp-compatible scaffold (done)

### What was added

```text
src/core/algorithms/ohpmocd/
├── defaults.rs      # HP-MOCD defaults + overlap params
├── individual.rs    # OhpIndividual + Top-K OhpMembership representation
├── objectives.rs    # Fractional-weight decomposed modularity
├── operators.rs     # Top-K overlap ensemble crossover + topology mutation
├── evolve.rs        # NSGA-II evolution loop for OhpIndividual
├── utils.rs         # max_q_selection_ohp
└── mod.rs           # OhpMocd PyO3 class
```

---

## Phase 3 — Top-K Overlapping Chromosome (done)

Implemented `OhpMembership` storing 1 to K memberships per node:

```rust
pub struct OhpMembership {
    pub communities: Vec<CommunityId>,
}
pub type OhpPartition = FxHashMap<NodeId, OhpMembership>;
```

- Primary community is at index 0 (`primary()`).
- Up to `max_memberships_per_node = K` unique community IDs per node.

---

## Phase 4 — Initialization (done)

Implemented `generate_population_ohp_seeded`:

- Random crisp initialization (1 primary community per node, additional memberships empty).
- Overlap emerges strictly through evolutionary operators.

---

## Phase 5 — Consensus Ensemble Crossover (done)

Implemented `ensemble_crossover_ohp_with_rng`:

- **primary**: majority parent label across 4 parents.
- **secondary memberships**: runner-up parent labels held by $\ge 50\%$ of parents.

---

## Phase 6 — Local-Move Majority Mutation (done)

Implemented `mutate_ohp_with_rng`:

- **Primary Community**: adopts neighborhood majority community $\arg\max_c |\{u \in N(v) : c \in M(u)\}|$.
- **Secondary Memberships**: any runner-up community with $\ge 2$ neighbor connections in $N(v)$.

---

## Phase 7 — Fractional Modularity Objectives + NSGA-II (done)

Implemented `calculate_ohp_objectives`:

- Extended Shi's decomposed modularity (`intra`, `inter`) using fractional node membership weights $r_{v, c} = 1 / |M(v)|$ for $|M(v)| \in [1, K]$.
- Parameter-free post-hoc boundary merge based strictly on $\Delta Q > 0$.
- Zero heuristic policing thresholds: variation is purely stochastic, and selection is governed strictly by NSGA-II non-dominated sorting.

---

## Phase 8 — Output + API (done)

- `OhpMocd` PyO3 class and `pymocd.ohpmocd(...)` function.
- Standard 4 GA parameters: `pop_size`, `num_gens`, `cross_rate`, `mut_rate`.
- Returns `dict[node, list[int]]`. Isolated nodes get `[-1]`.

---

## Engineering rules followed

- `hpmocd/` and other detectors kept completely intact.
- OHP-local modules created in `src/core/algorithms/ohpmocd/`.
- Deterministic seeds in all tests.
