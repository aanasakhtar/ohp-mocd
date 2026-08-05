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

## Phase 5 — Top-K Overlap-Aware Crossover (done)

Implemented `ensemble_crossover_ohp_with_rng`:

- **primary**: majority parent label across 4 parents.
- **additional memberships**: runner-up parent labels evaluated in order of parent frequency, kept if topology support passes:
  `support(v, c) = |{u ∈ N(v) : c ∈ M(u)}| / |N(v)| ≥ overlap_support_threshold` (default 0.25), up to $K$ total memberships.

---

## Phase 6 — Top-K Topology-Guided Mutation (done)

Implemented `mutate_ohp_with_rng`:

- **Add overlap**: adds top-supported neighbor communities $B$ with `support(v, B) ≥ overlap_support_threshold` (0.25) up to $K$ memberships.
- **Remove overlap**: removes additional memberships $B$ with `support(v, B) < overlap_removal_threshold` (0.15).
- **Switch primary**: switches primary community if a neighbor community has higher support by `switch_margin` (0.20).

---

## Phase 7 — Fractional Modularity Objectives + NSGA-II (done)

Implemented `calculate_ohp_objectives`:

- Extended Shi's decomposed modularity (`intra`, `inter`) using fractional node membership weights $r_{v, c} = 1 / |M(v)|$ for $|M(v)| \in [1, K]$.
- Reduces identically to crisp decomposed modularity when $|M(v)| = 1$.
- `max_q_selection_ohp` selects the solution maximizing modularity $Q = 1 - \text{intra} - \text{inter}$ from the NSGA-II rank-1 Pareto front.

### Post-benchmark tuning decisions

After the DBLP benchmark run in this branch, the overlapping mode was under-detecting boundary nodes and was much slower than the disjoint baseline. The following tuning decisions were applied:

1. **Selection no longer clones OHP partitions into crisp placeholders.**

- Survivor selection now ranks OHP individuals directly from their objective vectors.
- This removes repeated `OhpPartition -> Partition -> OhpPartition` conversions during every generation.

2. **Neighborhood support is cached per node during evaluation and operator steps.**

- `f3`, crossover, and mutation now reuse a single neighbor-community histogram per node instead of rescanning the same adjacency list repeatedly.
- This keeps the support logic the same but cuts redundant work.

3. **Offspring generation is parallelized deterministically.**

- Parent plans are sampled once, then child construction runs in parallel with child-specific RNG seeds.
- This preserves seeded reproducibility while using CPU cores more effectively.

4. **Overlap pressure was increased for the high-overlap benchmark regime.**

- The default target overlap rate was raised from $0.20$ to $0.75$.
- The crossover/mutation thresholds were relaxed so secondary memberships survive more often.
- Phase 1, where $f_3$ is disabled, was shortened so overlap pressure starts earlier.

5. **The crisp objective path remains exact.**

- Crisp-mode OHP still reduces to the original HP-MOCD modularity objectives.
- A regression test now checks this directly.

### Why these choices

The benchmarked DBLP graph reports about $75.5\%$ overlapping nodes, while the earlier OHP settings only pushed toward roughly $20\%$ overlap and repeatedly discarded borderline secondary memberships. The new defaults are intentionally more permissive so the algorithm can actually express the overlap density seen in the dataset instead of being capped by overly conservative thresholds.

---

## Phase 8 — Output + API (done)

- `OhpMocd` PyO3 class and `pymocd.ohpmocd(...)` function.
- Crisp mode (`max_memberships_per_node=1`) returns `dict[node, int]`.
- Top-K Overlapping mode (`max_memberships_per_node >= 2`) returns `dict[node, list[int]]` (containing 1 to K memberships per node). Isolated nodes get `[-1]`.

---

## Testing checklist (all passed)

1. Crisp regression: `max_memberships=1` matches HP-MOCD under same seed/params (passed)
2. Membership validity: 1 to K unique communities per node (passed for K=2, K=3)
3. Crossover: additional memberships kept only with topology support (passed)
4. Mutation: add / remove / switch respect thresholds (passed)
5. Output format for overlaps (passed)
6. All 86 repository unit tests pass cleanly (passed)

---

## Engineering rules followed

- `hpmocd/` and other detectors kept completely intact.
- OHP-local modules created in `src/core/algorithms/ohpmocd/`.
- Deterministic seeds in all tests.
