# OHP-MOCD Development Notes

**Status:** Phase 2 complete (crisp-compatible scaffold)  
**Baseline:** Fork of [pymocd](https://github.com/oliveira-sh/pymocd) — HP-MOCD remains unchanged  
**Goal:** Extend HP-MOCD into **OHP-MOCD** (overlapping communities: 1–2 memberships per node)

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

| Question | Answer |
|----------|--------|
| Chromosome | `Partition = FxHashMap<NodeId, CommunityId>` — one label per node |
| Shared `Individual` | Crisp only; do **not** change it for overlap |
| NSGA-II objectives | Generic over `objectives.len()` (2 or 3 OK) |
| Crossover | Majority vote over parent labels → one child label |
| Mutation | Node adopts neighbors’ majority community |
| Fitness | Shi decomposed modularity (`intra`, `inter`); assumes **disjoint** communities |
| Final pick | `max_q_selection`: maximize `n_obj − Σ objectives` (≈ max Q for 2 objs) |
| API surface | `api.rs` + `lib.rs` (+ stubs via `stub_gen`) |

**Implication:** Overlap needs an OHP-specific individual, operators, and objectives. Reuse NSGA-II ranking/survival; do not plug overlap into shared `create_offspring`.

---

## Phase 2 — Crisp-compatible scaffold (done)

### What was added

```text
src/core/algorithms/ohpmocd/
├── defaults.rs      # HP-MOCD defaults + overlap params (for later)
├── individual.rs    # OhpIndividual (still crisp Partition for now)
├── objectives.rs    # Delegates to decomposed modularity
├── operators.rs     # Seeded crisp operators + HP-MOCD reference helper
├── evolve.rs        # seed=None → shared nsga2; seed=Some → deterministic path
├── utils.rs         # max_q_selection (same as HP-MOCD)
└── mod.rs           # OhpMocd PyO3 class
```

Also wired into:

- `src/core/algorithms/mod.rs`
- `src/api.rs` → `pymocd.ohpmocd(...)`
- `src/lib.rs` → `OhpMocd` class
- `README.md` (minimal usage)

### Behavior

- Default: `max_memberships_per_node=1` → crisp, HP-MOCD-compatible pipeline
- Optional `seed` for reproducible runs / regression tests
- `max_memberships_per_node > 1` → `NotImplementedError` (until Phase 3+)
- **`hpmocd/` was not modified**

### Verified

```text
cargo test ohpmocd --no-default-features   # 6/6 passed
maturin develop --release
python -c "import pymocd; ..."             # ohpmocd(G, seed=42) works
```

### Python usage (current)

```python
import networkx as nx
import pymocd

G = nx.karate_club_graph()
part = pymocd.ohpmocd(G)                              # crisp
part = pymocd.ohpmocd(G, max_memberships_per_node=1, seed=42)

alg = pymocd.OhpMocd(G, max_memberships_per_node=1, seed=42)
part = alg.run()
```

### Local setup notes (Windows)

- Rust on **D:** via `RUSTUP_HOME` / `CARGO_HOME` (C: was too full)
- Project venv: `D:\Research\ohp-mocd\.venv` — deactivate conda before `maturin`
- Low RAM: use `$env:CARGO_BUILD_JOBS = "1"` for builds

---

## Phase 3 — Overlapping chromosome (next)

Add OHP-specific membership storage (do not change shared `Individual`):

```text
1 ≤ memberships(node) ≤ 2
primary always set; secondary optional and ≠ primary
```

Suggested shape: compact primary + optional secondary (indexed by node), with optional reverse `comm → nodes` cache invalidated after crossover/mutation.

Keep crisp path when `max_memberships_per_node=1`.

---

## Phase 4 — Initialization

Reuse crisp random init (one community per node).  
**Do not** inject overlap at init — overlap should emerge from operators.

---

## Phase 5 — Overlap-aware crossover

Extend ensemble logic:

- **primary** = majority parent label  
- **secondary** = runner-up, kept only if topology support passes  

```text
support(v, c) = |{u ∈ N(v) : c ∈ M(u)}| / |N(v)|
```

Keep secondary only if `support(v, secondary) ≥ overlap_support_threshold`.

---

## Phase 6 — Topology-guided mutation

OHP-only operators (do not replace shared `mutate`):

| Action | Rule |
|--------|------|
| Add overlap `[A] → [A,B]` | `support(v,B) ≥ overlap_support_threshold` |
| Remove `[A,B] → [A]` | `support(v,B) < overlap_removal_threshold` |
| Switch primary `[A] → [B]` | `support(v,B) − support(v,A) ≥ switch_margin` |

Target communities come from the local neighborhood only.

Defaults already sketched in `ohpmocd/defaults.rs`.

---

## Phase 7 — Objectives + NSGA-II

- Extend / replace decomposed modularity for overlapping memberships (current formula assumes disjoint communities).
- Candidate third objective:  
  `f3 = Overlap-Supported Edge Coverage − λ × Membership Complexity`
- NSGA-II already supports 3 objectives; **do not** reuse 2-obj `max_q_selection` blindly for 3 objs.
- Keep HP-MOCD / other algorithms unchanged.

---

## Phase 8 — Output + API

Return overlapping covers, e.g.:

```python
{node: [c1] | [c1, c2]}   # isolated → [-1]
```

Document + test. Crisp mode may keep `dict[node, community]` or always use lists for consistency (decide and document).

---

## Testing checklist (remaining)

1. Crisp regression: `max_memberships=1` matches HP-MOCD under same seed/params  
2. Membership validity: 1–2 unique communities per node  
3. Crossover: secondary only with topology support  
4. Mutation: add / remove / switch respect thresholds  
5. Output format for overlaps  
6. Existing algorithm tests still pass  

Use small hand-built graphs (e.g. two communities + one boundary node).

---

## Engineering rules

- Keep `hpmocd/` and other detectors intact  
- Prefer OHP-local modules over shared changes  
- Any shared change must be backward-compatible + tested  
- Deterministic seeds in tests  
- This is an **independent experimental extension**, not an official HP-MOCD paper implementation  

---

## Suggested owner split

| Phase | Focus |
|-------|--------|
| 3–4 | Representation + init |
| 5–6 | Crossover + mutation |
| 7 | Objectives + selection |
| 8 | Python API + docs + integration tests |

Questions / design decisions still open: exact overlap modularity definition, whether f3 is a true third objective vs a penalty, and final Python output shape.
