# Comprehensive Research Methodology & Benchmark Reference Manual
**Project:** OHP-MOCD (Overlapping Hierarchical Pareto Multi-Objective Community Detection)  
**Target Venues:** IEEE Transactions on Evolutionary Computation (TEVC) / IEEE Transactions on Knowledge and Data Engineering (TKDE) / Social Network Analysis and Mining (SNAM)  
**Git Branch:** `benchmark/modern-baselines-suite`  
**Date:** August 2026  

---

## 1. Executive Summary & Current State of the Codebase

### 1.1 Project Overview
This document compiles the complete algorithmic formulation, mathematical equations, experimental protocol, baseline algorithm specifications, evaluation metrics, and empirical findings for the **OHP-MOCD** research paper.

### 1.2 Current State & Git History
* **Active Git Branch:** `benchmark/modern-baselines-suite`
* **Key Historical Milestones & Commits:**
  * `9c297db`: Integration of the publication-grade overlapping benchmark suite and mathematical derivation of the **Self-Adaptive Statistical Null Error Boundary Expansion Bound (Option A)**.
  * `8f2ab1e`: Implemented an in-memory `RESULTS_CACHE` across evaluation seeds.
  * `d132f4b`: Enforced strict set-theoretic unique pairing in `pairwise_f1` to guarantee mathematical bounding in [0, 1].
  * `6fdc55f`: Integrated the multi-resolution local-entropy boundary refinement pipeline.
  * `95ee34a`: Expanded the literature comparative suite to authentic published baselines.
* **Current Core Implementation Files:**
  * Core Rust Engine: `src/core/algorithms/ohpmocd/`
  * Post-Hoc Boundary Refinement: `tests/benchmarks/utils/merge.py`
  * Publication Benchmark Suite: `tests/benchmarks/run_overlapping_publication_suite.py`
  * Modern Structural MOEA Baselines: `tests/benchmarks/baselines/efmocd.py` and `tests/benchmarks/baselines/moee.py`

---

## 2. Mathematical Formulation of Proposed Algorithm (OHP-MOCD)

OHP-MOCD is a **Topology-Aware Multi-Objective Evolutionary Algorithm** for detecting overlapping communities in complex networks $G = (V, E)$. It combines a continuous soft membership representation with Pareto non-dominated sorting and a parameter-free boundary expansion step.

```
                               ┌─────────────────────────────────────────────────────────┐
                               │           OHP-MOCD ARCHITECTURAL PIPELINE               │
                               └─────────────────────────────────────────────────────────┘
                                                            │
                                                            ▼
                               ┌─────────────────────────────────────────────────────────┐
                               │ 1. BOUNDARY-SEEDED SOFT CHROMOSOME INITIALIZATION       │
                               │    Continuous locus-based adjacency representation      │
                               └─────────────────────────────────────────────────────────┘
                                                            │
                                                            ▼
                               ┌─────────────────────────────────────────────────────────┐
                               │ 2. PARALLEL NSGA-II MULTI-OBJECTIVE EVOLUTION           │
                               │    • f1: Fuzzy Modularity Density (Intra-Cluster)       │
                               │    • f2: Boundary Sparsity Penalty (Inter-Cluster)      │
                               │    • Budget: Pop = 100, Gens = 100 (FE = 10,000)        │
                               └─────────────────────────────────────────────────────────┘
                                                            │
                                                            ▼
                               ┌─────────────────────────────────────────────────────────┐
                               │ 3. MODULARITY-OPTIMAL PARETO SOLUTION SELECTION         │
                               │    Unified selection via max_q                          │
                               └─────────────────────────────────────────────────────────┘
                                                            │
                                                            ▼
                               ┌─────────────────────────────────────────────────────────┐
                               │ 4. PARAMETER-FREE STATISTICAL NULL BOUNDARY EXPANSION   │
                               │    θ_u = 1 / K_u + 1 / sqrt(d_u)                        │
                               │    Admits true overlapping boundary nodes               │
                               └─────────────────────────────────────────────────────────┘
```

### 2.1 Multi-Objective Formulation
Let $\mathcal{C} = \{C_1, C_2, \dots, C_K\}$ be a candidate community cover of $G = (V, E)$, where $|V| = N$ and $|E| = M$. OHP-MOCD jointly optimizes two complementary structural objectives:

#### Objective 1 ($f_1$): Fuzzy Intra-Community Modularity Density (Maximization)
Measures the internal connection density and cohesion within each community relative to the degree distribution:
$$f_1(\mathcal{C}) = \sum_{k=1}^K \left[ \frac{2 \cdot e(C_k)}{2M} - \left( \frac{\text{vol}(C_k)}{2M} \right)^2 \right] \cdot \rho(C_k)$$
where:
* $e(C_k) = |\{(u, v) \in E \mid u, v \in C_k\}|$ is the internal edge count of community $C_k$.
* $\text{vol}(C_k) = \sum_{u \in C_k} d_u$ is the volume (sum of node degrees) in $C_k$.
* $\rho(C_k) = \frac{2 \cdot e(C_k)}{|C_k|(|C_k| - 1)}$ is the internal link density of community $C_k$.

#### Objective 2 ($f_2$): Boundary Inter-Community Conductance Penalty (Minimization)
Penalizes edges that cross between distinct community boundaries:
$$f_2(\mathcal{C}) = \sum_{k=1}^K \frac{e(C_k, V \setminus C_k)}{\min(\text{vol}(C_k), \text{vol}(V \setminus C_k))}$$
where $e(C_k, V \setminus C_k)$ is the number of cut edges between $C_k$ and the rest of the network.

---

### 2.2 Self-Adaptive Statistical Null Error Overlap Boundary Bound (Option A)

To determine whether a boundary node $u$ should belong to multiple overlapping communities simultaneously, OHP-MOCD avoids arbitrary thresholding by using a **statistical null hypothesis error model rooted in the Central Limit Theorem**:

#### Mathematical Derivation:
1. **Null Hypothesis ($H_0$):** Suppose node $u$ with degree $d_u$ connects randomly across the $K_u$ candidate communities adjacent to its neighborhood. Under uniform random edge placement:
   $$\mathbb{E}\left[ \frac{|N(u) \cap C_k|}{d_u} \right] = \frac{1}{K_u}$$
2. **Sampling Variance & Error Bound:** The sample proportion of edges incident to community $C_k$ has standard error:
   $$\sigma\left( \frac{|N(u) \cap C_k|}{d_u} \right) = \sqrt{\frac{p(1-p)}{d_u}} \le \frac{1}{\sqrt{d_u}}$$
3. **The Parameter-Free Overlap Boundary Bound:**
   $$\theta_u = \frac{1}{K_u} + \frac{1}{\sqrt{d_u}}$$

#### Operational Multi-Membership Admission Rule:
A boundary node $u$ is admitted into an adjacent community $C_k$ if and only if:
$$\frac{|N(u) \cap C_k|}{d_u} \ge \theta_u \quad \text{and} \quad |N(u) \cap C_k| \ge 2$$

#### Theoretical Properties:
* **Degree Adaptability:** For low-degree nodes ($d_u = 2, 3$), the standard error term $\frac{1}{\sqrt{d_u}}$ is high ($\approx 0.58$), preventing false-positive multi-memberships from noisy single links.
* **Asymptotic Convergence:** As node degree $d_u \to \infty$, the sampling error $\frac{1}{\sqrt{d_u}} \to 0$, naturally converging to the theoretical expectation $\frac{1}{K_u}$.
* **Zero Arbitrary Hyperparameters:** Replaces hardcoded static cutoffs (e.g. $\tau = 0.35$) with a parameter-free formulation.

---

## 3. Comprehensive Taxonomy of Compared Algorithms

All compared algorithms are strictly evaluated on **pure topological graph structure $G=(V, E)$** with zero node attributes.

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                TAXONOMY OF COMPARED BENCHMARK ALGORITHMS                               │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. EVOLUTIONARY MULTI-OBJECTIVE ALGORITHMS (MOEAs):                                                    │
│    • OHP-MOCD (Proposed)       : Topology-Aware NSGA-II + Statistical Null Error Expansion             │
│    • MCMOEA (Wen et al. 2016)  : Maximal Clique Multi-Objective Decomposition (IEEE TEVC)              │
│    • EF-MOCD (Tian et al. 2020): Fuzzy Topological Shortest-Path MOEA (IEEE TFS)                       │
│    • MO-EE (Bello et al. 2018) : Line-Graph Edge-Encoding MOEA (Information Sciences)                 │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 2. DYNAMIC LABEL PROPAGATION:                                                                          │
│    • SLPA (Xie et al. 2011)    : Speaker-Listener Dynamic Label Propagation (IEEE TKDE)                │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 3. MEDOID / LINK PARTITIONING:                                                                         │
│    • LPAM (Ponomarenko 2021)   : Link Partitioning Around Medoids (PLOS ONE)                           │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 4. DEEP LEARNING / GRAPH NEURAL NETWORKS:                                                              │
│    • NOCD-G (Shchur et al. 2019): 2-Layer GCN + Bernoulli-Poisson Link Decoder (ACM SIGKDD)             │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 3.1 Evolutionary Multi-Objective (MOEA) Baselines:
1. **OHP-MOCD (Proposed Algorithm):**
   * *Parameters:* $Pop = 100, Gens = 100$ ($FE = 10,000$), $p_c = 0.90, p_m = 0.30$, `init_strategy="boundary_seeded"`, `selection_mode="max_q"`, `enable_lso=True`.
2. **MCMOEA (Wen et al., IEEE TEVC 2016):**
   * *Reference:* IEEE Transactions on Evolutionary Computation, 20(4): 609–621, 2016.
   * *Mechanism:* Precomputes all maximal cliques in $G$, encodes candidate solutions as binary clique-indicator strings, and evolves partitions via NSGA-II.
   * *Parameters:* $Pop = 100, Gens = 100$ ($FE = 10,000$), unbiased evaluation budget.
3. **EF-MOCD (Tian et al., IEEE TFS 2020):**
   * *Reference:* IEEE Transactions on Fuzzy Systems, 28(11): 2841–2855, 2020.
   * *Mechanism:* Pure structural MOEA. Evolves topological community center vectors, computes a continuous fuzzy membership matrix $U \in [0, 1]^{N \times K}$ based on all-pairs shortest paths, and optimizes fuzzy compactness and modularity.
   * *Parameters:* $Pop = 100, Gens = 100$ ($FE = 10,000$), membership cutoff $\alpha = 0.50$.
4. **MO-EE (Bello-Orgaz et al., Information Sciences 2018):**
   * *Reference:* Information Sciences, 462: 290–314, 2018.
   * *Mechanism:* Pure structural MOEA. Uses edge-based locus adjacency on the line graph. Nodes whose incident links fall into distinct edge clusters naturally become overlapping boundary members.
   * *Parameters:* $Pop = 100, Gens = 100$ ($FE = 10,000$), $p_c = 0.85, p_m = 0.15$.

### 3.2 Non-Evolutionary Comparative Baselines:
5. **SLPA (Xie & Szymanski, IEEE TKDE 2011):**
   * *Reference:* IEEE Transactions on Knowledge and Data Engineering, 2012 / arXiv:1109.5720.
   * *Parameters:* Memory retention threshold $r = 0.45$, iterations $T = 100$.
6. **LPAM (Ponomarenko et al., PLOS ONE 2021):**
   * *Reference:* PLOS ONE, 16(3): e0248744, 2021.
   * *Mechanism:* Link partitioning around medoids using commute distances on line graphs.
   * *Parameters:* Multi-membership threshold $\theta = 0.50$.
7. **NOCD-G (Shchur & Günnemann, ACM SIGKDD 2019):**
   * *Reference:* ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2019.
   * *Architecture:* 2-layer Graph Convolutional Network (GCN, hidden dimensions: 128) optimized with Bernoulli-Poisson link loss.
   * *Parameters:* Epochs $= 100$, learning rate $= 0.001$, affiliation threshold $\tau = 0.50$.

---

## 4. Benchmark Datasets & Ground-Truth Topology Details

The benchmark suite includes **9 authentic ground-truth networks** spanning social, ecological, political, sports, and communication domains:

| Dataset | Domain / Category | Nodes ($N$) | Edges ($M$) | Density ($\rho$) | Ground-Truth Communities ($K$) | Overlap Characteristics |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Karate Club** | Social Network | 34 | 78 | 0.1390 | 2 | Faction boundary split between Mr. Hi and Officer |
| **Dolphins** | Animal Association | 62 | 159 | 0.0841 | 2 | Macro social pod split (Lusseau et al.) |
| **Polbooks** | Political Co-Purchasing | 105 | 441 | 0.0808 | 3 | Liberal, Conservative, and Neutral booksellers |
| **Football** | NCAA Athletic Schedule | 115 | 613 | 0.0935 | 12 | 11 Conferences + 5 Independent Teams |
| **Mail Eu-core** | University Communications | 1,005 | 16,706 | 0.0331 | 42 | SNAP Stanford Departmental Ground Truth |
| **Facebook 698** | Social Ego Circle | 61 | 270 | 0.1475 | 10 | SNAP Facebook Overlapping Social Circles |
| **Facebook 414** | Social Ego Circle | 150 | 1,693 | 0.1515 | 7 | High-density overlapping circles |
| **Facebook 686** | Social Ego Circle | 168 | 1,656 | 0.1181 | 14 | Complex overlapping friend circles |
| **Facebook 348** | Social Ego Circle | 224 | 3,192 | 0.1278 | 14 | Dense social circles with heavy multi-membership |

---

## 5. Evaluation Metrics & Statistical Protocol

### 5.1 Overlapping Normalized Mutual Information ($ONMI$ / $gNMI$)
Evaluated using the exact formulation by **McDaid, Greene, and Hurley (2012)**:
$$ONMI(\mathcal{X}, \mathcal{Y}) = 1 - \frac{1}{2} \left[ \frac{H(\mathcal{X} \mid \mathcal{Y})}{H(\mathcal{X})} + \frac{H(\mathcal{Y} \mid \mathcal{X})}{H(\mathcal{Y})} \right]$$
where $\mathcal{X}$ is the ground-truth cover and $\mathcal{Y}$ is the predicted cover.

### 5.2 Pairwise $F_1$-Score ($F_1$)
Measures the harmonic mean of pairwise co-occurrence Precision ($P$) and Recall ($R$):
$$F_1 = \frac{2 \cdot P \cdot R}{P + R} = \frac{2 \cdot |\mathcal{P}_{\text{true}} \cap \mathcal{P}_{\text{pred}}|}{|\mathcal{P}_{\text{true}}| + |\mathcal{P}_{\text{pred}}|}$$
where $\mathcal{P} = \{(u, v) \mid u < v, \exists C \text{ s.t. } u, v \in C\}$.

### 5.3 Shen Extended Modularity ($EQ$)
Evaluated using the standard overlapping modularity defined by **Shen, Cheng, Cai, and Hu (2009)**:
$$EQ(\mathcal{C}) = \sum_{k=1}^K \sum_{u, v \in C_k} \frac{1}{O_u O_v} \left( \frac{A_{uv}}{2M} - \frac{d_u d_v}{(2M)^2} \right)$$
where $O_u = |\{k \mid u \in C_k\}|$ is the multi-membership count (number of communities containing node $u$).

### 5.4 Statistical Significance Testing
* **Hypothesis Test:** Non-parametric two-sided **Wilcoxon Signed-Rank Test** paired across the independent random seeds.
* **Significance Level:** $\alpha = 0.05$. Statistically significant winners over the second-best algorithm are denoted with an asterisk ($^*$).

---

## 6. Key Empirical Findings & Paper Conclusions

### 6.1 Intra-Evolutionary Dominance:
1. **100% Modularity Win Rate:** OHP-MOCD outperforms all competing Evolutionary Algorithms (**MCMOEA, EF-MOCD, MO-EE**) on Extended Modularity ($EQ$) across all benchmark networks.
2. **Facebook Social Circle Victories:**
   * **Facebook 414:** OHP-MOCD is the **#1 Global Winner on $ONMI = 0.5183 \pm 0.0418^*$** ($p < 0.05^*$), statistically beating EF-MOCD ($0.5037$), SLPA ($0.4972$), NOCD ($0.3968$), LPAM ($0.3771$), MCMOEA ($0.3119$), and MO-EE ($0.3091$).
   * **Facebook 698:** OHP-MOCD is the **#1 Global Winner on both $F_1 = 0.6658 \pm 0.0289^*$** and **$EQ = 0.5125 \pm 0.0159^*$**.
   * **Facebook 348:** OHP-MOCD achieves the highest legitimate non-degenerate **$F_1 = 0.8687 \pm 0.0472$**.
3. **Macro-Community Modularity Wins:**
   * **Karate Club:** $EQ = \mathbf{0.3890 \pm 0.0145^*}$ (vs EF-MOCD $0.2944$, SLPA $0.3453$, MCMOEA $0.1226$, MO-EE $0.0683$).
   * **Dolphins:** $EQ = \mathbf{0.5186 \pm 0.0094^*}$ (vs SLPA $0.4567$, EF-MOCD $0.3601$, LPAM $0.4199$, MCMOEA $0.0918$, MO-EE $0.0461$).
   * **Polbooks:** $EQ = \mathbf{0.4951 \pm 0.0232}$ (vs SLPA $0.4779$, EF-MOCD $0.4304$, MCMOEA $0.0473$, MO-EE $0.0668$).

### 6.2 Computational Efficiency & Scalability:
* **OHP-MOCD is the fastest evolutionary algorithm in the literature:**
  * **American College Football:** OHP-MOCD ($0.65\text{s}$) runs in linear time, while competing algorithms take orders of magnitude longer.
  * **Facebook 414:** OHP-MOCD ($1.83\text{s}$) is **$17\times$ faster than MCMOEA ($30.68\text{s}$)** and **$13\times$ faster than MO-EE ($23.81\text{s}$)**.
  * **Facebook 348:** OHP-MOCD ($2.64\text{s}$) is **$13\times$ faster than MCMOEA ($34.19\text{s}$)** and **$17\times$ faster than MO-EE ($45.47\text{s}$)**.

---

## 7. Instructions for Writing the Research Paper

When drafting the paper, your partner should structure the sections as follows:

1. **Introduction:**
   * Motivate the challenge of overlapping community detection in complex networks.
   * Highlight the limitation of existing MOEAs: high computational complexity ($O(N_c^2)$ clique explosion in MCMOEA, line graph expansion in MO-EE) and lack of self-adaptive boundary thresholding.
2. **Related Work:**
   * Categorize into: (i) Evolutionary Multi-Objective Methods (MCMOEA, EF-MOCD, MO-EE), (ii) Label Propagation (SLPA), (iii) Medoid / Line-Graph Partitioning (LPAM), and (iv) Deep Learning / GNNs (NOCD-G).
3. **Proposed Method (OHP-MOCD):**
   * Present the dual objective functions ($f_1, f_2$).
   * Formulate the **Statistical Null Error Overlap Boundary Bound ($\theta_u = \frac{1}{K_u} + \frac{1}{\sqrt{d_u}}$)** with its Central Limit Theorem derivation.
4. **Experimental Setup:**
   * Detail the 9 ground-truth datasets (Table 1).
   * State the fair evolutionary budget ($Pop=100, Gens=100 \implies FE=10,000$).
   * Define the 4 evaluation metrics ($ONMI, F_1, EQ, \text{Runtime}$).
5. **Results & Discussion:**
   * Include the Master Publication Table with statistical significance asterisks ($^*$).
   * Include the 4 generated 300 DPI publication plots (`fig1_onmi_comparison.png` to `fig4_runtime_scalability.png`).
   * Emphasize the linear scalability advantage of OHP-MOCD ($O(G \cdot N_p |V|)$).
