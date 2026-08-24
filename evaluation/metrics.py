"""Evaluation metrics for disjoint and overlapping community detection.

Disjoint metrics: NMI, AMI, Modularity, Pairwise F1.
Overlapping metrics: Omega Index, Overlapping NMI (ONMI).

For overlapping covers we project nodes to their best-supported community
when computing projected metrics (NMI, Modularity) so reported values are
consistent with the algorithm's internal projection. Use `onmi()` for the
true overlapping NMI (McDaid et al., 2011).
"""

from __future__ import annotations
import itertools
import numpy as np
import networkx as nx
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score


# Helpers

def _partition_to_label_array(
    communities: list[frozenset],
    nodes: list,
) -> np.ndarray:
    label = np.full(len(nodes), -1, dtype=int)
    node_idx = {v: i for i, v in enumerate(nodes)}
    for cid, comm in enumerate(communities):
        for v in comm:
            if v in node_idx and label[node_idx[v]] == -1:
                label[node_idx[v]] = cid
    return label


def _cover_to_label_array_support_based(
    communities: list[frozenset],
    nodes: list,
    G: nx.Graph | None = None,
) -> np.ndarray:
    node_idx = {v: i for i, v in enumerate(nodes)}
    label = np.full(len(nodes), -1, dtype=int)

    node_to_comms: dict = {}
    for cid, comm in enumerate(communities):
        for v in comm:
            node_to_comms.setdefault(v, {})[cid] = comm

    for v, cid_map in node_to_comms.items():
        if v not in node_idx:
            continue
        if len(cid_map) == 1:
            label[node_idx[v]] = next(iter(cid_map))
            continue
        if G is not None and v in G:
            nbrs = set(G.neighbors(v))
            best_cid = max(
                cid_map.keys(),
                key=lambda cid: len(nbrs & cid_map[cid])
            )
        else:
            best_cid = max(cid_map.keys(), key=lambda cid: len(cid_map[cid]))
        label[node_idx[v]] = best_cid

    return label


def _cover_to_hard_partition(
    communities: list[frozenset],
    G: nx.Graph | None = None,
) -> list[frozenset]:
    """
    Project an overlapping cover to a disjoint hard partition covering 100% of nodes in G.
    """
    if G is not None:
        nodes = list(G.nodes())
    else:
        nodes = list(set().union(*communities)) if communities else []

    node_idx = {v: i for i, v in enumerate(nodes)}
    labels = _cover_to_label_array_support_based(communities, nodes, G=G)

    groups: dict[int, set] = {}
    for idx, v in enumerate(nodes):
        cid = int(labels[idx])
        if cid >= 0:
            groups.setdefault(cid, set()).add(v)
        else:
            groups.setdefault(-1000000 - idx, set()).add(v)

    return [frozenset(members) for members in groups.values() if members]


def _is_overlapping(communities: list[frozenset]) -> bool:
    seen: set = set()
    for comm in communities:
        for v in comm:
            if v in seen:
                return True
            seen.add(v)
    return False


def nmi(
    pred: list[frozenset],
    true: list[frozenset],
    G: nx.Graph | None = None,
) -> float:
    nodes = list(set().union(*pred, *true))
    pred_overlapping = _is_overlapping(pred)

    if pred_overlapping:
        y_pred = _cover_to_label_array_support_based(pred, nodes, G=G)
    else:
        y_pred = _partition_to_label_array(pred, nodes)

    y_true = _partition_to_label_array(true, nodes)
    mask = (y_pred != -1) & (y_true != -1)
    if mask.sum() == 0:
        return 0.0
    return normalized_mutual_info_score(y_true[mask], y_pred[mask])


def ami(
    pred: list[frozenset],
    true: list[frozenset],
    G: nx.Graph | None = None,
) -> float:
    nodes = list(set().union(*pred, *true))
    pred_overlapping = _is_overlapping(pred)

    if pred_overlapping:
        y_pred = _cover_to_label_array_support_based(pred, nodes, G=G)
    else:
        y_pred = _partition_to_label_array(pred, nodes)

    y_true = _partition_to_label_array(true, nodes)
    mask = (y_pred != -1) & (y_true != -1)
    if mask.sum() == 0:
        return 0.0
    return adjusted_mutual_info_score(y_true[mask], y_pred[mask])


def modularity(G: nx.Graph, communities: list[frozenset]) -> float:
    return nx.community.modularity(G, communities)


def pairwise_f1(pred: list[frozenset], true: list[frozenset]) -> float:
    """
    Pairwise F1 score (paper Section 5.1.1).
    Computes exact true positive pairs via community intersections with O(1) memory footprint.
    """
    n_pred_pairs = sum(len(c) * (len(c) - 1) // 2 for c in pred)
    n_true_pairs = sum(len(c) * (len(c) - 1) // 2 for c in true)

    if n_pred_pairs == 0 or n_true_pairs == 0:
        return 0.0

    tp = 0
    for cp in pred:
        for ct in true:
            k = len(cp & ct)
            if k >= 2:
                tp += k * (k - 1) // 2

    precision = tp / n_pred_pairs
    recall = tp / n_true_pairs

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def omega_index(pred: list[frozenset], true: list[frozenset]) -> float:
    nodes = list(set().union(*pred, *true))
    node_idx = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)

    def co_occurrence_count(communities: list[frozenset], ni: int) -> np.ndarray:
        count = np.zeros((ni, ni), dtype=np.int32)
        for c in communities:
            members = [node_idx[v] for v in c if v in node_idx]
            for a, b in itertools.combinations(members, 2):
                count[a, b] += 1
                count[b, a] += 1
        return count

    pred_co = co_occurrence_count(pred, n)
    true_co = co_occurrence_count(true, n)

    max_k = max(pred_co.max(), true_co.max(), 1)
    pairs_total = n * (n - 1) / 2
    if pairs_total == 0:
        return 1.0

    observed = sum(
        np.sum((pred_co == k) & (true_co == k)) / 2
        for k in range(max_k + 1)
    )

    expected = sum(
        (np.sum(pred_co == k) / 2) * (np.sum(true_co == k) / 2) / (pairs_total ** 2)
        * pairs_total
        for k in range(max_k + 1)
    )

    denom = pairs_total - expected
    if denom <= 0:
        return 1.0

    return (observed - expected) / denom


def onmi(pred: list[frozenset], true: list[frozenset]) -> float:
    def _h(p: float) -> float:
        if p <= 0 or p >= 1:
            return 0.0
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

    nodes = list(set().union(*pred, *true))
    n = len(nodes)
    if n == 0:
        return 1.0
    node_idx = {v: i for i, v in enumerate(nodes)}

    def _membership_matrix(communities: list[frozenset]) -> np.ndarray:
        mat = np.zeros((len(communities), n), dtype=np.float32)
        for cid, comm in enumerate(communities):
            for v in comm:
                if v in node_idx:
                    mat[cid, node_idx[v]] = 1.0
        return mat

    P = _membership_matrix(pred)
    T = _membership_matrix(true)

    def _community_entropy(mat: np.ndarray) -> np.ndarray:
        p = mat.mean(axis=1)
        return np.array([_h(pi) for pi in p])

    def _conditional_entropy_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        n_a, n_b = A.shape[0], B.shape[0]
        H_A_given_B = np.zeros((n_a, n_b))
        for i in range(n_a):
            a = A[i]
            pa = a.mean()
            for j in range(n_b):
                b = B[j]
                pb = b.mean()
                p11 = np.mean(a * b)
                p10 = pa - p11
                p01 = pb - p11
                p00 = 1 - pa - pb + p11
                joint = [p00, p01, p10, p11]
                h_joint = -sum(p * np.log2(p) for p in joint if p > 0)
                h_b = _h(pb)
                H_A_given_B[i, j] = max(0.0, h_joint - h_b)
        return H_A_given_B

    H_P = _community_entropy(P)
    H_T = _community_entropy(T)

    H_P_given_T_matrix = _conditional_entropy_matrix(P, T)
    H_T_given_P_matrix = _conditional_entropy_matrix(T, P)

    H_P_given_T = H_P_given_T_matrix.min(axis=1)
    H_T_given_P = H_T_given_P_matrix.min(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        norm_P = np.where(H_P > 0, H_P_given_T / H_P, 0.0)
        norm_T = np.where(H_T > 0, H_T_given_P / H_T, 0.0)

    if len(norm_P) == 0 or len(norm_T) == 0:
        return 0.0

    return float(1.0 - 0.5 * (norm_P.mean() + norm_T.mean()))


def evaluate_disjoint(
    G: nx.Graph,
    pred: list[frozenset],
    true: list[frozenset],
) -> dict[str, float]:
    return {
        "NMI":        nmi(pred, true, G=G),
        "AMI":        ami(pred, true, G=G),
        "Modularity": modularity(G, pred),
        "F1":         pairwise_f1(pred, true),
    }


def evaluate_overlapping(
    G: nx.Graph,
    pred: list[frozenset],
    true: list[frozenset],
) -> dict[str, float]:
    pred_hard = _cover_to_hard_partition(pred, G=G)
    return {
        "NMI":        nmi(pred, true, G=G),
        "AMI":        ami(pred, true, G=G),
        "Modularity": modularity(G, pred_hard),
        "F1":         pairwise_f1(pred, true),
        "Omega":      omega_index(pred, true),
        "ONMI":       onmi(pred, true),
    }
