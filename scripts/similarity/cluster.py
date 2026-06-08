"""Hierarchical clustering + template-aware force-merge for component similarity.

This is the second stage of clustering. It takes the candidate pairs produced
by :mod:`scripts.similarity.filter` and:

  1. Computes a pairwise distance matrix from MiniLM embeddings (cosine
     distance = 1 - cos similarity).
  2. Sets non-candidate distances to 1.0 so the agglomerative clustering
     can never bridge across the structural pre-filter.
  3. Tries multiple distance thresholds and picks the one with the best
     silhouette score (only when 1 < n_clusters < N).
  4. Builds a :class:`PartFamily` per cluster.
  5. Applies a *template-aware force-merge*: any two clusters whose members
     share a template_signature with Jaccard >= ``template_jaccard_threshold``
     are union-merged, regardless of their embedding distance.

The template-aware step is the bridge into the per-component feature templates
produced by Step 1. If two components have already produced (or would produce)
near-identical templates, they MUST be in the same family — embedding distance
is not a reliable signal for that case.

Notes
-----
We use sklearn's AgglomerativeClustering with a precomputed distance matrix
and ``average`` linkage. ``complete`` linkage would over-merge outliers;
``single`` linkage would chain. ``average`` is the right default for our
small N (typically 5-50 components per category root).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from scripts.similarity.fingerprint import ComponentFingerprint
from scripts.similarity.store import _get_model


# A list of distance thresholds to try. We pick the one that gives the best
# silhouette score. 0.10 is very tight (almost identical components), 0.30
# is loose (semantically related components).
DEFAULT_DISTANCE_THRESHOLDS: tuple[float, ...] = (0.10, 0.15, 0.20, 0.25, 0.30)

# Force-merge trigger: any two clusters whose members have a template_signature
# with at least this much Jaccard overlap (by parameter names) get merged.
DEFAULT_TEMPLATE_JACCARD: float = 0.8


@dataclass
class PartFamily:
    family_id: str
    category_root: str
    member_component_ids: list[str]
    shared_value_keys: list[str]
    shared_attribute_keys: list[str]
    canonical_template_id: Optional[str]
    avg_template_signature_jaccard: float = 0.0
    created_at: Optional[str] = None  # filled by orchestrator

    def to_dict(self) -> dict:
        return {
            "family_id": self.family_id,
            "category_root": self.category_root,
            "member_component_ids": list(self.member_component_ids),
            "shared_value_keys": list(self.shared_value_keys),
            "shared_attribute_keys": list(self.shared_attribute_keys),
            "canonical_template_id": self.canonical_template_id,
            "avg_template_signature_jaccard": self.avg_template_signature_jaccard,
            "created_at": self.created_at,
        }


def _set_signature_jaccard(sig_a: Optional[str], sig_b: Optional[str]) -> float:
    """Jaccard of two template signatures. A signature is 'mode|p1|p2|...'.

    Returns 0.0 if either side is None (no signature = no evidence of similarity).
    """
    if not sig_a or not sig_b:
        return 0.0
    sa = set(sig_a.split("|"))
    sb = set(sig_b.split("|"))
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def _set_intersection_sorted(lists: list[list[str]]) -> list[str]:
    """Sorted intersection of a list of string-lists.

    An element is in the intersection iff it appears in *every* input list.
    """
    if not lists:
        return []
    sets = [set(lst) for lst in lists]
    common = sets[0]
    for s in sets[1:]:
        common &= s
    return sorted(common)


def _best_template_jaccard_across_clusters(
    cluster_a_members: list[ComponentFingerprint],
    cluster_b_members: list[ComponentFingerprint],
) -> float:
    """Maximum pairwise signature-Jaccard across two clusters."""
    best = 0.0
    for ma in cluster_a_members:
        if not ma.template_signature:
            continue
        for mb in cluster_b_members:
            if not mb.template_signature:
                continue
            j = _set_signature_jaccard(ma.template_signature, mb.template_signature)
            if j > best:
                best = j
    return best


def _force_merge(
    families: list[PartFamily],
    fps_by_id: dict[str, ComponentFingerprint],
    template_jaccard_threshold: float = DEFAULT_TEMPLATE_JACCARD,
) -> list[PartFamily]:
    """Union-merge any two families whose members share a near-identical
    template signature.

    Uses union-find. After the loop, collapses each equivalence class back
    into a single family. Canonical template is recomputed as the member
    with the highest variant_count that has a signature.
    """
    if not families:
        return families

    parent: dict[int, int] = {i: i for i in range(len(families))}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, j in combinations(range(len(families)), 2):
        members_i = [fps_by_id[m] for m in families[i].member_component_ids if m in fps_by_id]
        members_j = [fps_by_id[m] for m in families[j].member_component_ids if m in fps_by_id]
        if not members_i or not members_j:
            continue
        best = _best_template_jaccard_across_clusters(members_i, members_j)
        if best >= template_jaccard_threshold:
            union(i, j)

    # Group families by their root.
    groups: dict[int, list[int]] = {}
    for k in parent:
        groups.setdefault(find(k), []).append(k)

    merged: list[PartFamily] = []
    for group_indices in groups.values():
        members: set[str] = set()
        cat_root = families[group_indices[0]].category_root
        for idx in group_indices:
            members.update(families[idx].member_component_ids)
        members_sorted = sorted(members)
        fps_in_family = [fps_by_id[m] for m in members_sorted if m in fps_by_id]
        shared_value = _set_intersection_sorted([fp.value_keys for fp in fps_in_family])
        shared_attr = _set_intersection_sorted([fp.attribute_keys for fp in fps_in_family])
        with_sig = [fp for fp in fps_in_family if fp.template_signature]
        canon = max(with_sig, key=lambda fp: (fp.variant_count, fp.component_id), default=None)

        # Average pairwise signature Jaccard within the family (for diagnostics).
        avg_j = 0.0
        if len(with_sig) >= 2:
            jaccards = []
            for a, b in combinations(with_sig, 2):
                jaccards.append(_set_signature_jaccard(a.template_signature, b.template_signature))
            avg_j = sum(jaccards) / len(jaccards) if jaccards else 0.0

        fid = families[group_indices[0]].family_id
        merged.append(
            PartFamily(
                family_id=fid,
                category_root=cat_root,
                member_component_ids=members_sorted,
                shared_value_keys=shared_value,
                shared_attribute_keys=shared_attr,
                canonical_template_id=canon.component_id if canon else None,
                avg_template_signature_jaccard=avg_j,
            )
        )
    return merged


def _silhouette_pick(
    distance_matrix: np.ndarray,
    candidate_thresholds: tuple[float, ...] = DEFAULT_DISTANCE_THRESHOLDS,
) -> tuple[Optional[np.ndarray], float]:
    """Try several distance thresholds; return the labels that scored best.

    Returns ``(labels, best_threshold)``. ``labels`` is None if no threshold
    produced 1 < n_clusters < N (in which case the caller should default to
    a sensible fallback).
    """
    n = distance_matrix.shape[0]
    if n < 3:
        return None, 0.0
    best_score = -2.0
    best_labels: Optional[np.ndarray] = None
    best_thr = 0.0
    for thr in candidate_thresholds:
        labels = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=thr,
            metric="precomputed",
            linkage="average",
        ).fit_predict(distance_matrix)
        n_clusters = len(set(labels))
        if not (1 < n_clusters < n):
            continue
        # silhouette_score expects a 1-D label vector and a distance matrix.
        score = silhouette_score(distance_matrix, labels, metric="precomputed")
        if score > best_score:
            best_score = score
            best_labels = labels
            best_thr = thr
    return best_labels, best_thr


def cluster_fingerprints(
    fps: list[ComponentFingerprint],
    candidate_pairs: set[tuple[str, str]],
    template_jaccard_threshold: float = DEFAULT_TEMPLATE_JACCARD,
    distance_thresholds: tuple[float, ...] = DEFAULT_DISTANCE_THRESHOLDS,
) -> list[PartFamily]:
    """Cluster fingerprints into part families.

    Parameters
    ----------
    fps:
        All component fingerprints (including ones that didn't pass the
        structural pre-filter — those will become singletons).
    candidate_pairs:
        Pairs that passed the structural pre-filter. Non-candidate pairs
        will never cluster.
    template_jaccard_threshold:
        Force-merge trigger based on template_signature Jaccard.
    distance_thresholds:
        Distance thresholds to try in the silhouette search.

    Returns
    -------
    list[PartFamily] — one per family (singletons included). Each family has
    a unique ``family_id`` of the form ``"fam_{category_root}_{n:03d}"``.
    """
    if not fps:
        return []

    fps_by_id = {fp.component_id: fp for fp in fps}

    # Singleton-only case: no clustering needed.
    if len(fps) == 1:
        fp = fps[0]
        return [
            PartFamily(
                family_id=f"fam_{fp.category_root or 'unknown'}_000",
                category_root=fp.category_root,
                member_component_ids=[fp.component_id],
                shared_value_keys=list(fp.value_keys),
                shared_attribute_keys=list(fp.attribute_keys),
                canonical_template_id=fp.component_id if fp.template_signature else None,
            )
        ]

    # If no candidate pairs were provided, every component is a singleton.
    if not candidate_pairs:
        return [
            PartFamily(
                family_id=f"fam_{fp.category_root or 'unknown'}_{i:03d}",
                category_root=fp.category_root,
                member_component_ids=[fp.component_id],
                shared_value_keys=list(fp.value_keys),
                shared_attribute_keys=list(fp.attribute_keys),
                canonical_template_id=fp.component_id if fp.template_signature else None,
            )
            for i, fp in enumerate(fps)
        ]

    # Compute embeddings and distance matrix.
    model = _get_model()
    texts = [fp.text_for_embedding for fp in fps]
    vecs = np.array(model.encode(texts), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    nvecs = vecs / np.clip(norms, 1e-9, None)
    dist = 1.0 - (nvecs @ nvecs.T)
    np.fill_diagonal(dist, 0.0)

    # Zero out non-candidate distances so the clusterer can never bridge
    # across the structural pre-filter.
    cid_idx = {fp.component_id: i for i, fp in enumerate(fps)}
    for i in range(len(fps)):
        for j in range(len(fps)):
            if i == j:
                continue
            pair = tuple(sorted((fps[i].component_id, fps[j].component_id)))
            if pair not in candidate_pairs:
                dist[i, j] = 1.0

    # Pick the best threshold by silhouette. If silhouette is degenerate (N<3
    # or all candidate pairs at one distance) we still want to make a per-pair
    # cluster from each candidate pair so the force-merge step can run.
    labels, _thr = _silhouette_pick(dist, candidate_thresholds=distance_thresholds)
    if labels is None:
        # Fall back: every candidate pair forms its own 2-component cluster;
        # non-candidate components become singletons. Use union-find so that
        # chains of candidate pairs (A-B, B-C) collapse to one cluster.
        parent = {i: i for i in range(len(fps))}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for pair in candidate_pairs:
            if pair[0] in cid_idx and pair[1] in cid_idx:
                union(cid_idx[pair[0]], cid_idx[pair[1]])

        groups: dict[int, list[int]] = {}
        for i in range(len(fps)):
            groups.setdefault(find(i), []).append(i)
        labels = np.array([0] * len(fps), dtype=np.int64)
        for new_label, members in enumerate(groups.values()):
            for m in members:
                labels[m] = new_label

    # Build per-cluster member lists, then PartFamily objects.
    clusters: dict[int, list[ComponentFingerprint]] = {}
    for fp, lab in zip(fps, labels):
        clusters.setdefault(int(lab), []).append(fp)

    # Group clusters by category_root so family_ids are stable and unique.
    by_root: dict[str, list[list[ComponentFingerprint]]] = {}
    for members in clusters.values():
        # Use the first member's root as the cluster's root. (All members of a
        # cluster share the same root because the pre-filter enforces it.)
        root = members[0].category_root or "unknown"
        by_root.setdefault(root, []).append(members)

    families: list[PartFamily] = []
    family_seq = 0
    for root in sorted(by_root.keys()):
        for members in sorted(by_root[root], key=lambda ms: -len(ms)):
            member_ids = [m.component_id for m in members]
            shared_value = _set_intersection_sorted([m.value_keys for m in members])
            shared_attr = _set_intersection_sorted([m.attribute_keys for m in members])
            with_sig = [m for m in members if m.template_signature]
            canon = max(with_sig, key=lambda m: (m.variant_count, m.component_id), default=None)
            avg_j = 0.0
            if len(with_sig) >= 2:
                jaccards = [
                    _set_signature_jaccard(a.template_signature, b.template_signature)
                    for a, b in combinations(with_sig, 2)
                ]
                avg_j = sum(jaccards) / len(jaccards) if jaccards else 0.0
            families.append(
                PartFamily(
                    family_id=f"fam_{root}_{family_seq:03d}",
                    category_root=root,
                    member_component_ids=member_ids,
                    shared_value_keys=shared_value,
                    shared_attribute_keys=shared_attr,
                    canonical_template_id=canon.component_id if canon else None,
                    avg_template_signature_jaccard=avg_j,
                )
            )
            family_seq += 1

    # Template-aware force-merge.
    families = _force_merge(families, fps_by_id, template_jaccard_threshold=template_jaccard_threshold)
    return families
