"""Structural pre-filter for component similarity.

This is the FIRST stage of clustering. It runs *before* any embedding lookup.
Two components become candidates for clustering iff:

  1. They share the same ``category_root`` (after normalization), and
  2. They share enough ``value_keys`` by Jaccard similarity.

The pre-filter is intentionally cheap and exact: O(N^2) over the component
list, pure-Python set ops, no model calls. It is meant to reject obvious
non-matches so the expensive embedding + clustering stages only see plausible
pairs.

Critical design rule: an empty ``category_root`` (``"unknown"``) NEVER pairs.
This is the safety valve for components whose ``category_code`` is missing —
they become singletons and are not clustered.
"""
from __future__ import annotations

from itertools import combinations
from typing import Iterable

from scripts.similarity.fingerprint import ComponentFingerprint


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _jaccard_threshold(value_keys_a: Iterable[str], value_keys_b: Iterable[str], threshold: float) -> bool:
    return _jaccard(set(value_keys_a), set(value_keys_b)) >= threshold


def structural_candidates(
    fps: list[ComponentFingerprint],
    jaccard_threshold: float = 0.5,
    require_nonempty_keys: bool = True,
) -> set[tuple[str, str]]:
    """Return unordered ``(cid_a, cid_b)`` pairs that pass the structural filter.

    Parameters
    ----------
    fps:
        The component fingerprints to filter.
    jaccard_threshold:
        Minimum Jaccard similarity on ``value_keys`` for a pair to be a
        candidate. Default 0.5 (half the keys must overlap).
    require_nonempty_keys:
        If True (default), reject pairs where either side has empty
        ``value_keys`` (we cannot prove they share geometry). Set False to
        allow those pairs — useful only in tests.

    Returns
    -------
    set of ``(cid_a, cid_b)`` with ``cid_a < cid_b`` (lexicographic).
    """
    if jaccard_threshold < 0.0 or jaccard_threshold > 1.0:
        raise ValueError(
            f"structural_candidates: jaccard_threshold must be in [0, 1], got {jaccard_threshold}"
        )

    out: set[tuple[str, str]] = set()
    for a, b in combinations(fps, 2):
        if a.category_root != b.category_root or a.category_root == "unknown":
            continue
        if require_nonempty_keys and (not a.value_keys or not b.value_keys):
            continue
        if not _jaccard_threshold(a.value_keys, b.value_keys, jaccard_threshold):
            continue
        # Order-insensitive: always store (min, max) so equality is well-defined.
        out.add(tuple(sorted((a.component_id, b.component_id))))
    return out


def same_root_components(
    fps: list[ComponentFingerprint],
) -> dict[str, list[str]]:
    """Group component_ids by category_root.

    Returns ``{category_root: [component_id, ...]}`` for every root with at
    least one member. Roots of ``"unknown"`` are excluded.

    Useful for diagnostics / reports — shows which categories are eligible
    for clustering at all.
    """
    groups: dict[str, list[str]] = {}
    for fp in fps:
        if fp.category_root == "unknown":
            continue
        groups.setdefault(fp.category_root, []).append(fp.component_id)
    return groups
