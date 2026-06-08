"""Tests for scripts.similarity.filter."""
from __future__ import annotations

import pytest

from scripts.similarity.filter import _jaccard, same_root_components, structural_candidates
from scripts.similarity.fingerprint import ComponentFingerprint


def _make_fp(cid: str, category_root: str, value_keys: list[str]) -> ComponentFingerprint:
    return ComponentFingerprint(
        component_id=cid,
        name=cid,
        description="",
        category_code=category_root,
        category_root=category_root,
        attribute_keys=[],
        value_keys=value_keys,
        variant_count=0,
        template_signature=None,
        text_for_embedding="",
    )


# --- _jaccard ------------------------------------------------------------------

def test_jaccard_identical_sets_is_one():
    assert _jaccard({"a", "b"}, {"a", "b"}) == 1.0


def test_jaccard_disjoint_sets_is_zero():
    assert _jaccard({"a"}, {"b"}) == 0.0


def test_jaccard_both_empty_is_one():
    assert _jaccard(set(), set()) == 1.0


def test_jaccard_one_empty_is_zero():
    assert _jaccard(set(), {"a"}) == 0.0
    assert _jaccard({"a"}, set()) == 0.0


def test_jaccard_partial_overlap():
    # {a, b} ∩ {b, c} = {b}, |union| = 3
    assert _jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1.0 / 3.0)


# --- structural_candidates -----------------------------------------------------

def test_different_category_roots_rejected():
    a = _make_fp("A", "bearing", ["OD", "ID", "B"])
    b = _make_fp("B", "shaft", ["OD", "ID", "B"])
    assert structural_candidates([a, b]) == set()


def test_same_root_with_shared_keys_accepted_at_threshold_0_5():
    a = _make_fp("A", "bearing", ["OD", "ID", "B"])
    b = _make_fp("B", "bearing", ["OD", "ID", "B", "C"])
    # Jaccard = 3/4 = 0.75 ≥ 0.5
    assert structural_candidates([a, b]) == {("A", "B")}


def test_same_root_with_high_overlap_accepted():
    a = _make_fp("A", "bearing", ["OD", "ID", "B"])
    b = _make_fp("B", "bearing", ["OD", "ID", "B", "C", "D"])
    assert structural_candidates([a, b], jaccard_threshold=0.5) == {("A", "B")}


def test_same_root_with_low_overlap_rejected():
    a = _make_fp("A", "bearing", ["OD", "ID", "B"])
    b = _make_fp("B", "bearing", ["X", "Y", "Z"])
    assert structural_candidates([a, b], jaccard_threshold=0.5) == set()


def test_unknown_category_excluded():
    a = _make_fp("A", "unknown", ["OD", "ID", "B"])
    b = _make_fp("B", "unknown", ["OD", "ID", "B"])
    # Even though value_keys match, "unknown" is excluded by design.
    assert structural_candidates([a, b]) == set()


def test_jaccard_threshold_respected():
    a = _make_fp("A", "bearing", ["OD", "ID", "B", "C"])  # 4 keys
    b = _make_fp("B", "bearing", ["OD", "ID", "X", "Y"])  # 2 overlap / 6 union = 0.333
    # At threshold 0.3, accepted; at 0.4, rejected.
    assert structural_candidates([a, b], jaccard_threshold=0.3) == {("A", "B")}
    assert structural_candidates([a, b], jaccard_threshold=0.4) == set()


def test_pair_is_unordered():
    a = _make_fp("Z", "bearing", ["OD", "ID", "B"])
    b = _make_fp("A", "bearing", ["OD", "ID", "B"])
    result = structural_candidates([a, b])
    # Always sorted (min, max) — so (A, Z) regardless of input order.
    assert result == {("A", "Z")}


def test_empty_value_keys_excluded_by_default():
    a = _make_fp("A", "bearing", ["OD", "ID", "B"])
    b = _make_fp("B", "bearing", [])
    # require_nonempty_keys=True is the default
    assert structural_candidates([a, b]) == set()


def test_empty_value_keys_allowed_when_opt_in():
    a = _make_fp("A", "bearing", ["OD", "ID", "B"])
    b = _make_fp("B", "bearing", [])
    # With require_nonempty_keys=False, jaccard=0/3=0 < 0.5 still rejects.
    assert structural_candidates([a, b], require_nonempty_keys=False) == set()
    # ... but at jaccard_threshold=0 it would be accepted.
    assert structural_candidates([a, b], jaccard_threshold=0.0, require_nonempty_keys=False) == {("A", "B")}


def test_invalid_threshold_raises():
    a = _make_fp("A", "bearing", ["OD", "ID"])
    b = _make_fp("B", "bearing", ["OD", "ID"])
    with pytest.raises(ValueError):
        structural_candidates([a, b], jaccard_threshold=-0.1)
    with pytest.raises(ValueError):
        structural_candidates([a, b], jaccard_threshold=1.1)


def test_three_components_pairwise():
    a = _make_fp("A", "bearing", ["OD", "ID", "B"])
    b = _make_fp("B", "bearing", ["OD", "ID", "B"])
    c = _make_fp("C", "shaft",   ["OD", "ID", "B"])
    # Bearings pair, but C is in a different root and not paired with A or B.
    assert structural_candidates([a, b, c]) == {("A", "B")}


def test_empty_input():
    assert structural_candidates([]) == set()


def test_single_component():
    a = _make_fp("A", "bearing", ["OD", "ID", "B"])
    assert structural_candidates([a]) == set()


# --- same_root_components ------------------------------------------------------

def test_same_root_groups_by_category():
    a = _make_fp("A", "bearing", ["OD"])
    b = _make_fp("B", "bearing", ["OD"])
    c = _make_fp("C", "shaft", ["OD"])
    d = _make_fp("D", "unknown", ["OD"])
    out = same_root_components([a, b, c, d])
    assert out == {"bearing": ["A", "B"], "shaft": ["C"]}
