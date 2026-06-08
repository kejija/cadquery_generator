"""Tests for scripts.similarity.cluster."""
from __future__ import annotations

import os

import pytest

from scripts.similarity.cluster import (
    DEFAULT_TEMPLATE_JACCARD,
    PartFamily,
    _best_template_jaccard_across_clusters,
    _force_merge,
    _set_intersection_sorted,
    _set_signature_jaccard,
    _silhouette_pick,
    cluster_fingerprints,
)
from scripts.similarity.fingerprint import ComponentFingerprint

# Skip the embedding-dependent tests in environments without HF access.
_skip = os.environ.get("HERMES_SKIP_EMBEDDINGS") == "1"


def _make_fp(
    cid: str,
    category_root: str = "bearing",
    value_keys: list[str] | None = None,
    template_signature: str | None = None,
    variant_count: int = 0,
    text: str = "",
) -> ComponentFingerprint:
    return ComponentFingerprint(
        component_id=cid,
        name=cid,
        description="",
        category_code=category_root,
        category_root=category_root,
        attribute_keys=["brand"],
        value_keys=value_keys if value_keys is not None else ["OD", "ID", "B"],
        variant_count=variant_count,
        template_signature=template_signature,
        text_for_embedding=text or f"{cid} bearing {category_root}",
    )


# --- _set_signature_jaccard -----------------------------------------------------

def test_signature_jaccard_identical():
    assert _set_signature_jaccard("a|b|c", "a|b|c") == 1.0


def test_signature_jaccard_disjoint():
    assert _set_signature_jaccard("a|b", "c|d") == 0.0


def test_signature_jaccard_partial():
    # {a, b, c} ∩ {b, c, d} = 2, union = 4
    assert _set_signature_jaccard("a|b|c", "b|c|d") == 0.5


def test_signature_jaccard_returns_zero_for_none():
    assert _set_signature_jaccard(None, "a|b") == 0.0
    assert _set_signature_jaccard("a|b", None) == 0.0
    assert _set_signature_jaccard(None, None) == 0.0


# --- _set_intersection_sorted ---------------------------------------------------

def test_set_intersection_sorted_empty():
    assert _set_intersection_sorted([]) == []


def test_set_intersection_sorted_single_list():
    assert _set_intersection_sorted([["c", "a", "b"]]) == ["a", "b", "c"]


def test_set_intersection_sorted_common_across_all():
    common = _set_intersection_sorted([["a", "b", "c"], ["a", "b", "x"], ["a", "y", "b"]])
    assert common == ["a", "b"]


def test_set_intersection_sorted_disjoint():
    assert _set_intersection_sorted([["a"], ["b"]]) == []


# --- _best_template_jaccard_across_clusters -------------------------------------

def test_best_template_jaccard_picks_highest_pair():
    a = _make_fp("A", template_signature="m|p1|p2|p3")
    b = _make_fp("B", template_signature="m|p1|p2|x")
    c = _make_fp("C", template_signature="m|p1|p2|p3")
    out = _best_template_jaccard_across_clusters([a, b], [c])
    # (A, C) = {m, p1, p2, p3} ∩ {m, p1, p2, p3} / union = 1.0
    assert out == 1.0


def test_best_template_jaccard_with_missing_signatures():
    a = _make_fp("A", template_signature="m|p1")
    b = _make_fp("B", template_signature=None)
    c = _make_fp("C", template_signature="m|p2")
    out = _best_template_jaccard_across_clusters([a, b], [c])
    # B has no signature, so the only (a, c) pair is considered: 1/3.
    assert out == pytest.approx(1.0 / 3.0)


# --- _force_merge ---------------------------------------------------------------

def test_force_merge_merges_high_template_jaccard_families():
    a = _make_fp("A", template_signature="m|p1|p2|p3")
    b = _make_fp("B", template_signature="m|p1|p2|p3")
    fps_by_id = {"A": a, "B": b}
    fams = [
        PartFamily(
            family_id="fam1", category_root="bearing",
            member_component_ids=["A"], shared_value_keys=[],
            shared_attribute_keys=[], canonical_template_id="A",
        ),
        PartFamily(
            family_id="fam2", category_root="bearing",
            member_component_ids=["B"], shared_value_keys=[],
            shared_attribute_keys=[], canonical_template_id="B",
        ),
    ]
    merged = _force_merge(fams, fps_by_id, template_jaccard_threshold=0.8)
    assert len(merged) == 1
    assert set(merged[0].member_component_ids) == {"A", "B"}


def test_force_merge_does_not_merge_low_template_jaccard():
    a = _make_fp("A", template_signature="m|p1|p2")
    b = _make_fp("B", template_signature="m|p3|p4")
    fps_by_id = {"A": a, "B": b}
    fams = [
        PartFamily("f1", "bearing", ["A"], [], [], "A"),
        PartFamily("f2", "bearing", ["B"], [], [], "B"),
    ]
    merged = _force_merge(fams, fps_by_id, template_jaccard_threshold=0.8)
    assert len(merged) == 2


def test_force_merge_picks_highest_variant_count_as_canonical():
    a = _make_fp("A", template_signature="m|p1|p2", variant_count=2)
    b = _make_fp("B", template_signature="m|p1|p2", variant_count=99)
    fps_by_id = {"A": a, "B": b}
    fams = [
        PartFamily("f1", "bearing", ["A"], [], [], "A"),
        PartFamily("f2", "bearing", ["B"], [], [], "B"),
    ]
    merged = _force_merge(fams, fps_by_id, template_jaccard_threshold=0.8)
    assert len(merged) == 1
    assert merged[0].canonical_template_id == "B"


def test_force_merge_canonical_falls_back_to_first_when_no_signatures():
    a = _make_fp("A", template_signature=None)
    b = _make_fp("B", template_signature=None)
    fps_by_id = {"A": a, "B": b}
    fams = [
        PartFamily("f1", "bearing", ["A"], [], [], None),
        PartFamily("f2", "bearing", ["B"], [], [], None),
    ]
    merged = _force_merge(fams, fps_by_id, template_jaccard_threshold=0.8)
    # No signatures → no force-merge; both stay separate.
    assert len(merged) == 2


# --- _silhouette_pick -----------------------------------------------------------

def test_silhouette_pick_returns_none_for_too_few_samples():
    import numpy as np
    # 2 samples can't silhouette.
    dist = np.array([[0.0, 0.1], [0.1, 0.0]])
    labels, _ = _silhouette_pick(dist)
    assert labels is None


def test_silhouette_pick_returns_labels_for_three_clear_clusters():
    import numpy as np
    # Three obvious clusters with tight intra-distance and large inter-distance.
    dist = np.array([
        [0.0, 0.05, 0.5, 0.5, 0.5, 0.5],
        [0.05, 0.0, 0.5, 0.5, 0.5, 0.5],
        [0.5, 0.5, 0.0, 0.05, 0.5, 0.5],
        [0.5, 0.5, 0.05, 0.0, 0.5, 0.5],
        [0.5, 0.5, 0.5, 0.5, 0.0, 0.05],
        [0.5, 0.5, 0.5, 0.5, 0.05, 0.0],
    ])
    labels, thr = _silhouette_pick(dist, candidate_thresholds=(0.10, 0.15, 0.20, 0.25))
    assert labels is not None
    # Should produce 3 clusters, each with 2 members.
    n_clusters = len(set(labels.tolist()))
    assert n_clusters == 3
    assert thr > 0.0


# --- cluster_fingerprints --------------------------------------------------------

def test_cluster_fingerprints_empty_input():
    assert cluster_fingerprints([], set()) == []


def test_cluster_fingerprints_single_input_is_singleton():
    fp = _make_fp("A")
    out = cluster_fingerprints([fp], set())
    assert len(out) == 1
    assert out[0].member_component_ids == ["A"]
    assert out[0].family_id.startswith("fam_bearing_")


def test_cluster_fingerprints_no_candidate_pairs_yields_singletons():
    fps = [_make_fp(f"X{i}") for i in range(4)]
    out = cluster_fingerprints(fps, candidate_pairs=set())
    assert len(out) == 4
    for fam in out:
        assert len(fam.member_component_ids) == 1


@pytest.mark.skipif(_skip, reason="HERMES_SKIP_EMBEDDINGS=1; requires MiniLM")
def test_cluster_fingerprints_groups_bearings_and_singletons_shaft():
    # Two bearings + one shaft. Bearings are in the same root, share value_keys,
    # and have distinct-enough text that the embedding should cluster them.
    fps = [
        _make_fp("brg1", category_root="bearing", value_keys=["OD", "ID", "B"],
                 text="Deep groove ball bearing 6204 OD ID B"),
        _make_fp("brg2", category_root="bearing", value_keys=["OD", "ID", "B"],
                 text="Angular contact ball bearing OD ID B"),
        _make_fp("shaft1", category_root="shaft", value_keys=["OD", "L", "keyway"],
                 text="Linear shaft L D keyway"),
    ]
    candidate_pairs = {("brg1", "brg2")}  # only the bearing pair survives the filter
    out = cluster_fingerprints(fps, candidate_pairs=candidate_pairs)
    # At minimum: brg1 and brg2 are in the same family.
    bearings_family = [f for f in out if "brg1" in f.member_component_ids]
    assert len(bearings_family) == 1
    assert set(bearings_family[0].member_component_ids) == {"brg1", "brg2"}
    # The shaft is in its own family (different root).
    shaft_family = [f for f in out if "shaft1" in f.member_component_ids]
    assert len(shaft_family) == 1
    assert shaft_family[0].member_component_ids == ["shaft1"]


@pytest.mark.skipif(_skip, reason="HERMES_SKIP_EMBEDDINGS=1; requires MiniLM")
def test_cluster_fingerprints_force_merges_matching_template_signatures():
    # Two components in different positions in the embedding space but with
    # identical template signatures should be force-merged.
    fps = [
        _make_fp("A", category_root="bearing", value_keys=["OD", "ID", "B"],
                 template_signature="m|p1|p2|p3|p4",
                 text="Component A in some weird domain"),
        _make_fp("B", category_root="bearing", value_keys=["OD", "ID", "B"],
                 template_signature="m|p1|p2|p3|p4",
                 text="Component B in a different weird domain"),
    ]
    # Even if structural filter wouldn't pick them up (we feed the pair
    # explicitly), force-merge should kick in via template signature.
    candidate_pairs = {("A", "B")}
    out = cluster_fingerprints(fps, candidate_pairs=candidate_pairs)
    assert len(out) == 1
    assert set(out[0].member_component_ids) == {"A", "B"}


def test_cluster_fingerprints_returns_valid_partfamily_objects():
    fps = [
        _make_fp("A", category_root="bearing", value_keys=["OD", "ID", "B"]),
        _make_fp("B", category_root="bearing", value_keys=["OD", "ID", "B"]),
    ]
    out = cluster_fingerprints(fps, candidate_pairs={("A", "B")})
    for fam in out:
        assert isinstance(fam, PartFamily)
        assert fam.family_id.startswith("fam_")
        assert isinstance(fam.member_component_ids, list)
        assert isinstance(fam.shared_value_keys, list)
        assert 0.0 <= fam.avg_template_signature_jaccard <= 1.0
