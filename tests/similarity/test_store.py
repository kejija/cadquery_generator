"""Tests for scripts.similarity.store.

The first call into FingerprintStore triggers a one-time download of
``all-MiniLM-L6-v2`` (~80MB) from HuggingFace. The tests in this file use
``all-MiniLM-L6-v1`` (also ~80MB but already cached if MiniLM is present) or
the default. To skip model download entirely, set ``HERMES_SKIP_EMBEDDINGS=1``.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from scripts.similarity.fingerprint import ComponentFingerprint
from scripts.similarity.store import FingerprintStore, _get_model


# --- fixtures ------------------------------------------------------------------

def _make_fp(cid: str, **overrides) -> ComponentFingerprint:
    defaults = {
        "component_id": cid,
        "name": f"Component {cid}",
        "description": "test description",
        "category_code": "ball_bearing",
        "category_root": "bearing",
        "attribute_keys": ["brand"],
        "value_keys": ["OD", "ID", "B"],
        "variant_count": 4,
        "template_signature": "simplified_single_body|OD|ID|B",
        "text_for_embedding": f"Component {cid} | test description | category:ball_bearing | params:OD ID B",
    }
    defaults.update(overrides)
    return ComponentFingerprint(**defaults)


# Skip the whole module if the user opts out of downloading models.
_skip = os.environ.get("HERMES_SKIP_EMBEDDINGS") == "1"
pytestmark = pytest.mark.skipif(
    _skip, reason="HERMES_SKIP_EMBEDDINGS=1; embeddings disabled for this run"
)


# --- persistence ---------------------------------------------------------------

def test_upsert_and_get_round_trip(tmp_path: Path):
    store = FingerprintStore(persist_dir=tmp_path, collection_name="test_round_trip")
    fp = _make_fp("A")
    store.upsert(fp)
    out = store.get("A")
    assert out is not None
    assert out.component_id == "A"
    assert out.category_root == "bearing"
    assert out.value_keys == ["OD", "ID", "B"]  # round-trip preserves insertion order
    assert out.variant_count == 4
    assert out.template_signature is not None


def test_persistence_across_instances(tmp_path: Path):
    store_a = FingerprintStore(persist_dir=tmp_path, collection_name="coll_persist")
    fp = _make_fp("X", name="Persistent Component")
    store_a.upsert(fp)
    del store_a

    # New instance over the same dir should see the same record.
    store_b = FingerprintStore(persist_dir=tmp_path, collection_name="coll_persist")
    out = store_b.get("X")
    assert out is not None
    assert out.name == "Persistent Component"


def test_get_returns_none_for_missing_id(tmp_path: Path):
    store = FingerprintStore(persist_dir=tmp_path, collection_name="coll_missing")
    assert store.get("GHOST") is None


def test_upsert_overwrites_existing(tmp_path: Path):
    store = FingerprintStore(persist_dir=tmp_path, collection_name="coll_overwrite")
    fp1 = _make_fp("A", variant_count=1)
    fp2 = _make_fp("A", variant_count=99)
    store.upsert(fp1)
    store.upsert(fp2)
    out = store.get("A")
    assert out is not None
    assert out.variant_count == 99
    assert store.count() == 1


def test_upsert_many_batched(tmp_path: Path):
    store = FingerprintStore(persist_dir=tmp_path, collection_name="coll_batch")
    fps = [_make_fp(f"X{i}") for i in range(8)]
    store.upsert_many(fps)
    assert store.count() == 8
    for fp in fps:
        assert store.get(fp.component_id) is not None


def test_all_fingerprints_returns_everything(tmp_path: Path):
    store = FingerprintStore(persist_dir=tmp_path, collection_name="coll_all")
    for cid in ("a", "b", "c"):
        store.upsert(_make_fp(cid))
    out = store.all_fingerprints()
    assert {fp.component_id for fp in out} == {"a", "b", "c"}


# --- query ---------------------------------------------------------------------

def test_query_similar_returns_neighbors_in_distance_order(tmp_path: Path):
    store = FingerprintStore(persist_dir=tmp_path, collection_name="coll_query")
    # Two bearings + one shaft. Bearings should be close to each other and
    # far from the shaft.
    store.upsert(_make_fp("bearing1", text_for_embedding="Deep groove ball bearing OD ID B"))
    store.upsert(_make_fp("bearing2", text_for_embedding="Angular contact ball bearing OD ID B"))
    store.upsert(_make_fp("shaft1", text_for_embedding="Linear shaft L D keyway"))

    q = _make_fp("q", text_for_embedding="Deep groove ball bearing OD ID B")
    results = store.query_similar(q, n=3)
    assert len(results) == 3
    # The first hit should be one of the bearings.
    assert "bearing" in results[0][0].component_id
    # Distances should be non-decreasing.
    dists = [d for _, d in results]
    assert dists == sorted(dists)
    # Bearings should be closer to the query than the shaft.
    bearing_dists = [d for fp, d in results if "bearing" in fp.component_id]
    shaft_dist = [d for fp, d in results if "shaft" in fp.component_id][0]
    assert max(bearing_dists) < shaft_dist


def test_query_similar_with_n_smaller_than_corpus(tmp_path: Path):
    store = FingerprintStore(persist_dir=tmp_path, collection_name="coll_query_n")
    for i in range(5):
        store.upsert(_make_fp(f"x{i}"))
    results = store.query_similar(_make_fp("q"), n=2)
    assert len(results) == 2


# --- reset ---------------------------------------------------------------------

def test_reset_clears_collection(tmp_path: Path):
    store = FingerprintStore(persist_dir=tmp_path, collection_name="coll_reset")
    store.upsert(_make_fp("A"))
    assert store.count() == 1
    store.reset()
    assert store.count() == 0
    assert store.get("A") is None


# --- model caching -------------------------------------------------------------

def test_get_model_returns_same_instance():
    a = _get_model()
    b = _get_model()
    assert a is b
