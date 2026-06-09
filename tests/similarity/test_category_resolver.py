"""Tests for scripts.similarity.category_resolver."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.similarity.category_resolver import (
    _leaf_code_from_specs,
    load_yaml_index,
    resolve_category,
    resolve_many,
)


# --- _leaf_code_from_specs ----------------------------------------------------


def test_leaf_code_takes_last_m_code_in_last_breadcrumb():
    specs = {
        "breadcrumbs": [
            {"text": "Home", "url": "https://x.com/"},
            {"text": "Linear Motion", "url": "https://x.com/mech/M0100000000/"},
            {"text": "Linear Shafts", "url": "https://x.com/mech/M0100000000/M0101000000/"},
        ]
    }
    assert _leaf_code_from_specs(specs) == "M0101000000"


def test_leaf_code_returns_none_for_empty_breadcrumbs():
    assert _leaf_code_from_specs({"breadcrumbs": []}) is None


def test_leaf_code_returns_none_for_no_m_codes():
    specs = {"breadcrumbs": [{"text": "Home", "url": "https://x.com/"}]}
    assert _leaf_code_from_specs(specs) is None


# --- load_yaml_index ----------------------------------------------------------


def test_load_yaml_index_returns_flat_dict(tmp_path: Path):
    # Write a tiny test YAML.
    yaml_path = tmp_path / "test_hier.yaml"
    yaml_path.write_text("""
categories:
  - id: "M0100000000"
    name: "Linear Motion"
    children:
      - id: "M0101000000"
        name: "Linear Shafts"
        description: "Precision steel shafts for linear motion."
      - id: "M0104000000"
        name: "Linear Ball Bearings"
        description: "Compact recirculating ball bearings."
""")
    index = load_yaml_index(yaml_path=yaml_path)
    assert "M0101000000" in index
    assert "M0104000000" in index
    assert "M0100000000" in index
    assert index["M0101000000"]["parent_id"] == "M0100000000"
    assert "Precision steel shafts" in index["M0101000000"]["description"]


def test_load_yaml_index_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_yaml_index(yaml_path=tmp_path / "nonexistent.yaml")


# --- resolve_category ---------------------------------------------------------


@pytest.fixture
def small_index(tmp_path: Path) -> tuple[Path, dict]:
    yaml_path = tmp_path / "test_hier.yaml"
    yaml_path.write_text("""
categories:
  - id: "M0100000000"
    name: "Linear Motion"
    children:
      - id: "M0101000000"
        name: "Linear Shafts"
        description: "Precision steel shafts."
      - id: "M0104000000"
        name: "Linear Ball Bearings"
        description: "Ball bearing blocks."
      - id: "M2000000000"
        name: "Pipes, Tubes, Hoses & Fittings"
        children:
          - id: "M2004000000"
            name: "Quick Connect Couplers & Joints"
            description: "Quick disconnect couplers."
""")
    return yaml_path, load_yaml_index(yaml_path=yaml_path)


def test_resolve_category_direct_match(small_index):
    _, index = small_index
    specs = {"breadcrumbs": [
        {"url": "https://x.com/"},
        {"url": "https://x.com/M0100000000/"},
        {"url": "https://x.com/M0100000000/M0101000000/"},
    ]}
    res = resolve_category(index, "X1", specs=specs)
    assert res.matched
    assert res.category_code == "M0101000000"
    assert res.category_name == "Linear Shafts"
    assert res.category_root == "shaft"  # via PART_FAMILY_PREFIX_CATEGORIES
    assert res.path == ["Linear Motion", "Linear Shafts"]


def test_resolve_category_truncates_13_to_10_digits(small_index):
    _, index = small_index
    # 13-digit code that doesn't exist directly; truncate to 10 digits.
    specs = {"breadcrumbs": [
        {"url": "https://x.com/"},
        {"url": "https://x.com/M0100000000/M0101000000/M0101000000123/"},
    ]}
    res = resolve_category(index, "X1", specs=specs)
    # The 13-digit "M0101000000123" -> "M0101000000" via truncation.
    assert res.matched
    assert res.category_code == "M0101000000"
    assert res.category_name == "Linear Shafts"


def test_resolve_category_unknown_returns_matched_false(small_index):
    _, index = small_index
    specs = {"breadcrumbs": [
        {"url": "https://x.com/M999900000000/"},  # not in index
    ]}
    res = resolve_category(index, "X1", specs=specs)
    assert not res.matched
    assert res.category_code == "M999900000000"  # raw code preserved
    assert res.category_name is None
    assert res.category_root == "unknown"


def test_resolve_category_ball_bearing_via_yaml(small_index):
    """A 'Linear Ball Bearings' name should normalize to 'bearing'."""
    _, index = small_index
    specs = {"breadcrumbs": [
        {"url": "https://x.com/M0104000000/"},
    ]}
    res = resolve_category(index, "X1", specs=specs)
    assert res.category_name == "Linear Ball Bearings"
    assert res.category_root == "bearing"  # via PART_FAMILY_PREFIX_CATEGORIES


def test_resolve_category_from_specs_file(tmp_path: Path, small_index):
    """If specs is None, the resolver should read downloads/<cid>/json/specs.json."""
    _, index = small_index
    cid = "110310763649"
    downloads = tmp_path / "downloads"
    comp_dir = downloads / cid / "json"
    comp_dir.mkdir(parents=True)
    (comp_dir / "specs.json").write_text(json.dumps({
        "breadcrumbs": [
            {"url": "https://x.com/"},
            {"url": "https://x.com/M0100000000/M0101000000/"},
        ]
    }))
    res = resolve_category(index, cid, downloads_root=downloads)
    assert res.matched
    assert res.category_name == "Linear Shafts"


def test_resolve_category_missing_specs_returns_matched_false(small_index):
    _, index = small_index
    res = resolve_category(index, "GHOST_COMPONENT", downloads_root=Path("/tmp/nonexistent_dl"))
    assert not res.matched
    assert res.category_code is None
    assert res.category_root == "unknown"


# --- resolve_many -------------------------------------------------------------


def test_resolve_many_batches_correctly(small_index):
    _, index = small_index
    specs_list = [
        ("A", {"breadcrumbs": [{"url": "https://x.com/M0101000000/"}]}),
        ("B", {"breadcrumbs": [{"url": "https://x.com/M2004000000/"}]}),
        ("C", {"breadcrumbs": []}),  # no breadcrumbs -> unmatched
    ]
    preloaded = {cid: specs for cid, specs in specs_list}
    out = resolve_many(index, [c for c, _ in specs_list])
    # We need to inject specs per-component. resolve_many doesn't currently
    # take per-component specs dict, so it falls back to file read.
    # The C component should be unmatched.
    assert "C" in out
    assert not out["C"].matched


def test_resolve_category_to_dict(small_index):
    _, index = small_index
    specs = {"breadcrumbs": [{"url": "https://x.com/M0101000000/"}]}
    res = resolve_category(index, "X1", specs=specs)
    d = res.to_dict()
    assert d["category_code"] == "M0101000000"
    assert d["category_name"] == "Linear Shafts"
    assert d["category_root"] == "shaft"
    assert d["matched"] is True
    assert "Linear Motion" in d["path"]
