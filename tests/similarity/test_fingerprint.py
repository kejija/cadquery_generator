"""Tests for scripts.similarity.fingerprint."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.similarity.fingerprint import (
    CATEGORY_SYNONYMS,
    ComponentFingerprint,
    _normalize_category,
    build_fingerprint,
)


REAL_TEMPLATE = Path(
    "/home/keji/fe/cadquery_generator/output/feature_templates/110300324920.feature_template.json"
)


# --- category normalization ----------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("ball_bearing", "bearing"),
        ("angular_contact_bearing", "bearing"),
        ("cylindrical_roller_bearing", "bearing"),
        ("tapered_roller_bearing", "bearing"),
        ("needle_roller_bearing", "bearing"),
        ("thrust_bearing", "bearing"),
        ("linear_shaft", "shaft"),
        ("support_shaft", "shaft"),
        ("precision_shaft", "shaft"),
        ("motor_shaft", "shaft"),
    ],
)
def test_known_synonyms_normalize_to_root(raw, expected):
    assert _normalize_category(raw) == expected


def test_unknown_category_returns_unknown_for_empty():
    assert _normalize_category("") == "unknown"


def test_unknown_category_returns_unknown_for_none():
    assert _normalize_category(None) == "unknown"


def test_unmapped_category_uses_first_token():
    assert _normalize_category("widget_gizmo") == "widget"


def test_synonyms_dict_covers_documented_cases():
    # Sanity: if a synonym is dropped from CATEGORY_SYNONYMS, fail loudly.
    assert "ball_bearing" in CATEGORY_SYNONYMS
    assert "angular_contact_bearing" in CATEGORY_SYNONYMS
    assert "linear_shaft" in CATEGORY_SYNONYMS


def test_normalize_is_case_insensitive():
    assert _normalize_category("BALL_BEARING") == "bearing"
    assert _normalize_category("Ball_Bearing") == "bearing"


# --- build_fingerprint ---------------------------------------------------------

def test_build_fingerprint_from_real_component():
    if not REAL_TEMPLATE.exists():
        pytest.skip(f"real feature_template not present at {REAL_TEMPLATE}")
    fp = build_fingerprint(REAL_TEMPLATE, "110300324920")
    assert isinstance(fp, ComponentFingerprint)
    assert fp.component_id == "110300324920"
    # No category_code in the template, so category_root must be 'unknown'.
    assert fp.category_root == "unknown"
    assert fp.template_signature is not None
    assert "simplified_single_body" in fp.template_signature
    assert fp.text_for_embedding != ""
    # value_keys must come from geometry params; if none, from the full set.
    assert isinstance(fp.value_keys, list)


def test_build_fingerprint_handles_missing_optional_fields(tmp_path: Path):
    minimal = tmp_path / "empty.feature_template.json"
    minimal.write_text(json.dumps({}))
    fp = build_fingerprint(minimal, "EMPTY")
    assert fp.component_id == "EMPTY"
    assert fp.category_root == "unknown"
    assert fp.template_signature is None
    assert fp.variant_count == 0
    assert fp.attribute_keys == []
    assert fp.value_keys == []
    # name falls back to component_id
    assert fp.name == "EMPTY"
    assert fp.text_for_embedding  # at minimum the mode: tag is present


def test_build_fingerprint_uses_geometry_categorized_params(tmp_path: Path):
    doc = {
        "modeling_mode": "extruded_prism",
        "parameters": [
            {"name": "OD", "category": "geometry", "parameter_type": "length"},
            {"name": "ID", "category": "geometry", "parameter_type": "length"},
            {"name": "B", "category": "geometry", "parameter_type": "length"},
            {"name": "size_no", "parameter_type": "integer", "category": "catalog"},
        ],
    }
    p = tmp_path / "bearing.feature_template.json"
    p.write_text(json.dumps(doc))
    fp = build_fingerprint(p, "BRG1")
    assert fp.value_keys == ["B", "ID", "OD"]
    assert fp.template_signature is not None
    assert fp.template_signature.startswith("extruded_prism|")


def test_build_fingerprint_falls_back_to_length_typed_params(tmp_path: Path):
    doc = {
        "modeling_mode": "solid_of_revolution",
        "parameters": [
            {"name": "L", "parameter_type": "length"},  # no category tag
            {"name": "D", "parameter_type": "length"},
            {"name": "size_no", "parameter_type": "integer"},
        ],
    }
    p = tmp_path / "shaft.feature_template.json"
    p.write_text(json.dumps(doc))
    fp = build_fingerprint(p, "SHAFT1")
    assert set(fp.value_keys) == {"L", "D"}


def test_build_fingerprint_includes_category_in_text(tmp_path: Path):
    doc = {
        "part_family": "Deep Groove Ball Bearing 6204",
        "modeling_mode": "simplified_single_body",
        "metadata": {"category_code": "ball_bearing"},
        "parameters": [{"name": "OD", "category": "geometry"}],
    }
    p = tmp_path / "x.feature_template.json"
    p.write_text(json.dumps(doc))
    fp = build_fingerprint(p, "X1")
    assert "category:ball_bearing" in fp.text_for_embedding
    assert fp.category_root == "bearing"


def test_build_fingerprint_raises_on_bad_path(tmp_path: Path):
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError):
        build_fingerprint(missing, "GHOST")


def test_build_fingerprint_raises_on_malformed_json(tmp_path: Path):
    p = tmp_path / "broken.feature_template.json"
    p.write_text("{ this is not json")
    with pytest.raises(ValueError) as exc_info:
        build_fingerprint(p, "BROKEN")
    assert "BROKEN" in str(exc_info.value)


# --- to_dict -------------------------------------------------------------------

def test_fingerprint_to_dict_round_trip():
    fp = ComponentFingerprint(
        component_id="A",
        name="A",
        description="desc",
        category_code="ball_bearing",
        category_root="bearing",
        attribute_keys=["brand"],
        value_keys=["OD", "ID"],
        variant_count=4,
        template_signature="single_body|OD|ID",
        text_for_embedding="A | desc | category:ball_bearing",
    )
    d = fp.to_dict()
    for k in (
        "component_id", "name", "description", "category_code", "category_root",
        "attribute_keys", "value_keys", "variant_count",
        "template_signature", "text_for_embedding",
    ):
        assert k in d
    assert d["category_root"] == "bearing"
    # asdict preserves the dataclass field order; the constructor stored them
    # in the order we passed in. We just check the SET matches.
    assert sorted(d["value_keys"]) == ["ID", "OD"]
    assert d["variant_count"] == 4
