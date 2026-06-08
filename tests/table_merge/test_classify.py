"""Tests for scripts.table_merge.classify."""
from __future__ import annotations

import pytest

from scripts.table_merge.classify import (
    DEFAULT_SYNONYMS,
    TableType,
    classify_table,
    resolve_header,
    resolve_headers,
)


# --- classify_table ------------------------------------------------------------

def test_classify_explicit_override_wins():
    table = {
        "table_type": "dimension_table",
        "headers": ["A", "B"],
        "rows": [["x", 1]],
    }
    assert classify_table(table) == TableType.DIMENSION_TABLE


def test_classify_explicit_override_invalid_falls_through():
    table = {
        "table_type": "garbage_value",
        "headers": ["D", "L", "OD", "ID"],
        "rows": [["M1", 1, 2, 3]],
    }
    # Should classify as dimension_table by the heuristic.
    assert classify_table(table) == TableType.DIMENSION_TABLE


def test_classify_dimension_table():
    table = {
        "headers": ["Model Number", "OD", "ID", "L", "W", "H"],
        "rows": [
            ["BRG-6204", 47, 20, 14, 14, 14],
            ["BRG-6205", 52, 25, 15, 15, 15],
        ],
    }
    assert classify_table(table) == TableType.DIMENSION_TABLE


def test_classify_variant_lookup_with_explicit_primary_key():
    table = {
        "primary_key": "D",
        "headers": ["D", "B", "C"],
        "rows": [[22, 7, 0.5], [25, 8, 0.5]],
    }
    assert classify_table(table) == TableType.VARIANT_LOOKUP


def test_classify_variant_lookup_by_small_cardinality_first_col():
    table = {
        "headers": ["No.", "B", "C"],
        "rows": [[8, 7, 0.5], [10, 8, 0.5], [12, 9, 0.5]],
    }
    # First column is small-cardinality numeric (size codes 8/10/12).
    assert classify_table(table) == TableType.VARIANT_LOOKUP


def test_classify_configurable_options():
    table = {
        "headers": ["end_form", "keyway", "material", "thread_type"],
        "rows": [
            ["A", "yes", "SS", "M"],
            ["B", "no", "CS", "F"],
            ["C", "yes", "AL", "M"],
        ],
    }
    # 3 option-like columns (end_form, keyway, material) + boolean cells.
    assert classify_table(table) == TableType.CONFIGURABLE_OPTIONS


def test_classify_unknown_for_empty_table():
    assert classify_table({}) == TableType.UNKNOWN


def test_classify_unknown_for_non_dict_input():
    assert classify_table("not a table") == TableType.UNKNOWN  # type: ignore[arg-type]
    assert classify_table(None) == TableType.UNKNOWN  # type: ignore[arg-type]


def test_classify_too_few_headers_unknown():
    table = {"headers": ["D", "L"], "rows": [[1, 2]]}
    # Only 2 headers — too few for either dimension_table or options.
    assert classify_table(table) == TableType.UNKNOWN


# --- resolve_header / resolve_headers -----------------------------------------

def test_resolve_header_known_synonyms():
    syns = {"OD": ["OD", "Outer Diameter", "D_outer"]}
    assert resolve_header("OD", syns) == "OD"
    assert resolve_header("Outer Diameter", syns) == "OD"
    assert resolve_header("outer diameter", syns) == "OD"
    assert resolve_header("D_outer", syns) == "OD"


def test_resolve_header_unknown_passthrough():
    syns = {"OD": ["OD"]}
    assert resolve_header("Widget", syns) == "Widget"


def test_resolve_header_empty_input():
    assert resolve_header("", {"OD": ["OD"]}) == ""
    assert resolve_header("   ", {"OD": ["OD"]}) == ""


def test_resolve_header_uses_default_synonyms():
    # Without an explicit synonyms arg, DEFAULT_SYNONYMS is used.
    assert resolve_header("Outer Diameter") == "OD"
    assert resolve_header("Bore") == "ID"  # DEFAULT_SYNONYMS has Bore -> ID
    assert resolve_header("Main Diameter") == "OD"


def test_resolve_headers_batch():
    headers = ["OD", "Inner Diameter", "width", "WeirdField"]
    out = resolve_headers(headers)
    assert out == {"OD": "OD", "Inner Diameter": "ID", "width": "W", "WeirdField": "WeirdField"}


def test_default_synonyms_covers_documented_cases():
    # Sanity check: documented engineering synonyms are present.
    for canonical, alias in [
        ("OD", "Outer Diameter"),
        ("ID", "Inner Diameter"),
        ("L", "Length"),
        ("T", "Thickness"),
        ("R", "Radius"),
    ]:
        assert canonical in DEFAULT_SYNONYMS
        assert alias in DEFAULT_SYNONYMS[canonical], f"Missing {canonical} -> {alias}"


def test_classify_then_resolve_workflow():
    """End-to-end: classify a table, then resolve its headers to canonical."""
    table = {
        "headers": ["Model Number", "Outer Diameter", "Inner Diameter", "Length", "Width"],
        "rows": [["X1", 47, 20, 14, 14]],
    }
    ttype = classify_table(table)
    assert ttype == TableType.DIMENSION_TABLE
    canonical = resolve_headers(table["headers"])
    assert canonical["Outer Diameter"] == "OD"
    assert canonical["Inner Diameter"] == "ID"
    assert canonical["Length"] == "L"
    assert canonical["Width"] == "W"
