"""Tests for scripts.table_merge.merge."""
from __future__ import annotations

import pytest

from scripts.table_merge.merge import (
    Conflict,
    MergedTable,
    merge_tables,
)


def _t(cid: str, ti: int, **kwargs) -> dict:
    base = {"component_id": cid, "table_idx": ti}
    base.update(kwargs)
    return base


# --- merge_tables dispatch ----------------------------------------------------

def test_merge_empty_raises():
    with pytest.raises(ValueError):
        merge_tables([])


def test_merge_mixed_types_raises():
    a = _t("A", 0, table_type="dimension_table", headers=["Model", "D", "L"], rows=[["X1", 10, 20]])
    b = _t("B", 0, table_type="variant_lookup", headers=["D", "B"], rows=[[22, 7]])
    with pytest.raises(ValueError):
        merge_tables([a, b])


def test_merge_unknown_type_raises():
    a = _t("A", 0, table_type="garbage_type", headers=["x"], rows=[[1]])
    with pytest.raises(ValueError):
        merge_tables([a])


# --- VARIANT_LOOKUP -----------------------------------------------------------

def test_merge_variant_lookup_unions_keys_no_conflict():
    tables = [
        _t("A", 0, primary_key="D", table_type="variant_lookup",
           headers=["D", "B", "C"],
           rows=[[22, 7, 0.5], [25, 8, 0.5]]),
        _t("B", 0, primary_key="D", table_type="variant_lookup",
           headers=["D", "B", "r"],
           rows=[[22, 7, 0.5], [30, 10, 0.6]]),
    ]
    out = merge_tables(tables)
    assert isinstance(out, MergedTable)
    assert out.table_type.value == "variant_lookup"
    # D=22 has 2 sources, D=25 has 1, D=30 has 1.
    keys = sorted(r.get("D") for r in out.rows)
    assert keys == [22, 25, 30]
    # No conflicts because the B/C values agree where they overlap.
    assert out.conflicts == []


def test_merge_variant_lookup_flags_conflict_on_same_key_different_value():
    tables = [
        _t("A", 0, primary_key="D", table_type="variant_lookup",
           headers=["D", "B"],
           rows=[[22, 7], [25, 8]]),
        _t("B", 0, primary_key="D", table_type="variant_lookup",
           headers=["D", "B"],
           rows=[[22, 7.5], [25, 8]]),  # B=7.5 disagrees with A's B=7
    ]
    out = merge_tables(tables)
    # 22 -> 1 conflict, 25 -> 0 conflicts
    assert len(out.conflicts) == 1
    c = out.conflicts[0]
    assert c.table_type == "variant_lookup"
    assert c.row_key == "22"
    assert c.column == "B"
    assert c.value_a == 7
    assert c.value_b == 7.5
    assert c.source_a.endswith("#t0")
    assert c.source_b.endswith("#t1")  # component B is at table_index 1


def test_merge_variant_lookup_resolves_header_synonyms():
    """Headers should be canonicalized via DEFAULT_SYNONYMS before joining."""
    tables = [
        _t("A", 0, primary_key="D", table_type="variant_lookup",
           headers=["D", "Outer Diameter"],
           rows=[[22, 47]]),
        _t("B", 0, primary_key="D", table_type="variant_lookup",
           headers=["D", "OD"],
           rows=[[22, 47]]),
    ]
    out = merge_tables(tables)
    # Both rows should have the same canonical key (OD) for the second column.
    row = next(r for r in out.rows if r.get("D") == 22)
    assert "OD" in row
    assert "Outer Diameter" not in row
    assert out.conflicts == []


def test_merge_variant_lookup_first_column_used_when_no_primary_key():
    tables = [
        _t("A", 0, table_type="variant_lookup",
           headers=["No.", "B", "C"],
           rows=[[8, 7, 0.5], [10, 8, 0.5]]),
        _t("B", 0, table_type="variant_lookup",
           headers=["No.", "B", "C"],
           rows=[[8, 7, 0.5], [12, 9, 0.5]]),
    ]
    out = merge_tables(tables)
    # "No." resolves to canonical "size_no" via DEFAULT_SYNONYMS.
    keys = sorted(r.get("size_no") for r in out.rows)
    assert keys == [8, 10, 12]
    # D=8 has matching values across A and B -> no conflict.
    assert out.conflicts == []


# --- DIMENSION_TABLE ----------------------------------------------------------

def test_merge_dimension_tables_joins_by_model_number():
    tables = [
        _t("A", 0, table_type="dimension_table",
           headers=["Model Number", "OD", "L"],
           rows=[["X1", 47, 14], ["X2", 52, 15]]),
        _t("B", 0, table_type="dimension_table",
           headers=["PartNumber", "OD", "W"],
           rows=[["X1", 47, 14], ["X3", 60, 16]]),
    ]
    out = merge_tables(tables)
    # X1 appears in both, no conflict. X2 only A, X3 only B.
    # The merge forces the first-column canonical to "model_number" so
    # "Model Number" and "PartNumber" both become that.
    keys = sorted(r.get("model_number") for r in out.rows)
    assert keys == ["X1", "X2", "X3"]
    # OD must be canonicalized (Model Number -> "model_number", PartNumber -> "model_number")
    x1 = next(r for r in out.rows if r.get("model_number") == "X1")
    assert "OD" in x1
    # No conflict because A and B agree on OD=47 for X1.
    assert out.conflicts == []


def test_merge_dimension_tables_flags_row_conflict():
    tables = [
        _t("A", 0, table_type="dimension_table",
           headers=["Model", "OD"],
           rows=[["X1", 47]]),
        _t("B", 0, table_type="dimension_table",
           headers=["Model", "OD"],
           rows=[["X1", 48]]),  # different OD for the same model
    ]
    out = merge_tables(tables)
    assert len(out.conflicts) == 1
    c = out.conflicts[0]
    assert c.table_type == "dimension_table"
    assert c.row_key == "X1"
    assert c.column == "OD"
    assert c.value_a == 47
    assert c.value_b == 48


def test_merge_dimension_tables_missing_values_kept_missing():
    tables = [
        _t("A", 0, table_type="dimension_table",
           headers=["Model", "OD", "ID"],
           rows=[["X1", 47, 20]]),
        _t("B", 0, table_type="dimension_table",
           headers=["Model", "OD"],
           rows=[["X1", 47]]),  # missing ID
    ]
    out = merge_tables(tables)
    x1 = next(r for r in out.rows if r.get("model_number") == "X1")
    assert x1.get("OD") == 47
    assert x1.get("ID") == 20
    # No conflict for missing values.
    assert out.conflicts == []


# --- CONFIGURABLE_OPTIONS -----------------------------------------------------

def test_merge_configurable_options_unions_rows():
    tables = [
        _t("A", 0, table_type="configurable_options",
           headers=["end_form", "keyway", "material"],
           rows=[["A", "yes", "SS"], ["B", "no", "SS"]]),
        _t("B", 0, table_type="configurable_options",
           headers=["end_form", "keyway", "material"],
           rows=[["C", "yes", "AL"], ["D", "no", "SS"]]),
    ]
    out = merge_tables(tables)
    # 4 unique option combos: each tuple of (end_form, keyway, material) is its own row.
    # A/yes/SS, B/no/SS, C/yes/AL, D/no/SS — all distinct.
    assert len(out.rows) == 4
    # No conflicts for option tables.
    assert out.conflicts == []


def test_merge_configurable_options_dedupes_identical_combos():
    """Two sources claiming the same exact option combo should collapse to one row."""
    tables = [
        _t("A", 0, table_type="configurable_options",
           headers=["end_form", "keyway", "material"],
           rows=[["A", "yes", "SS"]]),
        _t("B", 0, table_type="configurable_options",
           headers=["end_form", "keyway", "material"],
           rows=[["A", "yes", "SS"]]),  # same combo, duplicate
    ]
    out = merge_tables(tables)
    assert len(out.rows) == 1
    row = out.rows[0]
    assert row["end_form"] == "A"
    assert row["keyway"] == "yes"
    assert row["material"] == "SS"
    assert row["_n_sources"] == 2  # both A and B contribute


# --- MergedTable.to_dict ------------------------------------------------------

def test_merged_table_to_dict_round_trip():
    tables = [
        _t("A", 0, primary_key="D", table_type="variant_lookup",
           headers=["D", "OD"],
           rows=[[22, 47]]),
    ]
    out = merge_tables(tables)
    d = out.to_dict()
    assert d["table_type"] == "variant_lookup"
    assert "canonical_headers" in d
    assert "rows" in d
    assert "conflicts" in d
    assert "source_component_ids" in d
    assert "source_table_ids" in d
    assert d["source_component_ids"] == ["A"]


# --- Conflict.to_dict ---------------------------------------------------------

def test_conflict_to_dict():
    c = Conflict(
        table_type="variant_lookup",
        row_key="22",
        column="OD",
        value_a=47,
        value_b=48,
        source_a="A#t0",
        source_b="B#t0",
    )
    d = c.to_dict()
    assert d["row_key"] == "22"
    assert d["column"] == "OD"
    assert d["value_a"] == 47
    assert d["value_b"] == 48
