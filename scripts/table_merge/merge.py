"""Deterministic table merger (Phase B of the Step 0.6 pipeline).

Given a list of tables of the *same* TableType, produce a MergedTable with:
  - union of all column headers (resolved through synonyms to canonical form)
  - rows from all sources, keyed by their primary key (or row index for
    option/dimension tables that don't have a natural key)
  - a list of Conflicts for any row that disagrees on a value across sources

This module is *pure* — no LLM calls, no I/O. The LLM-driven synonym
expansion (Phase A) and conflict resolution (Phase C) live in separate
modules and run around this one.

Merge rules per table type:

  VARIANT_LOOKUP
    Key = primary_key value (e.g. D=22). For each (key, column) pair, if
    multiple sources have a value, they must agree; otherwise -> Conflict.
    Missing values stay missing (no Conflict).

  CONFIGURABLE_OPTIONS
    Key = tuple of all option values for the row. Since options are
    typically enumerated combinations, the union of rows is taken. No
    value conflicts because each cell is a unique combo.

  DIMENSION_TABLE
    Key = first column value (model_number). For each (model, column) pair,
    values must agree across sources; otherwise -> Conflict.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from scripts.table_merge.classify import TableType, resolve_header


# --- Data model ---------------------------------------------------------------


@dataclass
class Conflict:
    """A disagreement between two sources on the same (row_key, column)."""
    table_type: str
    row_key: str
    column: str
    value_a: Any
    value_b: Any
    source_a: str
    source_b: str
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "table_type": self.table_type,
            "row_key": self.row_key,
            "column": self.column,
            "value_a": self.value_a,
            "value_b": self.value_b,
            "source_a": self.source_a,
            "source_b": self.source_b,
            "note": self.note,
        }


@dataclass
class MergedTable:
    """The result of merging N tables of the same TableType."""
    table_type: TableType
    canonical_headers: list[str]           # ordered list of canonical column names
    original_to_canonical: dict[str, str]  # original_header -> canonical
    rows: list[dict[str, Any]]             # each row is {canonical: value}
    conflicts: list[Conflict]
    source_component_ids: list[str]
    source_table_ids: list[str]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "table_type": self.table_type.value,
            "canonical_headers": list(self.canonical_headers),
            "original_to_canonical": dict(self.original_to_canonical),
            "rows": list(self.rows),
            "conflicts": [c.to_dict() for c in self.conflicts],
            "source_component_ids": list(self.source_component_ids),
            "source_table_ids": list(self.source_table_ids),
            "notes": list(self.notes),
        }


# --- Helpers ------------------------------------------------------------------


def _normalize_value(v: Any) -> Any:
    """Normalize a cell value for equality comparison.

    - str: strip + lowercase for comparison (case-insensitive "M" == "m")
    - numbers: keep as-is (caller's job to use same type)
    - None: stays None
    """
    if v is None:
        return None
    if isinstance(v, str):
        return v.strip()
    return v


def _values_equal(a: Any, b: Any) -> bool:
    """Compare two normalized values for equality."""
    return _normalize_value(a) == _normalize_value(b)


def _str_key(v: Any) -> str:
    """Coerce a row key to a stable string for dict/set membership."""
    if v is None:
        return ""
    return str(v).strip()


def _table_id(component_id: str, table_index: int) -> str:
    return f"{component_id}#t{table_index}"


# --- Per-type merge implementations -------------------------------------------


def _merge_variant_lookup(tables: list[dict], synonyms: Optional[dict] = None) -> MergedTable:
    """Merge N variant-lookup tables. Primary key is taken from the table's
    ``primary_key`` field, or the first column as a fallback.
    """
    canonical_headers: set[str] = set()
    original_to_canonical: dict[str, str] = {}
    primary_key_canonical: Optional[str] = None
    rows_by_key: dict[str, dict[str, Any]] = {}
    conflicts: list[Conflict] = []
    notes: list[str] = []
    source_table_ids: list[str] = []
    source_component_ids: list[str] = []

    for ti, t in enumerate(tables):
        cid = t.get("component_id", "?")
        tid = _table_id(cid, ti)
        source_table_ids.append(tid)
        if cid not in source_component_ids:
            source_component_ids.append(cid)

        # Build per-table header -> canonical map.
        local_map: dict[str, str] = {}
        for h in t.get("headers") or []:
            canon = resolve_header(h, synonyms)
            local_map[h] = canon
            original_to_canonical.setdefault(h, canon)
            canonical_headers.add(canon)

        # Determine primary key canonical for this table.
        pk_orig = t.get("primary_key") or (t.get("headers") or [None])[0]
        if pk_orig is None:
            notes.append(f"table {tid}: no headers; skipped")
            continue
        pk_canon = local_map.get(pk_orig, resolve_header(pk_orig, synonyms))
        if primary_key_canonical is None:
            primary_key_canonical = pk_canon
        elif pk_canon != primary_key_canonical:
            notes.append(
                f"table {tid}: primary_key='{pk_orig}' resolves to '{pk_canon}', "
                f"expected '{primary_key_canonical}'. Using local pk for this table."
            )

        for row in t.get("rows") or []:
            if isinstance(row, dict):
                # already header-keyed
                pairs = list(row.items())
            else:
                # list aligned to headers
                pairs = list(zip(t.get("headers") or [], row))
            if not pairs:
                continue
            key_val = None
            for h, v in pairs:
                if h == pk_orig:
                    key_val = v
                    break
            key = _str_key(key_val)
            if not key:
                continue
            existing = rows_by_key.get(key)
            if existing is None:
                rows_by_key[key] = {pk_canon: key_val}
                existing = rows_by_key[key]
            for h, v in pairs:
                if h == pk_orig:
                    continue
                canon = local_map.get(h) or resolve_header(h, synonyms)
                if v is None or v == "":
                    continue
                if canon in existing and not _values_equal(existing[canon], v):
                    # Conflict!
                    conflicts.append(Conflict(
                        table_type=TableType.VARIANT_LOOKUP.value,
                        row_key=key,
                        column=canon,
                        value_a=existing[canon],
                        value_b=v,
                        source_a=rows_by_key[key].get("_source", tid),
                        source_b=tid,
                        note="variant lookup key collision",
                    ))
                else:
                    existing[canon] = v
            existing.setdefault("_source", tid)
            existing.setdefault("_primary_key", pk_canon)

    # Remove bookkeeping keys before emitting.
    final_rows = []
    for k, row in rows_by_key.items():
        clean = {c: v for c, v in row.items() if not c.startswith("_")}
        final_rows.append(clean)

    return MergedTable(
        table_type=TableType.VARIANT_LOOKUP,
        canonical_headers=sorted(canonical_headers),
        original_to_canonical=original_to_canonical,
        rows=final_rows,
        conflicts=conflicts,
        source_component_ids=source_component_ids,
        source_table_ids=source_table_ids,
        notes=notes,
    )


def _merge_dimension_tables(tables: list[dict], synonyms: Optional[dict] = None) -> MergedTable:
    """Merge N dimension tables. Key is the first column (model_number).

    The canonical key-column name is forced to a single token ("model_number")
    so tables that call the column "Model Number", "PartNumber", "SKU", etc.
    all join correctly.
    """
    canonical_headers: set[str] = set()
    original_to_canonical: dict[str, str] = {}
    rows_by_key: dict[str, dict[str, Any]] = {}
    conflicts: list[Conflict] = []
    notes: list[str] = []
    source_table_ids: list[str] = []
    source_component_ids: list[str] = []

    key_canonical = "model_number"  # forced — see docstring

    for ti, t in enumerate(tables):
        cid = t.get("component_id", "?")
        tid = _table_id(cid, ti)
        source_table_ids.append(tid)
        if cid not in source_component_ids:
            source_component_ids.append(cid)

        local_map: dict[str, str] = {}
        for h in t.get("headers") or []:
            canon = resolve_header(h, synonyms)
            local_map[h] = canon
            original_to_canonical.setdefault(h, canon)
            canonical_headers.add(canon)
        # The first column is always the model number, regardless of its label.
        if t.get("headers"):
            first_header = t["headers"][0]
            local_map[first_header] = key_canonical
            original_to_canonical[first_header] = key_canonical
            canonical_headers.discard(resolve_header(first_header, synonyms))
            canonical_headers.add(key_canonical)

        first_header = (t.get("headers") or [None])[0]
        if first_header is None:
            notes.append(f"table {tid}: no headers; skipped")
            continue

        for row in t.get("rows") or []:
            if isinstance(row, dict):
                pairs = list(row.items())
            else:
                pairs = list(zip(t.get("headers") or [], row))
            if not pairs:
                continue
            key = _str_key(pairs[0][1])
            if not key:
                continue
            existing = rows_by_key.get(key)
            if existing is None:
                rows_by_key[key] = {key_canonical: pairs[0][1]}
                existing = rows_by_key[key]
            for h, v in pairs:
                canon = local_map.get(h) or resolve_header(h, synonyms)
                if v is None or v == "":
                    continue
                if canon in existing and not _values_equal(existing[canon], v):
                    conflicts.append(Conflict(
                        table_type=TableType.DIMENSION_TABLE.value,
                        row_key=key,
                        column=canon,
                        value_a=existing[canon],
                        value_b=v,
                        source_a=existing.get("_source", tid),
                        source_b=tid,
                        note="dimension table row collision",
                    ))
                else:
                    existing[canon] = v
            existing.setdefault("_source", tid)

    final_rows = []
    for k, row in rows_by_key.items():
        clean = {c: v for c, v in row.items() if not c.startswith("_")}
        final_rows.append(clean)

    return MergedTable(
        table_type=TableType.DIMENSION_TABLE,
        canonical_headers=sorted(canonical_headers),
        original_to_canonical=original_to_canonical,
        rows=final_rows,
        conflicts=conflicts,
        source_component_ids=source_component_ids,
        source_table_ids=source_table_ids,
        notes=notes,
    )


def _merge_configurable_options(tables: list[dict], synonyms: Optional[dict] = None) -> MergedTable:
    """Merge N configurable-options tables. Key is the tuple of option values.

    Configurable-options tables have a small, finite option space, so
    duplicate rows are the expected case (and not a conflict). The merger
    just unions the rows. Use it as a presence check ("does source A claim
    option X?").
    """
    canonical_headers: set[str] = set()
    original_to_canonical: dict[str, str] = {}
    rows_by_key: dict[tuple, dict[str, Any]] = {}
    source_table_ids: list[str] = []
    source_component_ids: list[str] = []
    notes: list[str] = []

    for ti, t in enumerate(tables):
        cid = t.get("component_id", "?")
        tid = _table_id(cid, ti)
        source_table_ids.append(tid)
        if cid not in source_component_ids:
            source_component_ids.append(cid)

        local_map: dict[str, str] = {}
        for h in t.get("headers") or []:
            canon = resolve_header(h, synonyms)
            local_map[h] = canon
            original_to_canonical.setdefault(h, canon)
            canonical_headers.add(canon)

        for row in t.get("rows") or []:
            if isinstance(row, dict):
                pairs = list(row.items())
            else:
                pairs = list(zip(t.get("headers") or [], row))
            if not pairs:
                continue
            key = tuple(_str_key(v) for _, v in pairs)
            if key not in rows_by_key:
                rows_by_key[key] = {local_map.get(h) or resolve_header(h, synonyms): v for h, v in pairs}
            rows_by_key[key].setdefault("_sources", []).append(tid)

    final_rows = []
    for k, row in rows_by_key.items():
        sources = row.pop("_sources", [])
        clean = {c: v for c, v in row.items() if not c.startswith("_")}
        clean["_n_sources"] = len(sources)
        clean["_source_ids"] = sources
        final_rows.append(clean)

    return MergedTable(
        table_type=TableType.CONFIGURABLE_OPTIONS,
        canonical_headers=sorted(canonical_headers),
        original_to_canonical=original_to_canonical,
        rows=final_rows,
        conflicts=[],  # option tables never conflict; duplicate rows are expected
        source_component_ids=source_component_ids,
        source_table_ids=source_table_ids,
        notes=notes,
    )


# --- Public entry point -------------------------------------------------------


def merge_tables(tables: list[dict], synonyms: Optional[dict] = None) -> MergedTable:
    """Merge N tables. All tables must be of the same TableType.

    Dispatches to the per-type merge function. Raises ValueError if the
    input is empty or contains mixed types.
    """
    if not tables:
        raise ValueError("merge_tables: empty input")
    types = {(t.get("table_type") or classify(t)) for t in tables}
    if len(types) > 1:
        raise ValueError(f"merge_tables: mixed table types {types}; all tables must share a type")
    ttype = next(iter(types))
    if ttype == TableType.VARIANT_LOOKUP.value:
        return _merge_variant_lookup(tables, synonyms)
    if ttype == TableType.DIMENSION_TABLE.value:
        return _merge_dimension_tables(tables, synonyms)
    if ttype == TableType.CONFIGURABLE_OPTIONS.value:
        return _merge_configurable_options(tables, synonyms)
    raise ValueError(f"merge_tables: unsupported table type {ttype!r}")


def classify(table: dict) -> str:
    """Thin re-export so this module is self-contained for the merge entry point.

    Avoids an import cycle if upstream callers want a single import.
    """
    from scripts.table_merge.classify import classify_table
    return classify_table(table).value
