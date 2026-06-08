"""Table-type classification and seed header synonym map.

The table-merge pipeline operates on three distinct table archetypes, each
with its own merge rules. Before merging, every input table must be
classified as exactly one of:

  - VARIANT_LOOKUP:    keyed by a primary key (e.g. size code "D"); value
                        columns carry the dependent dimensions.
  - CONFIGURABLE_OPTIONS: rows are combinations of option values (e.g.
                        end_form A/B/C × keyway yes/no); few numeric columns.
  - DIMENSION_TABLE:   header is dimension symbols, rows are model_numbers;
                        every column is a numeric dimension.

The DEFAULT_SYNONYMS dict is the seed header-synonym map. It collapses
common engineering-drawing header variants to canonical symbols so the
deterministic merger (Task 8) can do most of the work without LLM calls.

The dynamic LLM-driven synonym expansion lives in
``scripts/table_merge/synonym_llm.py`` (Task 9).
"""
from __future__ import annotations

import re
from enum import Enum
from typing import Optional


class TableType(str, Enum):
    VARIANT_LOOKUP = "variant_lookup"
    CONFIGURABLE_OPTIONS = "configurable_options"
    DIMENSION_TABLE = "dimension_table"
    UNKNOWN = "unknown"


# --- DEFAULT_SYNONYMS ----------------------------------------------------------
# Maps canonical header -> list of accepted aliases (case-insensitive).
# Used by both the deterministic merger (Task 8) and the LLM-based resolver
# (Task 9) as a fallback. Add to this freely; it is a *seed* map, not exhaustive.
DEFAULT_SYNONYMS: dict[str, list[str]] = {
    # Diameters
    "OD": ["OD", "Outer Diameter", "Outer_Diameter", "D_outer", "outer_dia", "D_outer", "main_diameter", "shaft_diameter", "body_outer_diameter"],
    "ID": ["ID", "Inner Diameter", "Inner_Diameter", "D_inner", "inner_dia", "bore", "bore_diameter", "internal_bore_diameter", "d_in"],
    "D":  ["D", "diameter", "through_hole_diameter", "clearance_hole_diameter", "hole_diameter", "pivot_hole_diameter", "mounting_hole_diameter"],
    # Lengths
    "L":  ["L", "Length", "length", "main_length", "overall_length", "body_length", "total_length", "ℓ1", "ℓ", "L_"],
    "W":  ["W", "Width", "width", "overall_width", "body_width"],
    "H":  ["H", "Height", "height", "overall_height", "body_height"],
    "T":  ["T", "Thickness", "thickness", "part_thickness", "base_thickness", "wall_thickness"],
    # Section lengths (shafts)
    "M":  ["M", "reduced_section_length", "left_thread_length", "right_tapped_length"],
    "N":  ["N", "reduced_section_diameter", "secondary_diameter", "tapped_diameter"],
    # Hole patterns
    "hole_count": ["hole_count", "Hole Count", "Quantity of holes", "QTY", "qty", "n_holes", "Number of Holes"],
    "hole_pitch": ["hole_pitch", "Pitch", "Row pitch", "X pitch", "Y pitch", "spacing"],
    # Chamfer / fillet
    "R":  ["R", "radius", "Radius", "fillet_radius", "shoulder_fillet_radius", "outer_corner_radius", "corner_radius"],
    "chamfer": ["chamfer", "Chamfer", "edge_chamfer", "end_chamfer", "corner_chamfer_size", "Corner chamfer size"],
    # Thread
    "thread_size": ["thread_size", "Thread", "thread_spec", "Thread Spec", "thread_nominal_size", "Thread Nominal Size"],
    "thread_length": ["thread_length", "Thread Length", "threaded_length", "threaded_segment_length_in_step"],
    # Catalog
    "model_number": ["model_number", "Model Number", "Part Number", "PartNumber", "SKU", "part_no", "Model No", "Catalog Number"],
    "size_no": ["size_no", "Size", "Size No", "Catalog Size", "size_code", "No.", "No"],
}


# --- Detection heuristics ------------------------------------------------------

# Headers that strongly imply a dimension (numeric column).
_DIMENSION_HINT_RE = re.compile(
    r"^(OD|ID|D|L|W|H|T|M|N|R|diameter|length|width|height|thickness|radius|"
    r"chamfer|pitch|count|size|angle|taper|offset|fillet|thread)",
    re.IGNORECASE,
)

# Headers that strongly imply a categorical / option value.
_OPTION_HINT_RE = re.compile(
    r"(form|type|option|spec|grade|material|finish|seal|keyway|"
    r"end|end_form|tap|thread)$",
    re.IGNORECASE,
)

# Common boolean option values.
_BOOLEAN_VALUES = {"yes", "no", "y", "n", "true", "false", "with", "without", "w/", "w/o", "included", "excluded"}


def _looks_like_dim_table(table: dict) -> bool:
    """Heuristic: header is mostly dimension symbols; rows are model numbers.

    Criteria (all must hold):
      1. At least 3 column headers.
      2. ≥ 60% of headers match _DIMENSION_HINT_RE.
      3. At least one row exists.
      4. First row's first column looks like a model number (alphanumeric, not
         purely numeric).
    """
    headers = table.get("headers") or []
    if len(headers) < 3:
        return False
    if sum(1 for h in headers if _DIMENSION_HINT_RE.search(str(h))) / len(headers) < 0.6:
        return False
    rows = table.get("rows") or []
    if not rows:
        return False
    first_row = rows[0]
    if not first_row:
        return False
    first_val = str(first_row[0] if isinstance(first_row, list) else first_row.get(headers[0], ""))
    if not first_val or first_val.replace(".", "").replace("-", "").isdigit():
        # Purely numeric first column -> could be a variant lookup keyed by a
        # size code, not a model number. Fall through.
        return False
    return True


def _looks_like_variant_lookup(table: dict) -> bool:
    """Heuristic: an explicit primary_key field is set, OR the first column
    is a small-cardinality numeric column (a size code).
    """
    if table.get("primary_key"):
        return True
    headers = table.get("headers") or []
    if not headers:
        return False
    rows = table.get("rows") or []
    if not rows:
        return False
    first_col = [r[0] if isinstance(r, list) else r.get(headers[0]) for r in rows]
    if not first_col:
        return False
    # If the first column is mostly small integers or short codes (≤ 6 chars,
    # alphanumeric, with limited unique values), it's a size-keyed lookup.
    unique = {str(v).strip() for v in first_col if v is not None}
    if 1 < len(unique) <= 30:
        if all(re.match(r"^[\w.-]{1,8}$", v) for v in unique):
            return True
    return False


def _looks_like_options_table(table: dict) -> bool:
    """Heuristic: rows combine multiple option values; few numeric columns.

    Criteria:
      1. ≥ 2 column headers.
      2. ≥ 2 columns match _OPTION_HINT_RE.
      3. At least one row exists.
      4. At least one cell across all rows looks like a boolean option value.
    """
    headers = table.get("headers") or []
    if len(headers) < 2:
        return False
    if sum(1 for h in headers if _OPTION_HINT_RE.search(str(h))) < 2:
        return False
    rows = table.get("rows") or []
    if not rows:
        return False
    for r in rows:
        for cell in r if isinstance(r, list) else r.values():
            if str(cell).strip().lower() in _BOOLEAN_VALUES:
                return True
    return False


def classify_table(table: dict) -> TableType:
    """Classify a table into one of the four TableType values.

    Order of preference:
      1. Explicit ``table['table_type']`` override (if set).
      2. DIMENSION_TABLE (most specific).
      3. CONFIGURABLE_OPTIONS.
      4. VARIANT_LOOKUP.
      5. UNKNOWN.
    """
    if not isinstance(table, dict):
        return TableType.UNKNOWN

    explicit = table.get("table_type")
    if explicit:
        try:
            return TableType(explicit)
        except ValueError:
            pass

    if _looks_like_dim_table(table):
        return TableType.DIMENSION_TABLE
    if _looks_like_options_table(table):
        return TableType.CONFIGURABLE_OPTIONS
    if _looks_like_variant_lookup(table):
        return TableType.VARIANT_LOOKUP
    return TableType.UNKNOWN


def resolve_header(header: str, synonyms: Optional[dict[str, list[str]]] = None) -> str:
    """Resolve a single header to its canonical form.

    Uses ``synonyms`` (defaults to DEFAULT_SYNONYMS) for case-insensitive
    matching against alias lists. Returns the original header (stripped) if
    no match is found.
    """
    syns = synonyms if synonyms is not None else DEFAULT_SYNONYMS
    if not header:
        return ""
    h = str(header).strip()
    if not h:
        return ""
    hl = h.lower()
    for canonical, aliases in syns.items():
        if hl in {a.lower() for a in aliases}:
            return canonical
    return h


def resolve_headers(headers: list[str], synonyms: Optional[dict[str, list[str]]] = None) -> dict[str, str]:
    """Map every header to its canonical form.

    Returns ``{original_header: canonical}``. Headers that already match a
    canonical are mapped to themselves. The function preserves the original
    casing in the dict key for traceability.
    """
    return {h: resolve_header(h, synonyms) for h in headers}
