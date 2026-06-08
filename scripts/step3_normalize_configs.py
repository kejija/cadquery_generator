#!/usr/bin/env python3
"""
Step 3 — Normalize Datasheet Configuration Tables.

Reads each <component_id>'s specs.json (falling back to specs_after.json),
extracts part-number dimension tables, and emits a normalized
configurations JSON per Step 1 template.

This step is DETERMINISTIC by default. It uses heuristics:
  1. Find a table whose first row contains the literal "Model" or
     "Part Number" / "Part No." / "Type" header.
  2. Treat the first row as column headers.
  3. Each subsequent row is a part-number configuration.
  4. Numeric cells are coerced to numbers; thread specs ("M5", "#10-32")
     are kept as strings; blanks become null.
  5. Each cell is matched against the template's parameter symbols;
     extra columns are recorded as `extras`.

If deterministic extraction finds zero rows but the specs file has tables,
or if --allow-codex is passed, a single codex call is made to parse the
hard tables. The codex output is validated against a strict shape; any
failure falls back to a single-row default.

Outputs (per component):
  output/normalized_configs/<component_id>.configurations.json
    - schema-validated against schemas/normalized_configurations.schema.json
    - one configurations[] entry per part-number row

Usage:
  python scripts/step3_normalize_configs.py
  python scripts/step3_normalize_configs.py --limit 3
  python scripts/step3_normalize_configs.py --component 110300324920
  python scripts/step3_normalize_configs.py --allow-codex --codex-model gpt-5.4-mini
  python scripts/step3_normalize_configs.py --json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import jsonschema

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEMPLATES_DIR = REPO_ROOT / "output" / "feature_templates"
DEFAULT_DOWNLOADS_DIR = REPO_ROOT / "downloads"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "normalized_configs"
DEFAULT_SCHEMA = REPO_ROOT / "schemas" / "normalized_configurations.schema.json"
DEFAULT_CODEX_MODEL = "gpt-5.4-mini"

# Module-level diagnostic buffer (single-call scope: cleared at start of
# extract_configurations, consumed in process_one).
PER_TABLE_DIAG: list[dict] = []

# Header tokens that mark a part-number dimension table
PART_NUMBER_HEADERS = {
    "model", "part number", "part no", "part no.", "part_number",
    "type", "catalog no", "catalog number", "item", "no.", "no",
    "partnumber",
}
# Pattern for cleaning header text
_HEADER_CLEAN_RE = re.compile(r"[^a-z0-9]+")

# Pattern for a numeric cell (allow signed decimals, leading zeros, dot/decimal)
_NUMERIC_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")
# Thread spec patterns
_THREAD_RE = re.compile(r"^(M\d+(?:\.\d+)?(?:\s*x\s*\d+(?:\.\d+)?)?)$", re.IGNORECASE)
_IMPERIAL_THREAD_RE = re.compile(r"^#\d+(?:-\d+)?$")
_TOLERANCE_RE = re.compile(r"^±\d+(?:\.\d+)?$|^\+\d+(?:\.\d+)?$|^-\d+(?:\.\d+)?$")


# ----------------------------------------------------------------------------
# Data model
# ----------------------------------------------------------------------------
@dataclass
class ComponentResult:
    component_id: str
    template_id: str | None
    catalog_id: str | None
    method: str
    n_configurations: int
    parameter_symbols: list[str]
    warnings: list[str] = field(default_factory=list)
    codex_used: bool = False
    codex_error: str | None = None
    output_path: str | None = None
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------------
# Specs loading
# ----------------------------------------------------------------------------
def load_specs(component_dir: Path) -> tuple[dict | None, str | None]:
    """Return (specs_dict, path_used). Prefers specs_after.json when it
    actually has content; falls back to specs.json. A file is considered
    empty if it has 0 tables and 0 configFields."""
    json_dir = component_dir / "json"
    candidates: list[Path] = []
    for fname in ("specs_after.json", "specs.json"):
        p = json_dir / fname
        if p.exists():
            candidates.append(p)

    def _is_empty(d: dict) -> bool:
        return (not d.get("tables")) and (not d.get("configFields")) and (not d.get("specImages"))

    last_err: str | None = None
    for p in candidates:
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError as e:
            return None, f"{p.name}: invalid JSON ({e})"
        if not _is_empty(d):
            return d, str(p.relative_to(REPO_ROOT))
        last_err = f"{p.name}: empty (0 tables / 0 configFields)"

    # All candidates empty (or only one existed and was empty). Return the
    # first one so the caller still gets *something* to inspect.
    if candidates:
        try:
            return json.loads(candidates[0].read_text()), str(candidates[0].relative_to(REPO_ROOT))
        except json.JSONDecodeError:
            pass
    return None, last_err or "no specs.json or specs_after.json found"


def load_template(component_id: str, templates_dir: Path) -> tuple[dict | None, str | None]:
    p = templates_dir / f"{component_id}.feature_template.json"
    if not p.exists():
        return None, None
    return json.loads(p.read_text()), p.name


# ----------------------------------------------------------------------------
# Header cleaning & value coercion
# ----------------------------------------------------------------------------
def clean_header(s: str) -> str:
    return _HEADER_CLEAN_RE.sub("_", s.strip().lower()).strip("_")


def is_part_number_header(s: str) -> bool:
    return clean_header(s) in PART_NUMBER_HEADERS


def coerce_value(raw: str) -> Any:
    """Coerce a cell string to a typed value."""
    s = raw.strip()
    if s == "" or s in {"-", "—", "N/A", "n/a", "—"}:
        return None
    if _NUMERIC_RE.match(s):
        # int if no decimal, else float
        f = float(s)
        if f.is_integer():
            return int(f)
        return round(f, 6)
    if _THREAD_RE.match(s):
        return s  # M5, M3x0.5
    if _IMPERIAL_THREAD_RE.match(s):
        return s
    if _TOLERANCE_RE.match(s):
        return s
    return s


def header_to_symbol(header: str, template_params: list[dict]) -> str | None:
    """Try to map a raw header to a known template parameter symbol.
    Returns the template symbol if a match is found, else the cleaned header."""
    cleaned = clean_header(header)
    if not cleaned:
        return None
    # Exact match against parameter.name or parameter.symbol
    for p in template_params:
        if clean_header(p.get("name", "")) == cleaned:
            return p.get("name")
        sym = p.get("symbol")
        if sym and clean_header(str(sym)) == cleaned:
            return p.get("name")
    # Common synonyms
    synonyms = {
        "d": "D", "d1": "D1", "d2": "D2", "b": "B", "l": "L",
        "h": "H", "h1": "H1", "w": "W", "t": "T", "p": "P",
        "m": "M", "m1": "M1", "no": "no", "no_": "no",
        "len": "length", "length": "L", "od": "D1", "id": "D",
        "bore": "D", "width": "B",
    }
    if cleaned in synonyms:
        return synonyms[cleaned]
    return cleaned  # best-effort: cleaned header


# ----------------------------------------------------------------------------
# Deterministic table parser
# ----------------------------------------------------------------------------
def looks_like_part_number_cell(s: str) -> bool:
    """Heuristic: a cell is a part number if it has a letter prefix followed
    by alphanumeric characters (digits can be anywhere after the first letter).
    Examples accepted: 'MCSCS10', 'SL-SSCDN10', 'HGBPL1', 'EFS40120-8', 'MCSCN'."""
    s = s.strip()
    if not s:
        return False
    if is_part_number_header(s):
        return False
    # Letter prefix + any alphanumerics/dashes/underscores, must contain at
    # least one letter and at least one digit somewhere.
    if not re.match(r"^[A-Z][A-Z0-9][A-Z0-9_.\-]*$", s):
        return False
    has_letter = bool(re.search(r"[A-Z]", s))
    has_digit = bool(re.search(r"\d", s))
    # Pure-letter "family prefixes" (e.g. "MCSCN") are part-numbers only
    # when there is a companion numeric column we're about to concatenate.
    # Standalone, they're usually headers/family names.
    if has_letter and not has_digit:
        return False
    return True


def parse_table_deterministic(table: dict, template_params: list[dict]) -> tuple[list[dict] | None, str]:
    """Try to extract configurations from a single table.
    Returns (configurations, reason) where reason explains the outcome:
      - ("rows", "ok"): parsed N rows successfully
      - ("rows", "skipped_text_row"): one or more rows were non-numeric and skipped
      - (None, "not_pn_table"): no part-number column found
      - ([], "empty"): part-number column found but no usable rows
    """
    rows = table.get("rows") or []
    if not rows or len(rows) < 2:
        return [], "empty"
    headers = rows[0]
    if not isinstance(headers, list):
        return [], "empty"

    # Find part-number column index. Prefer "Part Number" / "Model" / "Type"
    # headers over the more ambiguous "No." / "Size" headers.
    pn_idx = None
    pn_priority = [
        "part_number", "partnumber", "model", "part_number_", "part no",
        "catalog_no", "catalog number", "item",
    ]
    for i, h in enumerate(headers):
        if not isinstance(h, str):
            continue
        if clean_header(h) in pn_priority:
            pn_idx = i
            break
    if pn_idx is None:
        for i, h in enumerate(headers):
            if isinstance(h, str) and is_part_number_header(h):
                pn_idx = i
                break
    if pn_idx is None:
        return None, "not_pn_table"

    # Optional variant column — pick the one with the most distinct values
    # (a real "Type" column will have many values; a family-prefix column has 1).
    variant_candidates: list[tuple[int, int]] = []  # (col_idx, distinct_count)
    for i, h in enumerate(headers):
        if i == pn_idx:
            continue
        if not isinstance(h, str):
            continue
        if clean_header(h) in {"type", "variant", "style", "series"}:
            distinct = set()
            for r in rows[1:]:
                if isinstance(r, list) and i < len(r):
                    v = str(r[i]).strip()
                    if v:
                        distinct.add(v)
            variant_candidates.append((i, len(distinct)))
    variant_idx = max(variant_candidates, key=lambda x: x[1])[0] if variant_candidates and max(v for _, v in variant_candidates) > 1 else None

    # Detect multi-column part-number (Type + No. pattern)
    no_col_idx = None
    type_col_idx = None
    for i, h in enumerate(headers):
        if i == pn_idx or i == variant_idx:
            continue
        if isinstance(h, str):
            ch = clean_header(h)
            if ch in {"no", "no_", "size", "size_no"}:
                no_col_idx = i
            elif ch in {"type", "series"}:
                type_col_idx = i

    # Build parameter symbol map for remaining columns
    col_meta: list[tuple[int, str | None, str]] = []
    for i, h in enumerate(headers):
        if i in (pn_idx, variant_idx, no_col_idx, type_col_idx):
            continue
        if not isinstance(h, str) or not h.strip():
            col_meta.append((i, None, ""))
            continue
        sym = header_to_symbol(h, template_params)
        col_meta.append((i, sym, h))

    configs: list[dict] = []
    skipped_text = 0
    not_dimension_table = False
    last_family_prefix: str | None = None
    for row_idx, row in enumerate(rows[1:], start=1):
        if not isinstance(row, list) or len(row) <= pn_idx:
            continue
        # Skip rows where the part-number cell looks like a sub-header
        # (all-text continuation of the column header block, e.g. "Type / No.")
        pn_cell = str(row[pn_idx]).strip()
        if not pn_cell:
            # Inherit family prefix from previous row if it was a letter-only prefix
            if last_family_prefix:
                pn_cell = last_family_prefix
            else:
                continue
        # Skip if pn cell is purely a non-part-number text and >60% of row is non-numeric
        non_empty = [c for c in row if str(c).strip()]
        if non_empty:
            numeric_count = sum(
                1 for c in non_empty
                if _NUMERIC_RE.match(str(c).strip())
                or _THREAD_RE.match(str(c).strip())
                or str(c).strip() in {"", "-", "—"}
            )
            if numeric_count / len(non_empty) < 0.3:
                skipped_text += 1
                continue

        # If we have no value-bearing columns at all (everything was
        # skipped-text), this is probably a configuration-options table,
        # not a dimension table.
        if not col_meta:
            not_dimension_table = True
            continue

        # If the row is left-shifted (first cell is a small integer, the
        # rest are dimension values), the family prefix from a previous
        # row is implicit. Inherit it.
        row_is_shifted = len(row) > len(headers)
        if (last_family_prefix
                and pn_cell != last_family_prefix
                and re.match(r"^\d{1,3}$", pn_cell)
                and (row_is_shifted or len(row) == len(headers))):
            composed_via_inherit = f"{last_family_prefix}{pn_cell}"
        else:
            composed_via_inherit = None

        # If the row is shifted (len > headers), the data columns are
        # offset by 1. Shift the cell index for col_meta lookups.
        col_offset = 1 if row_is_shifted else 0

        # If this looks like a multi-column PN row, try to compose.
        # We compose when EITHER:
        #   - pn_cell is a complete PN (has digits) AND no_col is numeric
        #   - pn_cell is a family prefix (no digits, all letters) AND sibling
        #     column is numeric (size selector pattern)
        #   - pn_cell is empty AND no_col is a complete PN
        #   - pn_cell was inherited from a previous family prefix row
        composed_pn = pn_cell
        if composed_via_inherit:
            composed_pn = composed_via_inherit
        elif no_col_idx is not None and no_col_idx < len(row):
            no_val = str(row[no_col_idx]).strip()
            pn_has_digit = bool(re.search(r"\d", pn_cell))
            no_is_numeric = bool(_NUMERIC_RE.match(no_val))
            no_is_complete_pn = looks_like_part_number_cell(no_val)
            if no_is_numeric and pn_cell and (looks_like_part_number_cell(pn_cell) or (re.match(r"^[A-Z][A-Z0-9_.\-]*$", pn_cell) and not pn_has_digit)):
                # Family prefix + numeric No. -> compose
                composed_pn = f"{pn_cell}{no_val}"
            elif no_is_complete_pn and not looks_like_part_number_cell(pn_cell) and pn_cell:
                composed_pn = f"{pn_cell}{no_val}"
            elif not pn_cell and no_is_complete_pn:
                composed_pn = no_val
        else:
            # No explicit No. column: try any sibling column that is purely
            # numeric and looks like a size selector (small integer, e.g. 8, 10, 12).
            if pn_cell and not re.search(r"\d", pn_cell) and re.match(r"^[A-Z][A-Z0-9_.\-]*$", pn_cell):
                # Family prefix in pn column. Look at the next column.
                for ci in range(pn_idx + 1, len(headers)):
                    if ci >= len(row):
                        break
                    v = str(row[ci]).strip()
                    if _NUMERIC_RE.match(v) and 0 < float(v) < 1000:
                        # Heuristic: only use this if the value is a small integer (likely a size)
                        if float(v).is_integer() and float(v) < 100:
                            composed_pn = f"{pn_cell}{int(float(v))}"
                            break

        if not looks_like_part_number_cell(composed_pn) and not looks_like_part_number_cell(pn_cell):
            continue

        final_pn = composed_pn if looks_like_part_number_cell(composed_pn) else pn_cell

        # Track family prefix for subsequent rows. Update only on letter-only
        # prefix rows; digit-only rows (sizes within the same family) do NOT
        # reset the prefix, so a chain like ['MCSCN', '8', '10', '12'] all
        # inherit the MCSCN prefix.
        if re.match(r"^[A-Z][A-Z0-9_.\-]*$", pn_cell) and not re.search(r"\d", pn_cell) and not looks_like_part_number_cell(pn_cell):
            last_family_prefix = pn_cell

        values: dict[str, Any] = {}
        for ci, sym, raw in col_meta:
            actual_ci = ci + col_offset
            if actual_ci >= len(row):
                continue
            cell = row[actual_ci]
            val = coerce_value(str(cell))
            if sym:
                values[sym] = val

        variant = "standard"
        if variant_idx is not None:
            actual_vi = variant_idx + col_offset
            if actual_vi < len(row):
                v = str(row[actual_vi]).strip()
                if v:
                    variant = v

        configs.append({
            "part_number": final_pn,
            "variant": variant,
            "values": values,
            "extras": {},
            "source_table_idx": table.get("idx"),
            "source_row_idx": row_idx,
        })

    # Priority for reason:
    #   - no rows at all -> empty
    #   - some rows extracted -> ok (or skipped_text_row if some non-data rows existed)
    #   - all rows were non-numeric AND no col_meta -> not_dimension_table
    if not configs:
        if not col_meta:
            reason = "not_dimension_table"
        elif not_dimension_table:
            reason = "not_dimension_table"
        else:
            reason = "empty"
    else:
        reason = "ok" if not skipped_text else "ok_with_text_skip"
    return configs, reason


# ----------------------------------------------------------------------------
# Codex fallback (optional)
# ----------------------------------------------------------------------------
CODEX_SYSTEM_PROMPT = """You extract part-number dimension tables from messy
MISUMI datasheet JSON. Input is a JSON dump of all `tables[*].rows` from a
specs file plus a list of template parameter symbols.

Output ONLY a JSON object of this shape:
{
  "configurations": [
    {
      "part_number": "MCSCS10",
      "variant": "standard",
      "values": {"D": 10, "B": 15, "M": "M5"}
    }
  ],
  "warnings": ["..."]
}

Rules:
- Emit one entry per part-number row.
- Numeric cells -> numbers. Thread specs like "M5" -> strings.
- Unknown columns: omit.
- If you cannot find a part-number table, return
  {"configurations": [], "warnings": ["no part-number table found"]}.
- Do not invent values.
"""


def call_codex_parse(specs_dump: str, template_params: list[dict],
                     model: str, timeout_s: int = 180) -> dict | None:
    user_payload = {
        "template_parameter_symbols": [
            {"name": p.get("name"), "symbol": p.get("symbol"), "units": p.get("units")}
            for p in template_params
        ],
        "tables": specs_dump[:60000],
    }
    prompt = (
        "Extract the part-number dimension table from this specs.json. "
        "Output only a JSON object as specified.\n"
        f"```json\n{json.dumps(user_payload, indent=2)}\n```"
    )
    cmd = [
        "codex", "exec",
        "-m", model,
        "--sandbox", "read-only",
        "--output-last-message", "-",
        prompt,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired:
        return {"_error": f"codex timed out after {timeout_s}s"}
    except FileNotFoundError:
        return {"_error": "codex CLI not found"}
    if proc.returncode != 0:
        return {"_error": f"codex exit {proc.returncode}: {proc.stderr.strip()[:200]}"}
    text = proc.stdout.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {"_error": "no JSON in codex output"}
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        return {"_error": f"codex output not valid JSON: {e}"}


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def extract_configurations(
    specs: dict, template: dict, allow_codex: bool, codex_model: str
) -> tuple[list[dict], list[str], str, bool, str | None]:
    """Returns (configurations, warnings, method, codex_used, codex_error)."""
    template_params = template.get("parameters", []) if template else []
    tables = (specs or {}).get("tables") or []
    warnings: list[str] = []

    configs: list[dict] = []
    tables_with_pn = 0
    PER_TABLE_DIAG.clear()
    for tbl in tables:
        if not isinstance(tbl, dict):
            continue
        rows = tbl.get("rows") or []
        if not rows or not isinstance(rows[0], list):
            continue
        # Check if this looks like a part-number table
        pn_col = None
        for i, h in enumerate(rows[0]):
            if isinstance(h, str) and is_part_number_header(h):
                pn_col = i
                break
        if pn_col is None:
            continue
        tables_with_pn += 1
        extracted, reason = parse_table_deterministic(tbl, template_params)
        PER_TABLE_DIAG.append({
            "table_idx": tbl.get("idx"),
            "rows_in_table": len(tbl.get("rows") or []),
            "extracted": len(extracted) if extracted else 0,
            "reason": reason,
        })
        if extracted:
            configs.extend(extracted)

    method = "deterministic"
    codex_used = False
    codex_error: str | None = None
    if not configs and tables and allow_codex:
        # Try codex
        codex_used = True
        method = "codex_assisted"
        specs_dump = json.dumps(tables)
        result = call_codex_parse(specs_dump, template_params, codex_model)
        if isinstance(result, dict) and "_error" not in result:
            codex_configs = result.get("configurations", [])
            if isinstance(codex_configs, list):
                for c in codex_configs:
                    if isinstance(c, dict) and c.get("part_number"):
                        configs.append({
                            "part_number": str(c["part_number"]).strip(),
                            "variant": c.get("variant", "standard") or "standard",
                            "values": c.get("values", {}) or {},
                        })
            for w in (result.get("warnings") or []):
                warnings.append(f"codex: {w}")
        else:
            codex_error = (result or {}).get("_error", "codex returned no result")
            warnings.append(f"codex fallback failed: {codex_error}")

    if not configs:
        method = "empty" if not tables else "single_row_default"
        # Emit a single-row default from template defaults
        values: dict[str, Any] = {}
        for p in template_params:
            if p.get("default_value") is not None and p.get("category") == "geometry":
                values[p["name"]] = p["default_value"]
        if values or template_params:
            pn = "DEFAULT"
            if template:
                # Try to derive a part number from template_id
                pn = f"DEFAULT-{template.get('template_id', 'part')[:24]}"
            configs.append({
                "part_number": pn,
                "variant": "default",
                "values": values,
                "extras": {},
                "source_table_idx": None,
                "source_row_idx": None,
            })
        warnings.append("no part-number rows found; emitted single-row default from template parameters")

    return configs, warnings, method, codex_used, codex_error


def process_one(component_id: str, templates_dir: Path, downloads_dir: Path,
                output_dir: Path, schema: dict,
                allow_codex: bool, codex_model: str) -> ComponentResult:
    t0 = time.time()
    template, template_fname = load_template(component_id, templates_dir)
    if template is None:
        return ComponentResult(
            component_id=component_id, template_id=None, catalog_id=None,
            method="skipped", n_configurations=0, parameter_symbols=[],
            warnings=[f"no feature template found in {templates_dir}"],
        )

    component_dir = downloads_dir / component_id
    if not component_dir.exists():
        return ComponentResult(
            component_id=component_id, template_id=template.get("template_id"),
            catalog_id=template.get("template_id"),
            method="skipped", n_configurations=0, parameter_symbols=[],
            warnings=[f"downloads component dir not found: {component_dir}"],
        )

    specs, specs_path = load_specs(component_dir)
    if specs is None:
        return ComponentResult(
            component_id=component_id, template_id=template.get("template_id"),
            catalog_id=template.get("template_id"),
            method="empty", n_configurations=0, parameter_symbols=[],
            warnings=[f"no specs file found ({specs_path or 'none'})"],
        )

    configs, warnings, method, codex_used, codex_error = extract_configurations(
        specs, template, allow_codex, codex_model)

    # Build output
    catalog_id = template.get("template_id", component_id)
    parameter_symbols = sorted({k for c in configs for k in (c.get("values") or {}).keys()})

    out = {
        "schema_version": "1.0",
        "catalog_id": catalog_id,
        "source_template_id": template.get("template_id"),
        "source_component_id": component_id,
        "units": template.get("units", "mm"),
        "parameter_symbols": parameter_symbols,
        "configurations": configs,
        "extraction": {
            "method": method,
            "tables_scanned": len(specs.get("tables") or []),
            "tables_with_part_numbers": sum(
                1 for t in (specs.get("tables") or [])
                if isinstance(t, dict) and t.get("rows") and isinstance(t["rows"][0], list)
                and any(isinstance(h, str) and is_part_number_header(h) for h in t["rows"][0])
            ),
            "rows_emitted": len(configs),
            "per_table": PER_TABLE_DIAG,
            "warnings": warnings,
        },
    }
    # Drop the per-row "extras" field to match schema (schema only has the documented fields)
    for c in out["configurations"]:
        c.pop("extras", None)

    # Validate against schema
    try:
        jsonschema.validate(out, schema)
        schema_valid = True
    except jsonschema.ValidationError as e:
        schema_valid = False
        warnings.append(f"output failed schema validation: {e.message}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{component_id}.configurations.json"
    if schema_valid:
        output_path.write_text(json.dumps(out, indent=2))

    return ComponentResult(
        component_id=component_id,
        template_id=template.get("template_id"),
        catalog_id=catalog_id,
        method=method,
        n_configurations=len(configs),
        parameter_symbols=parameter_symbols,
        warnings=warnings,
        codex_used=codex_used,
        codex_error=codex_error,
        output_path=str(output_path) if schema_valid else None,
        duration_seconds=round(time.time() - t0, 3),
    )


def discover_components(templates_dir: Path, limit: int | None,
                        component: str | None) -> list[str]:
    if component:
        return [component] if (templates_dir / f"{component}.feature_template.json").exists() else []
    files = sorted(templates_dir.glob("*.feature_template.json"))
    if limit is not None:
        files = files[:limit]
    return [f.name.split(".")[0] for f in files]


def print_summary(results: list[ComponentResult], as_json: bool) -> None:
    if as_json:
        summary = {
            "n": len(results),
            "n_deterministic": sum(1 for r in results if r.method == "deterministic"),
            "n_codex_assisted": sum(1 for r in results if r.method == "codex_assisted"),
            "n_empty_or_default": sum(1 for r in results if r.method in {"empty", "single_row_default"}),
            "n_codex_failed": sum(1 for r in results if r.codex_error),
            "total_configurations": sum(r.n_configurations for r in results),
            "results": [r.to_dict() for r in results],
        }
        print(json.dumps(summary, indent=2))
        return

    print(f"\nStep 3 normalize: {len(results)} components")
    print(f"  deterministic:    {sum(1 for r in results if r.method == 'deterministic')}")
    print(f"  codex_assisted:   {sum(1 for r in results if r.method == 'codex_assisted')}")
    print(f"  empty/default:    {sum(1 for r in results if r.method in {'empty', 'single_row_default'})}")
    print(f"  total configs:    {sum(r.n_configurations for r in results)}")
    if any(r.codex_error for r in results):
        print(f"  codex failures:   {sum(1 for r in results if r.codex_error)}")
    print()
    header = f"{'component_id':<16} {'method':<18} {'n_cfg':>5}  template_id"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r.component_id:<16} {r.method:<18} {r.n_configurations:>5}  {r.template_id or '-'}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--templates-dir", type=Path, default=DEFAULT_TEMPLATES_DIR)
    ap.add_argument("--downloads-dir", type=Path, default=DEFAULT_DOWNLOADS_DIR)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--component", type=str, default=None)
    ap.add_argument("--allow-codex", action="store_true",
                    help="If set and deterministic extraction finds no rows, fall back to one codex call")
    ap.add_argument("--codex-model", type=str, default=DEFAULT_CODEX_MODEL)
    ap.add_argument("--verbose", action="store_true",
                    help="Print per-table diagnostic info (table_idx, rows, extracted, reason)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if not args.schema.exists():
        print(f"ERROR: schema not found: {args.schema}", file=sys.stderr)
        return 2
    schema = json.loads(args.schema.read_text())

    component_ids = discover_components(args.templates_dir, args.limit, args.component)
    if not component_ids:
        print(f"ERROR: no components to process (templates dir empty?)", file=sys.stderr)
        return 2

    results: list[ComponentResult] = []
    for cid in component_ids:
        r = process_one(cid, args.templates_dir, args.downloads_dir,
                        args.output_dir, schema, args.allow_codex, args.codex_model)
        results.append(r)
        if not args.json:
            cw = f" codex={'Y' if r.codex_used else ('F' if r.codex_error else '-')}"
            print(f"  [{r.method:<18}] {cid}  n={r.n_configurations}{cw}")
            if args.verbose and r.output_path:
                cfg = json.loads(Path(r.output_path).read_text())
                for t in cfg.get("extraction", {}).get("per_table", []):
                    print(f"      table[{t['table_idx']}] rows={t['rows_in_table']} "
                          f"extracted={t['extracted']} reason={t['reason']}")
                for c in cfg.get("configurations", []):
                    vals = ", ".join(f"{k}={v}" for k, v in list(c.get("values", {}).items())[:6])
                    print(f"      -> {c['part_number']:<32} {vals}")
    print_summary(results, args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
