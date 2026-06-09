"""step0_6_table_merge — Phase B/C/D of the Step 0.6 pipeline.

Consumes ``output/families/<fid>/family.json`` produced by step 0.5,
classifies the per-member feature_template tables, merges them with
``scripts.table_merge.merge.merge_tables``, optionally calls the LLM
synonym resolver and conflict resolver, and writes
``output/families/<fid>/merged_tables.json`` plus a human-readable
``merge_report.md``.

Usage:
    PYTHONPATH=. .venv-sim/bin/python scripts/step0_6_table_merge.py --dry-run
    PYTHONPATH=. .venv-sim/bin/python scripts/step0_6_table_merge.py --family-id fam_X

Exits 0 on success (including the no-op case of an empty
``--families-dir``); warnings go to stderr, status to stdout.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from scripts.table_merge.classify import (
    DEFAULT_SYNONYMS,
    TableType,
    resolve_headers,
)
from scripts.table_merge.merge import merge_tables

SCHEMA_VERSION = "1.0"


# --- Section mapping ---------------------------------------------------------
# Per-section (columns, hard-coded table_type, table_idx).
SECTION_SCHEMA: dict[str, tuple[tuple[str, ...], str, int]] = {
    "parameters":                       (("name", "symbol", "description",
                                         "parameter_type", "category",
                                         "default_value", "units", "tolerance",
                                         "notes"),
                                        TableType.VARIANT_LOOKUP.value, 0),
    "components":                       (("id", "name", "component_type",
                                         "parent_component"),
                                        TableType.DIMENSION_TABLE.value, 1),
    "feature_graph":                    (("id", "feature_type", "operation",
                                         "modeling_primitive", "depends_on"),
                                        TableType.CONFIGURABLE_OPTIONS.value, 2),
    "catalog_material_options":         (("value",),
                                        TableType.CONFIGURABLE_OPTIONS.value, 3),
    "catalog_surface_treatment_options":(("value",),
                                        TableType.CONFIGURABLE_OPTIONS.value, 4),
    "catalog_cleaning_options":         (("value",),
                                        TableType.CONFIGURABLE_OPTIONS.value, 5),
    "catalog_packaging_options":        (("value",),
                                        TableType.CONFIGURABLE_OPTIONS.value, 6),
}
# catalog_metadata.<key> -> our section name
CATALOG_OPTION_KEYS = {
    "material_options": "catalog_material_options",
    "surface_treatment_options": "catalog_surface_treatment_options",
    "cleaning_options": "catalog_cleaning_options",
    "packaging_options": "catalog_packaging_options",
}


# --- CLI ---------------------------------------------------------------------


def main() -> int:
    args = _parse_args()
    return _run(
        families_dir=args.families_dir,
        templates_dir=args.templates_dir,
        out_dir=args.out_dir,
        state_db=args.state_db,
        dry_run=bool(args.dry_run or args.skip_llm),
        family_id=args.family_id,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="step0_6_table_merge",
        description="Merge per-member feature_template tables for each family.",
    )
    p.add_argument("--families-dir", type=Path, default=Path("output/families"))
    p.add_argument("--templates-dir", type=Path, default=Path("output/feature_templates"))
    p.add_argument("--out-dir", type=Path, default=Path("output/families"))
    p.add_argument("--state-db", type=Path, default=Path("data/workflow_state.sqlite3"))
    p.add_argument("--dry-run", action="store_true",
                   help="Deterministic path only; never call the LLM.")
    p.add_argument("--skip-llm", action="store_true", help="Alias for --dry-run.")
    p.add_argument("--family-id", type=str, default=None,
                   help="Process only the named family (debug).")
    p.add_argument("--require-human-approval", action="store_true",
                   help="Reserved for v2; v1 is a no-op.")
    return p.parse_args()


# --- Top-level run -----------------------------------------------------------


def _run(
    *, families_dir: Path, templates_dir: Path, out_dir: Path,
    state_db: Path, dry_run: bool, family_id: str | None,
) -> int:
    if not families_dir.exists():
        print(f"no families directory at {families_dir}; nothing to do.",
              file=sys.stderr)
        return 0
    index_path = families_dir / "index.json"
    if not index_path.exists():
        print(f"no index.json at {index_path}; nothing to do.", file=sys.stderr)
        return 0
    try:
        index = json.loads(index_path.read_text())
    except json.JSONDecodeError as e:
        print(f"failed to parse {index_path}: {e}", file=sys.stderr)
        return 1
    families = index.get("families") or []
    if family_id:
        families = [f for f in families if f.get("family_id") == family_id]
        if not families:
            print(f"no family with id {family_id!r} in index.json", file=sys.stderr)
            return 0
    if not families:
        print("no families to process; nothing to do.")
        return 0
    n_merged_total = 0
    for fam_entry in families:
        try:
            n_merged_total += _process_family(
                fam_entry, families_dir, templates_dir, out_dir,
                state_db, dry_run,
            )
        except Exception as e:  # noqa: BLE001 — log and continue
            print(f"ERROR processing family {fam_entry.get('family_id')}: {e}",
                  file=sys.stderr)
    print(f"merged tables written for {len(families)} families; "
          f"{n_merged_total} sections total."
          + (" (dry-run; no LLM calls)" if dry_run else ""))
    return 0


def _process_family(
    fam_entry: dict, families_dir: Path, templates_dir: Path,
    out_dir: Path, state_db: Path, dry_run: bool,
) -> int:
    fid = fam_entry.get("family_id") or "<unknown>"
    family_doc = _load_family_doc(fid, families_dir)
    if family_doc is None:
        return 0
    member_ids = (family_doc.get("member_component_ids")
                  or fam_entry.get("members")
                  or fam_entry.get("member_component_ids") or [])
    category_root = family_doc.get("category_root") or fam_entry.get("category_root") or ""

    # 1) Load each member's template; collect per-section tables.
    section_tables: dict[str, list[dict]] = {k: [] for k in SECTION_SCHEMA}
    present: list[str] = []
    warnings: list[str] = []
    for cid in member_ids:
        path = templates_dir / f"{cid}.feature_template.json"
        if not path.exists():
            msg = f"  {fid}: member {cid} template missing at {path}; skipping."
            print(msg, file=sys.stderr)
            warnings.append(msg.strip())
            continue
        try:
            template = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            msg = f"  {fid}: member {cid} template unparseable: {e}; skipping."
            print(msg, file=sys.stderr)
            warnings.append(msg.strip())
            continue
        present.append(cid)
        _extract_section_tables(template, cid, section_tables)

    if not present:
        print(f"  {fid}: no templates found for any of {member_ids}; skipping.",
              file=sys.stderr)
        return 0

    # 2-4) Per section: header resolution, classification, merge, LLM (skipped in dry-run).
    sections_out: dict[str, dict] = {}
    syn_calls = con_calls = 0
    t0 = time.time()
    for section_name, tables in section_tables.items():
        if not tables:
            continue
        doc, s, c = _merge_section(section_name, tables, fid, category_root, dry_run)
        sections_out[section_name] = doc
        syn_calls += s
        con_calls += c
    elapsed = time.time() - t0

    # 5) Write merged_tables.json
    out_path = out_dir / fid / "merged_tables.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "family_id": fid,
        "category_root": category_root,
        "n_members": len(present),
        "sections": sections_out,
    }, indent=2, default=str))

    # 6) Write merge_report.md
    report_path = out_dir / fid / "merge_report.md"
    report_path.write_text(_render_report(
        fid, category_root, len(member_ids), len(present),
        sections_out, syn_calls, con_calls, elapsed, warnings, dry_run,
    ))

    # 7) State-DB best-effort (do NOT create if missing)
    _touch_state_db(state_db, fid, present)
    return len(sections_out)


def _load_family_doc(fid: str, families_dir: Path) -> dict | None:
    p = families_dir / fid / "family.json"
    if not p.exists():
        print(f"  {fid}: missing family.json at {p}; skipping.", file=sys.stderr)
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError as e:
        print(f"  {fid}: failed to parse family.json: {e}", file=sys.stderr)
        return None


# --- Section extraction + merge ---------------------------------------------


def _extract_section_tables(template: dict, cid: str, out: dict[str, list[dict]]) -> None:
    """Convert one feature_template into per-section tables, appended to out."""
    sources = {
        "parameters": template.get("parameters") or [],
        "components": template.get("components") or [],
        "feature_graph": template.get("feature_graph") or [],
    }
    catalog = template.get("catalog_metadata") or {}
    for section_name, (columns, ttype, tidx) in SECTION_SCHEMA.items():
        if section_name in sources:
            for row in sources[section_name]:
                if not isinstance(row, dict):
                    continue
                tbl = {
                    "component_id": cid, "table_idx": tidx,
                    "table_type": ttype, "headers": list(columns),
                    "rows": [[row.get(c) for c in columns]],
                }
                if section_name == "parameters":
                    tbl["primary_key"] = columns[0]
                out[section_name].append(tbl)
        else:  # catalog_*_options
            src_key = next((k for k, v in CATALOG_OPTION_KEYS.items()
                            if v == section_name), None)
            if not src_key:
                continue
            values = catalog.get(src_key) or []
            rows = [[v] for v in values if v is not None and v != ""]
            if rows:
                out[section_name].append({
                    "component_id": cid, "table_idx": tidx,
                    "table_type": ttype, "headers": list(columns), "rows": rows,
                })


def _merge_section(
    section_name: str, tables: list[dict],
    family_id: str, category_root: str, dry_run: bool,
) -> tuple[dict, int, int]:
    """Merge all tables in one section; return (section_doc, n_syn_calls, n_con_calls)."""
    # a) Collect all headers; build local header->canonical via DEFAULT_SYNONYMS.
    seen: set[str] = set()
    all_headers: list[str] = []
    for t in tables:
        for h in t.get("headers") or []:
            if h not in seen:
                seen.add(h)
                all_headers.append(h)
    local_synonyms = resolve_headers(all_headers, DEFAULT_SYNONYMS)

    # b) Optional LLM expansion (skipped in dry-run per spec).
    syn_calls = 0
    if not dry_run:
        known = set(DEFAULT_SYNONYMS.keys())
        unknowns = [h for h in all_headers
                    if local_synonyms.get(h) == h and h not in known]
        if unknowns:
            from scripts.table_merge.synonym_llm import resolve_synonyms
            llm_map = resolve_synonyms(unknowns, family_id, category_root,
                                       openai_client=None)
            for k, v in llm_map.items():
                local_synonyms.setdefault(k, v)
            syn_calls += 1

    # c) Use the section's hard-coded type (per spec).
    _, section_type, _ = SECTION_SCHEMA[section_name]

    # d) Deterministic merge.
    merged = merge_tables(tables, synonyms=local_synonyms)

    # e) Optional LLM conflict resolution (skipped in dry-run per spec).
    con_calls = 0
    resolutions = []
    if not dry_run and merged.conflicts:
        from scripts.table_merge.conflict_llm import resolve_conflicts_sync
        resolutions = resolve_conflicts_sync(merged.conflicts, openai_client=None)
        con_calls += 1

    return {
        "table_type": merged.table_type.value,
        "headers": merged.canonical_headers,
        "rows": merged.rows,
        "n_conflicts": len(merged.conflicts),
        "conflicts": [c.to_dict() for c in merged.conflicts],
        "resolutions": [r.to_dict() for r in resolutions],
        "source_component_ids": merged.source_component_ids,
        "source_table_ids": merged.source_table_ids,
        "notes": merged.notes,
    }, syn_calls, con_calls


# --- Report rendering --------------------------------------------------------


def _render_report(
    fid: str, category_root: str, n_declared: int, n_present: int,
    sections: dict[str, dict], syn_calls: int, con_calls: int,
    elapsed: float, warnings: list[str], dry_run: bool,
) -> str:
    total_conflicts = sum(s.get("n_conflicts", 0) for s in sections.values())
    out = [
        f"# Table merge report — `{fid}`\n",
        f"- category_root: `{category_root}`",
        f"- members declared in family.json: {n_declared}",
        f"- members with templates present:  {n_present}",
        f"- sections processed: {len(sections)}",
        f"- LLM synonym calls: {syn_calls}",
        f"- LLM conflict calls: {con_calls}",
        f"- elapsed: {elapsed:.3f}s",
        f"- mode: {'dry-run (no LLM)' if dry_run else 'live (LLM allowed)'}\n",
        f"**Conflicts: {total_conflicts}**\n",
    ]
    if sections:
        out.append("## Sections\n")
        for name, s in sections.items():
            out.append(
                f"### `{name}` (type: `{s.get('table_type')}`)  —  "
                f"inputs: {len(s.get('source_table_ids') or [])}, "
                f"output rows: {len(s.get('rows') or [])}, "
                f"conflicts: {s.get('n_conflicts', 0)}"
            )
            if s.get("headers"):
                out.append("- headers: "
                           + ", ".join(f"`{h}`" for h in s["headers"]))
            for note in s.get("notes") or []:
                out.append(f"- note: {note}")
            out.append("")
    if warnings:
        out.append("## Warnings\n")
        out.extend(f"- {w}" for w in warnings)
        out.append("")
    return "\n".join(out)


# --- State-DB nudge ----------------------------------------------------------


def _touch_state_db(state_db: Path, fid: str, members: list[str]) -> None:
    """Best-effort: if state DB exists, add family_id col + update rows.
    Per spec, do NOT create the DB if missing — just skip."""
    if not state_db.exists():
        return
    try:
        import sqlite3
        with sqlite3.connect(state_db) as con:
            cols = {r[1] for r in con.execute("PRAGMA table_info(components)").fetchall()}
            if "family_id" not in cols:
                con.execute("ALTER TABLE components ADD COLUMN family_id TEXT")
            now = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
            for cid in members:
                try:
                    con.execute(
                        "UPDATE components SET family_id=?, updated_at=? "
                        "WHERE component_id=?",
                        (fid, now, cid),
                    )
                except sqlite3.OperationalError:
                    con.execute(
                        "UPDATE components SET family_id=? WHERE component_id=?",
                        (fid, cid),
                    )
            con.commit()
    except Exception as e:  # noqa: BLE001 — never let state-DB issues kill the merge
        print(f"  {fid}: state-db update failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
