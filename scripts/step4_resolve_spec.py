#!/usr/bin/env python3
"""
Step 4 — Resolve a Selected Part Number into a Concrete CAD Spec.

Reads:
  - the Step 1 feature template (or a Step 2-repaired copy if present)
  - the Step 3 normalized configurations JSON
  - a selected part number (CLI arg, or iterate all if --all)

Writes a single resolved_cad_spec.json per part number, where every
feature_graph parameter `value` is a concrete number/string (no $D-style
placeholders remaining). The spec validates against
schemas/resolved_cad_spec.schema.json.

This step is PURE DETERMINISTIC. No LLM, no network. Just dict walking.

Usage:
  python scripts/step4_resolve_spec.py --component 110300324920
  python scripts/step4_resolve_spec.py --component 110300324920 --part-number MCSCS10
  python scripts/step4_resolve_spec.py --all
  python scripts/step4_resolve_spec.py --all --limit 3
  python scripts/step4_resolve_spec.py --json
"""
from __future__ import annotations

import argparse
import json
import re
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
DEFAULT_REPAIRED_DIR = REPO_ROOT / "output" / "feature_templates_repaired"
DEFAULT_CONFIGS_DIR = REPO_ROOT / "output" / "normalized_configs"
DEFAULT_REVIEWS_DIR = REPO_ROOT / "output" / "feature_templates_reviewed"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "resolved_specs"
DEFAULT_SCHEMA = REPO_ROOT / "schemas" / "resolved_cad_spec.schema.json"

# $VAR and ${VAR} placeholders we expect to find
_PLACEHOLDER_RE = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?")


# ----------------------------------------------------------------------------
# Data model
# ----------------------------------------------------------------------------
@dataclass
class ResolveResult:
    component_id: str
    catalog_id: str | None
    part_number: str
    variant: str | None
    n_resolved_features: int
    n_unresolved: int
    unresolved_references: list[dict]
    review_flags_inherited: list[str]
    schema_valid: bool
    output_path: str | None
    warnings: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------------
def load_template_with_repair_precedence(component_id: str,
                                         templates_dir: Path,
                                         repaired_dir: Path) -> tuple[dict | None, str]:
    repaired = repaired_dir / f"{component_id}.feature_template.json"
    if repaired.exists():
        return json.loads(repaired.read_text()), "repaired"
    p = templates_dir / f"{component_id}.feature_template.json"
    if p.exists():
        return json.loads(p.read_text()), "original"
    return None, "missing"


def load_configs(component_id: str, configs_dir: Path) -> dict | None:
    p = configs_dir / f"{component_id}.configurations.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def load_review(component_id: str, reviews_dir: Path) -> dict | None:
    p = reviews_dir / f"{component_id}.review.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


# ----------------------------------------------------------------------------
# Resolution logic
# ----------------------------------------------------------------------------
def build_value_map(template: dict, config_row: dict) -> dict[str, Any]:
    """Build a flat name->value map from:
      1. The configuration row's `values` (by template parameter name)
      2. The template's top-level `parameters[].default_value` (fallback)
    """
    values: dict[str, Any] = {}

    # 1. Configuration row wins
    for k, v in (config_row.get("values") or {}).items():
        values[k] = v

    # 2. Template defaults fill in the rest
    for p in template.get("parameters", []) or []:
        name = p.get("name")
        if not name:
            continue
        if name in values:
            continue
        if p.get("default_value") is not None:
            values[name] = p["default_value"]

    return values


def resolve_placeholder_string(s: str, value_map: dict[str, Any]) -> tuple[Any, bool]:
    """If the whole string is a single $VAR placeholder, return the
    substituted value. Otherwise leave the string alone but log it."""
    if not isinstance(s, str):
        return s, True
    m = _PLACEHOLDER_RE.fullmatch(s.strip())
    if m and m.group(1) in value_map:
        return value_map[m.group(1)], True
    return s, False


def resolve_feature(feat: dict, value_map: dict[str, Any]) -> tuple[dict, list[dict]]:
    """Return (resolved_feature_dict, unresolved_references)."""
    unresolved: list[dict] = []
    fid = feat.get("id", "<no id>")
    new_feat = json.loads(json.dumps(feat))  # deep copy

    # Resolve each parameter
    new_params = []
    for p in new_feat.get("parameters", []) or []:
        name = p.get("name")
        raw = p.get("value")
        resolved = raw
        source = "carried_over"

        if name in value_map and value_map[name] is not None:
            resolved = value_map[name]
            source = "configuration_row"
        elif isinstance(raw, str):
            sub, ok = resolve_placeholder_string(raw, value_map)
            if ok and sub != raw:
                resolved = sub
                source = "template_default"
            elif _PLACEHOLDER_RE.search(str(raw)):
                # Has a $VAR we couldn't resolve
                unresolved.append({
                    "feature_id": fid, "parameter_name": name, "raw_value": raw,
                })
                source = "missing"
        elif raw is None and name in value_map:
            resolved = value_map[name]
            source = "configuration_row"
        elif raw is None:
            # Try synonynms: feature may reference a parameter that maps to a
            # different name in the template (e.g. "OD" -> "D1")
            for syn_key, syn_val in value_map.items():
                if name and syn_key and (name.lower() == syn_key.lower()):
                    resolved = syn_val
                    source = "configuration_row"
                    break

        new_params.append({
            "name": name,
            "value": resolved,
            "role": p.get("role", "driving_dimension"),
            "source": source,
        })
    new_feat["parameters"] = new_params

    # Resolve construction.depth if it's a string
    constr = new_feat.get("construction")
    if isinstance(constr, dict) and "depth" in constr:
        d = constr["depth"]
        if isinstance(d, str) and d in value_map:
            constr["depth"] = value_map[d]

    # Resolve position.x/y/z if they're strings
    pos = new_feat.get("position")
    if isinstance(pos, dict):
        for axis in ("x", "y", "z"):
            v = pos.get(axis)
            if isinstance(v, str) and v in value_map:
                pos[axis] = value_map[v]

    return new_feat, unresolved


def inherit_review_flags(template: dict, review: dict | None) -> list[str]:
    flags: list[str] = []
    for feat in template.get("feature_graph", []) or []:
        if feat.get("needs_review"):
            flags.append(f"{feat.get('id')}: {feat.get('review_reason') or '<no reason>'}")
    if review:
        for issue in review.get("issues", []) or []:
            if issue.get("level") in ("error", "warning"):
                flags.append(f"step2[{issue.get('code')}]: {issue.get('message')}")
    return flags


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def resolve_one(component_id: str, part_number: str | None,
                templates_dir: Path, repaired_dir: Path,
                configs_dir: Path, reviews_dir: Path,
                output_dir: Path, schema: dict) -> ResolveResult:
    t0 = time.time()
    template, source = load_template_with_repair_precedence(
        component_id, templates_dir, repaired_dir)
    if template is None:
        return ResolveResult(
            component_id=component_id, catalog_id=None, part_number=part_number or "",
            variant=None, n_resolved_features=0, n_unresolved=0,
            unresolved_references=[], review_flags_inherited=[],
            schema_valid=False, output_path=None,
            warnings=[f"no template found (repaired or original) for {component_id}"],
        )

    configs = load_configs(component_id, configs_dir)
    if configs is None:
        return ResolveResult(
            component_id=component_id, catalog_id=template.get("template_id"),
            part_number=part_number or "", variant=None,
            n_resolved_features=0, n_unresolved=0,
            unresolved_references=[], review_flags_inherited=[],
            schema_valid=False, output_path=None,
            warnings=[f"no normalized configurations for {component_id} (run step3 first)"],
        )

    config_rows = configs.get("configurations") or []
    if not config_rows:
        return ResolveResult(
            component_id=component_id, catalog_id=template.get("template_id"),
            part_number=part_number or "", variant=None,
            n_resolved_features=0, n_unresolved=0,
            unresolved_references=[], review_flags_inherited=[],
            schema_valid=False, output_path=None,
            warnings=[f"step3 emitted 0 configurations for {component_id}"],
        )

    # Pick the row
    if part_number:
        matching = [c for c in config_rows if c.get("part_number") == part_number]
        if not matching:
            return ResolveResult(
                component_id=component_id, catalog_id=template.get("template_id"),
                part_number=part_number, variant=None,
                n_resolved_features=0, n_unresolved=0,
                unresolved_references=[], review_flags_inherited=[],
                schema_valid=False, output_path=None,
                warnings=[f"part_number {part_number!r} not found in step3 configurations; "
                          f"available: {[c.get('part_number') for c in config_rows[:8]]}"],
            )
        row = matching[0]
    else:
        row = config_rows[0]  # default to first

    value_map = build_value_map(template, row)

    resolved_features: list[dict] = []
    all_unresolved: list[dict] = []
    for feat in template.get("feature_graph", []) or []:
        rf, unres = resolve_feature(feat, value_map)
        resolved_features.append(rf)
        all_unresolved.extend(unres)

    review = load_review(component_id, reviews_dir)
    review_flags = inherit_review_flags(template, review)

    # Build the spec
    catalog_id = template.get("template_id", component_id)
    spec = {
        "schema_version": "1.0",
        "template_id": template.get("template_id"),
        "catalog_id": catalog_id,
        "part_number": row.get("part_number"),
        "variant": row.get("variant", "standard"),
        "units": template.get("units", "mm"),
        "parameter_bindings": value_map,
        "unresolved_references": all_unresolved,
        "coordinate_system": template.get("coordinate_system", {}),
        "components": template.get("components", []),
        "resolved_features": resolved_features,
        "review_flags": review_flags,
    }

    # Validate
    try:
        jsonschema.validate(spec, schema)
        schema_valid = True
    except jsonschema.ValidationError as e:
        schema_valid = False
        return ResolveResult(
            component_id=component_id, catalog_id=catalog_id,
            part_number=row.get("part_number", ""), variant=row.get("variant"),
            n_resolved_features=len(resolved_features),
            n_unresolved=len(all_unresolved),
            unresolved_references=all_unresolved,
            review_flags_inherited=review_flags,
            schema_valid=False, output_path=None,
            warnings=[f"schema validation failed: {e.message} at {list(e.absolute_path)[:3]}"],
            duration_seconds=round(time.time() - t0, 3),
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_pn = re.sub(r"[^A-Za-z0-9._-]", "_", row.get("part_number", "unknown"))
    output_path = output_dir / f"{component_id}__{safe_pn}.resolved_cad_spec.json"
    output_path.write_text(json.dumps(spec, indent=2))

    return ResolveResult(
        component_id=component_id, catalog_id=catalog_id,
        part_number=row.get("part_number", ""), variant=row.get("variant"),
        n_resolved_features=len(resolved_features),
        n_unresolved=len(all_unresolved),
        unresolved_references=all_unresolved,
        review_flags_inherited=review_flags,
        schema_valid=True, output_path=str(output_path),
        duration_seconds=round(time.time() - t0, 3),
    )


def discover_components(templates_dir: Path, limit: int | None,
                        component: str | None) -> list[str]:
    if component:
        return [component]
    files = sorted(templates_dir.glob("*.feature_template.json"))
    if limit is not None:
        files = files[:limit]
    return [f.name.split(".")[0] for f in files]


def print_summary(results: list[ResolveResult], as_json: bool) -> None:
    if as_json:
        summary = {
            "n": len(results),
            "n_schema_valid": sum(1 for r in results if r.schema_valid),
            "n_with_unresolved": sum(1 for r in results if r.n_unresolved > 0),
            "n_with_review_flags": sum(1 for r in results if r.review_flags_inherited),
            "results": [r.to_dict() for r in results],
        }
        print(json.dumps(summary, indent=2))
        return

    print(f"\nStep 4 resolve: {len(results)} specs")
    print(f"  schema valid:        {sum(1 for r in results if r.schema_valid)}/{len(results)}")
    print(f"  with unresolved refs: {sum(1 for r in results if r.n_unresolved > 0)}")
    print(f"  with review flags:   {sum(1 for r in results if r.review_flags_inherited)}")
    print()
    header = f"{'component_id':<16} {'part_number':<24} {'features':>8} {'unres':>6} {'valid':>6}  catalog_id"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r.component_id:<16} {r.part_number[:24]:<24} {r.n_resolved_features:>8} "
              f"{r.n_unresolved:>6} {str(r.schema_valid):>6}  {r.catalog_id or '-'}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--templates-dir", type=Path, default=DEFAULT_TEMPLATES_DIR)
    ap.add_argument("--repaired-dir", type=Path, default=DEFAULT_REPAIRED_DIR,
                    help="Step 2 repaired templates take precedence over originals")
    ap.add_argument("--configs-dir", type=Path, default=DEFAULT_CONFIGS_DIR)
    ap.add_argument("--reviews-dir", type=Path, default=DEFAULT_REVIEWS_DIR)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    ap.add_argument("--component", type=str, default=None)
    ap.add_argument("--part-number", type=str, default=None,
                    help="Specific part number; if omitted, uses the first row from step3")
    ap.add_argument("--all", action="store_true",
                    help="Resolve every part number for the chosen component(s)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if not args.schema.exists():
        print(f"ERROR: schema not found: {args.schema}", file=sys.stderr)
        return 2
    schema = json.loads(args.schema.read_text())

    component_ids = discover_components(args.templates_dir, args.limit, args.component)
    if not component_ids:
        print(f"ERROR: no components", file=sys.stderr)
        return 2

    results: list[ResolveResult] = []
    for cid in component_ids:
        if args.all and not args.part_number:
            # Resolve every row
            cfgs = load_configs(cid, args.configs_dir)
            if not cfgs:
                results.append(ResolveResult(
                    component_id=cid, catalog_id=None, part_number="",
                    variant=None, n_resolved_features=0, n_unresolved=0,
                    unresolved_references=[], review_flags_inherited=[],
                    schema_valid=False, output_path=None,
                    warnings=[f"no configs for {cid}"]))
                continue
            for row in cfgs.get("configurations", []):
                r = resolve_one(cid, row.get("part_number"),
                                args.templates_dir, args.repaired_dir,
                                args.configs_dir, args.reviews_dir,
                                args.output_dir, schema)
                results.append(r)
                if not args.json:
                    print(f"  [OK ] {cid}::{r.part_number}  features={r.n_resolved_features}  unres={r.n_unresolved}")
        else:
            r = resolve_one(cid, args.part_number,
                            args.templates_dir, args.repaired_dir,
                            args.configs_dir, args.reviews_dir,
                            args.output_dir, schema)
            results.append(r)
            if not args.json:
                print(f"  [{'OK ' if r.schema_valid else 'ERR'}] {cid}::{r.part_number}  features={r.n_resolved_features}  unres={r.n_unresolved}")

    print_summary(results, args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
