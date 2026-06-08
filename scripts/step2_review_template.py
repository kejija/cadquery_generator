#!/usr/bin/env python3
"""
Step 2 — Review / Score / Repair the Feature Template.

Reads every <component_id>.feature_template.json in --templates-dir (default
output/feature_templates/), validates it against the structural schema, runs
a deterministic quality-gate scorer, performs a small set of mechanical
repairs, and (optionally) routes weak templates to a Codex repair call.

Outputs (per template):
  output/feature_templates_reviewed/<component_id>.review.json
    - pass: bool
    - score: float (0..1)
    - issues: [{level, code, message, feature_id?, parameter?}]
    - repairs_applied: [string]
    - repaired_template_path: <path> | null

Usage:
  python scripts/step2_review_template.py
  python scripts/step2_review_template.py --limit 3
  python scripts/step2_review_template.py --component 110300324920
  python scripts/step2_review_template.py --allow-codex-repair --codex-model gpt-5.4-mini
  python scripts/step2_review_template.py --json     # machine-readable summary

By default Codex repair is OFF. Pass --allow-codex-repair to enable. Codex
auth uses your ChatGPT subscription (free for you), default model gpt-5.4-mini.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Iterable

import jsonschema

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEMPLATES_DIR = REPO_ROOT / "output" / "feature_templates"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "feature_templates_reviewed"
DEFAULT_REPAIRED_DIR = REPO_ROOT / "output" / "feature_templates_repaired"
DEFAULT_SCHEMA = REPO_ROOT / "schemas" / "feature_template.schema.json"
DEFAULT_CODEX_MODEL = "gpt-5.4-mini"

# Conventional vocabulary — values outside these get flagged as needs_review.
CONVENTIONAL_UNITS = {"mm", "in"}
CONVENTIONAL_MODELING_MODES = {
    "simplified_single_body", "multi_body_part", "assembly",
}
CONVENTIONAL_FEATURE_TYPES = {
    "base_body", "hole", "pocket", "boss", "slot",
    "chamfer", "fillet", "pattern", "cut", "rib", "draft",
    # Engineering concepts seen in real templates
    "datum", "gdnt", "thread", "step", "neck", "shoulder",
    "taper", "keyway", "groove", "notch", "lug", "flange",
}
CONVENTIONAL_OPERATIONS = {"base", "add", "subtract", "modify", "metadata", "reference"}
CONVENTIONAL_ROLES = {
    "driving_dimension", "reference", "selector", "tolerance",
    "metadata", "validation", "constraint",
}
CONVENTIONAL_CATEGORIES = {
    "geometry", "catalog", "tolerance", "metadata", "material",
}
CONVENTIONAL_SIDES = {
    "left", "right", "top", "bottom", "front", "back", "both",
}
MIN_CONFIDENCE = 0.6
MAX_MISSING_INFORMATION = 10


# ----------------------------------------------------------------------------
# Data model
# ----------------------------------------------------------------------------
@dataclass
class Issue:
    level: str            # "error" | "warning" | "info"
    code: str             # short stable identifier
    message: str
    feature_id: str | None = None
    parameter: str | None = None
    field: str | None = None  # dotted path in the template

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class ReviewResult:
    component_id: str
    template_id: str | None
    part_family: str | None
    schema_valid: bool
    score: float
    passed: bool
    issues: list[dict] = field(default_factory=list)
    repairs_applied: list[str] = field(default_factory=list)
    codex_used: bool = False
    codex_attempted: bool = False
    codex_error: str | None = None
    repaired_template_path: str | None = None
    review_path: str | None = None
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------------
# Quality gate
# ----------------------------------------------------------------------------
def validate_schema(template: dict, schema: dict) -> list[Issue]:
    issues: list[Issue] = []
    validator = jsonschema.Draft7Validator(schema)
    for err in sorted(validator.iter_errors(template), key=lambda e: list(e.absolute_path)):
        issues.append(Issue(
            level="error",
            code="schema_violation",
            message=f"{err.message} at {list(err.absolute_path) or '<root>'}",
            field=".".join(str(p) for p in err.absolute_path) or None,
        ))
    return issues


def score_template(template: dict) -> tuple[float, list[Issue]]:
    """Pure deterministic scoring. No LLM. Mirrors the README example
    and adds the additional checks we agreed on."""
    issues: list[Issue] = []

    # Required top-level keys (jsonschema covers this; we keep a soft check)
    for k in ("template_id", "part_family", "units", "modeling_mode",
              "feature_graph", "components", "parameters", "coordinate_system"):
        if k not in template:
            issues.append(Issue("error", "missing_top_level_key", f"missing required key: {k}", field=k))

    # Units
    units = template.get("units")
    if units not in CONVENTIONAL_UNITS:
        issues.append(Issue("warning", "non_standard_units",
                            f"units={units!r} is not in conventional set",
                            field="units"))

    # Modeling mode
    mm = template.get("modeling_mode")
    if mm not in CONVENTIONAL_MODELING_MODES:
        issues.append(Issue("warning", "non_standard_modeling_mode",
                            f"modeling_mode={mm!r} not in conventional set",
                            field="modeling_mode"))

    # At least one base_body
    fg = template.get("feature_graph", [])
    base_bodies = [f for f in fg if f.get("feature_type") == "base_body"]
    if not base_bodies:
        issues.append(Issue("error", "no_base_body",
                            "feature_graph has no base_body feature"))
    elif len(base_bodies) > 1:
        issues.append(Issue("warning", "multiple_base_bodies",
                            f"feature_graph has {len(base_bodies)} base_body features; "
                            "consider whether that is intentional"))

    # Per-feature checks
    seen_ids: set[str] = set()
    for feat in fg:
        fid = feat.get("id", "<missing id>")
        if fid in seen_ids:
            issues.append(Issue("error", "duplicate_feature_id",
                                f"feature id {fid!r} appears more than once",
                                feature_id=fid))
        seen_ids.add(fid)

        ft = feat.get("feature_type")
        if ft not in CONVENTIONAL_FEATURE_TYPES:
            issues.append(Issue("warning", "non_standard_feature_type",
                                f"feature_type={ft!r} not in conventional set",
                                feature_id=fid, field="feature_type"))

        op = feat.get("operation")
        if op not in CONVENTIONAL_OPERATIONS:
            issues.append(Issue("warning", "non_standard_operation",
                                f"operation={op!r} not in conventional set",
                                feature_id=fid, field="operation"))

        mp = feat.get("modeling_primitive")
        if mp in (None, "", "unknown") and not feat.get("needs_review"):
            issues.append(Issue("warning", "unknown_modeling_primitive",
                                f"modeling_primitive is {mp!r} but needs_review is false",
                                feature_id=fid, field="modeling_primitive"))

        # Hole / pocket / boss axis check
        if ft in {"hole", "pocket", "boss"} and feat.get("axis") is None and not feat.get("needs_review"):
            issues.append(Issue("warning", "missing_axis",
                                f"{ft} feature has axis=null but needs_review=false",
                                feature_id=fid, field="axis"))

        # Confidence
        conf = feat.get("confidence", 1.0)
        if isinstance(conf, (int, float)) and conf < MIN_CONFIDENCE:
            issues.append(Issue("warning", "low_confidence",
                                f"confidence={conf} < {MIN_CONFIDENCE}",
                                feature_id=fid, field="confidence"))

        # needs_review with no reason
        if feat.get("needs_review") and not (feat.get("review_reason") or "").strip():
            issues.append(Issue("warning", "review_without_reason",
                                "needs_review=true but review_reason is empty",
                                feature_id=fid, field="review_reason"))

        # parameter role sanity
        for p in feat.get("parameters", []) or []:
            role = p.get("role")
            if role and role not in CONVENTIONAL_ROLES:
                issues.append(Issue("info", "non_standard_role",
                                    f"parameter role={role!r} not in conventional set",
                                    feature_id=fid, parameter=p.get("name"),
                                    field="parameters[].role"))

    # Missing information overflow
    mi = template.get("missing_information") or []
    if len(mi) > MAX_MISSING_INFORMATION:
        issues.append(Issue("warning", "too_many_missing_information",
                            f"missing_information has {len(mi)} items (> {MAX_MISSING_INFORMATION})",
                            field="missing_information"))

    # metadata leaking into feature graph (a soft heuristic)
    for feat in fg:
        if feat.get("feature_type") in {"metadata", "catalog_metadata"}:
            issues.append(Issue("warning", "metadata_in_feature_graph",
                                "feature_graph contains a metadata-type feature; "
                                "metadata should live in catalog_metadata, not feature_graph",
                                feature_id=feat.get("id"), field="feature_type"))

    # Compute score: 1.0 minus weighted penalty, clamped to [0, 1]
    penalty = 0.0
    weights = {"error": 0.30, "warning": 0.05, "info": 0.01}
    for i in issues:
        penalty += weights.get(i.level, 0.05)
    score = max(0.0, min(1.0, 1.0 - penalty))
    return score, issues


# ----------------------------------------------------------------------------
# Mechanical repairs
# ----------------------------------------------------------------------------
def apply_mechanical_repairs(template: dict) -> tuple[dict, list[str]]:
    """Make well-defined, low-risk repairs. Anything we cannot fix
    deterministically stays as an issue for codex/human review."""
    repairs: list[str] = []
    out = json.loads(json.dumps(template))  # deep copy

    for feat in out.get("feature_graph", []):
        fid = feat.get("id", "<no id>")

        # If a hole/pocket/boss is missing axis and not needs_review,
        # mark it for review (do NOT guess a default).
        if (feat.get("feature_type") in {"hole", "pocket", "boss"}
                and feat.get("axis") is None
                and not feat.get("needs_review")):
            feat["needs_review"] = True
            feat.setdefault("review_reason", "axis was null; auto-flagged for human/codex review")
            repairs.append(f"{fid}: needs_review set (axis was null)")

        # If needs_review=true and review_reason is empty, fill a placeholder.
        if feat.get("needs_review") and not (feat.get("review_reason") or "").strip():
            feat["review_reason"] = "needs_review set without reason (auto-filled by step2)"
            repairs.append(f"{fid}: empty review_reason filled with placeholder")

        # If modeling_primitive is null/empty and not needs_review, flag it.
        mp = feat.get("modeling_primitive")
        if mp in (None, "") and not feat.get("needs_review"):
            feat["needs_review"] = True
            feat.setdefault("review_reason", "modeling_primitive missing; auto-flagged")
            repairs.append(f"{fid}: needs_review set (modeling_primitive missing)")

    return out, repairs


# ----------------------------------------------------------------------------
# Codex repair (optional, off by default)
# ----------------------------------------------------------------------------
CODEX_SYSTEM_PROMPT = """You are a CAD feature-template repair assistant.
The user will paste a JSON engineering feature template and a list of
issues found by a static review. Your job: produce a JSON patch object
that fixes the issues. You MAY:
  - set needs_review=true on a feature and add a review_reason
  - default axis to one of "X", "Y", "Z" when the feature is clearly
    a hole/boss on a specific face and the coordinate system makes it
    unambiguous
  - set modeling_primitive to a reasonable non-"unknown" value when the
    construction notes make the primitive clear
  - add missing review_reason text
You MAY NOT:
  - change numeric values
  - change parameter symbols
  - add or remove features
  - invent parameters
Output ONLY a JSON object of the form:
  {"repairs": [{"feature_id": "...", "patch": {...}}, ...], "notes": "..."}
The patch is shallow-merged into the feature. Do not wrap in markdown.
"""


def call_codex_repair(template: dict, issues: list[dict],
                      model: str, timeout_s: int = 120) -> dict | None:
    """Call `codex exec` non-interactively and parse the JSON repair plan."""
    user_payload = {
        "issues": issues,
        "template": template,
    }
    prompt = (
        "Repair the following engineering feature template based on the issues. "
        "Output only a JSON object of the form "
        '{"repairs": [{"feature_id": "...", "patch": {...}}], "notes": "..."}. '
        f"Data:\n```json\n{json.dumps(user_payload, indent=2)[:60000]}\n```"
    )
    cmd = [
        "codex", "exec",
        "-m", model,
        "--sandbox", "read-only",
        "--output-last-message", "-",
        prompt,
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s, check=False,
        )
    except subprocess.TimeoutExpired:
        return {"_error": f"codex timed out after {timeout_s}s"}
    except FileNotFoundError:
        return {"_error": "codex CLI not found in PATH"}

    if proc.returncode != 0:
        return {"_error": f"codex exit {proc.returncode}: {proc.stderr.strip()[:200]}"}

    text = proc.stdout.strip()
    # Strip codex's "codex" wrapper output if present; find the first { ... last }
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {"_error": "no JSON object found in codex output"}
    candidate = text[start:end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        return {"_error": f"codex output not valid JSON: {e}; first 200 chars: {candidate[:200]}"}


def apply_codex_repairs(template: dict, plan: dict) -> tuple[dict, list[str], str | None]:
    if not isinstance(plan, dict):
        return template, [], "codex plan was not a JSON object"
    if "_error" in plan:
        return template, [], plan["_error"]
    repairs = plan.get("repairs", [])
    if not isinstance(repairs, list):
        return template, [], "codex plan 'repairs' is not a list"

    out = json.loads(json.dumps(template))
    by_id = {f.get("id"): f for f in out.get("feature_graph", [])}
    applied: list[str] = []
    for r in repairs:
        if not isinstance(r, dict):
            continue
        fid = r.get("feature_id")
        patch = r.get("patch")
        if not fid or fid not in by_id or not isinstance(patch, dict):
            continue
        target = by_id[fid]
        for k, v in patch.items():
            target[k] = v
        applied.append(f"codex patch on {fid}: keys={list(patch.keys())}")
    return out, applied, None


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def discover_templates(templates_dir: Path,
                       limit: int | None,
                       component: str | None) -> list[Path]:
    if component:
        candidate = templates_dir / f"{component}.feature_template.json"
        if not candidate.exists():
            return []
        return [candidate]
    files = sorted(templates_dir.glob("*.feature_template.json"))
    if limit is not None:
        files = files[:limit]
    return files


def review_one(template_path: Path, schema: dict,
               repaired_dir: Path, reviewed_dir: Path,
               allow_codex: bool, codex_model: str) -> ReviewResult:
    t0 = time.time()
    component_id = template_path.stem.replace(".feature_template", "")
    raw = json.loads(template_path.read_text())
    template_id = raw.get("template_id")
    part_family = raw.get("part_family")

    schema_issues = validate_schema(raw, schema)
    score, sem_issues = score_template(raw)
    all_issues = schema_issues + sem_issues

    # Mechanical repairs
    repaired, repairs = apply_mechanical_repairs(raw)

    # Re-score after mechanical repairs
    post_score, post_issues = score_template(repaired)
    score = post_score
    # The post-repair issues supersede pre-repair issues for non-schema checks
    all_issues = schema_issues + post_issues

    codex_used = False
    codex_attempted = False
    codex_error: str | None = None
    needs_codex = (
        allow_codex
        and any(i.level == "warning" for i in post_issues)
        and not all(i.level == "info" for i in post_issues)
    )

    if needs_codex:
        codex_attempted = True
        plan = call_codex_repair(repaired, [i.to_dict() for i in post_issues], codex_model)
        if plan is not None and "_error" not in plan:
            repaired, codex_repairs, err = apply_codex_repairs(repaired, plan)
            repairs.extend(codex_repairs)
            codex_used = True
            # Re-score after codex
            score, post_issues = score_template(repaired)
            all_issues = schema_issues + post_issues
            codex_error = err
        else:
            codex_error = (plan or {}).get("_error", "codex returned no plan")

    # Persist
    repaired_dir.mkdir(parents=True, exist_ok=True)
    reviewed_dir.mkdir(parents=True, exist_ok=True)
    repaired_path: Path | None = None
    if repairs:
        repaired_path = repaired_dir / f"{component_id}.feature_template.json"
        repaired_path.write_text(json.dumps(repaired, indent=2))

    # Pass criteria
    error_issues = [i for i in all_issues if i.level == "error"]
    warning_issues = [i for i in all_issues if i.level == "warning"]
    passed = (len(error_issues) == 0
              and score >= MIN_CONFIDENCE
              and not any(i.code == "schema_violation" for i in error_issues))

    review_obj = {
        "component_id": component_id,
        "template_id": template_id,
        "part_family": part_family,
        "schema_valid": not any(i.code == "schema_violation" for i in all_issues),
        "score": round(score, 3),
        "passed": passed,
        "n_errors": len(error_issues),
        "n_warnings": len(warning_issues),
        "issues": [i.to_dict() for i in all_issues],
        "repairs_applied": repairs,
        "codex": {
            "attempted": codex_attempted,
            "applied": codex_used,
            "model": codex_model if codex_attempted else None,
            "error": codex_error,
        },
    }
    review_path = reviewed_dir / f"{component_id}.review.json"
    review_path.write_text(json.dumps(review_obj, indent=2))

    return ReviewResult(
        component_id=component_id,
        template_id=template_id,
        part_family=part_family,
        schema_valid=review_obj["schema_valid"],
        score=review_obj["score"],
        passed=passed,
        issues=review_obj["issues"],
        repairs_applied=repairs,
        codex_used=codex_used,
        codex_attempted=codex_attempted,
        codex_error=codex_error,
        repaired_template_path=str(repaired_path) if repaired_path else None,
        review_path=str(review_path),
        duration_seconds=round(time.time() - t0, 3),
    )


def print_summary(results: list[ReviewResult], as_json: bool) -> None:
    if as_json:
        summary = {
            "n": len(results),
            "n_passed": sum(1 for r in results if r.passed),
            "n_schema_valid": sum(1 for r in results if r.schema_valid),
            "n_codex_attempted": sum(1 for r in results if r.codex_attempted),
            "n_codex_succeeded": sum(1 for r in results if r.codex_used),
            "mean_score": round(sum(r.score for r in results) / max(1, len(results)), 3),
            "results": [r.to_dict() for r in results],
        }
        print(json.dumps(summary, indent=2))
        return

    print(f"\nStep 2 review: {len(results)} templates")
    print(f"  passed:    {sum(1 for r in results if r.passed)}/{len(results)}")
    print(f"  schema ok: {sum(1 for r in results if r.schema_valid)}/{len(results)}")
    print(f"  mean score: {sum(r.score for r in results) / max(1, len(results)):.3f}")
    if any(r.codex_attempted for r in results):
        print(f"  codex:     {sum(1 for r in results if r.codex_used)}/{sum(1 for r in results if r.codex_attempted)} successful")
    print()
    header = f"{'component_id':<16} {'score':>6} {'pass':>5} {'schema':>7} {'repairs':>8} {'codex':>6}  template_id"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r.component_id:<16} {r.score:>6.3f} {str(r.passed):>5} "
              f"{str(r.schema_valid):>7} {len(r.repairs_applied):>8} "
              f"{('Y' if r.codex_used else ('-' if not r.codex_attempted else 'F')):>6}  "
              f"{r.template_id or '-'}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--templates-dir", type=Path, default=DEFAULT_TEMPLATES_DIR)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                    help="Where to write *.review.json (default: output/feature_templates_reviewed)")
    ap.add_argument("--repaired-dir", type=Path, default=DEFAULT_REPAIRED_DIR,
                    help="Where to write repaired *.feature_template.json (default: output/feature_templates_repaired)")
    ap.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--component", type=str, default=None,
                    help="Review a single component_id (no extension)")
    ap.add_argument("--allow-codex-repair", action="store_true",
                    help="If set, route templates with warnings to a Codex repair call")
    ap.add_argument("--codex-model", type=str, default=DEFAULT_CODEX_MODEL,
                    help=f"Codex model for repair (default: {DEFAULT_CODEX_MODEL})")
    ap.add_argument("--json", action="store_true", help="Emit machine-readable JSON summary")
    args = ap.parse_args(argv)

    if not args.schema.exists():
        print(f"ERROR: schema not found: {args.schema}", file=sys.stderr)
        return 2
    schema = json.loads(args.schema.read_text())

    if not args.templates_dir.exists():
        print(f"ERROR: templates dir not found: {args.templates_dir}", file=sys.stderr)
        return 2

    files = discover_templates(args.templates_dir, args.limit, args.component)
    if not files:
        print(f"ERROR: no templates found in {args.templates_dir}", file=sys.stderr)
        return 2

    results: list[ReviewResult] = []
    for f in files:
        r = review_one(f, schema, args.repaired_dir, args.output_dir,
                       args.allow_codex_repair, args.codex_model)
        results.append(r)
        if not args.json:
            status = "PASS" if r.passed else "FAIL"
            n_rep = len(r.repairs_applied)
            cx = "codex=Y" if r.codex_used else ("codex=F" if r.codex_attempted else "")
            print(f"  [{status}] {r.component_id}  score={r.score:.3f}  repairs={n_rep}  {cx}")
    print_summary(results, args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
