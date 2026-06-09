#!/usr/bin/env python3
"""
Step 4.5 — Resolve symbolic dimension references via Codex (LLM-assisted).

Reads every <component_id>__<part_number>.resolved_cad_spec.json in
output/resolved_specs/, scans for unresolved string-valued parameters
(like "D", "M", "P", "SC", "F", "C", "R", "W", "ℓ1"), and asks Codex
to resolve each one to a concrete number using:

  * the spec's coordinate system and parameter_bindings (for context)
  * the MISUMI catalog data at downloads/<cid>/json/specs.json (for the
    part family + size_no specific table values)
  * the feature's own construction.profile_description (for hints)

The result is written to output/symbol_bindings.json as a single file:

  {
    "<cid>__<pn>": {
      "D": 20.0,
      "M": 20.0,
      "P": 17.5,
      ...
    },
    ...
  }

This file is then loaded by scripts/step5_codegen.py and used to
substitute numeric values for symbolic references before emitter
dispatch. The bindings are intentionally additive: an unresolved symbol
not in the bindings file still gets the existing "skip with TODO"
treatment, so the script degrades gracefully if Codex is offline or
returns garbage.

By default this step is OFF (no Codex calls). Pass --allow-llm to
enable. Codex is called with model gpt-5.4-mini, low reasoning — symbol
resolution from a table is mechanical, not creative.

Usage:
  python scripts/step5_resolve_symbols.py --all                  # resolve all 12 specs
  python scripts/step5_resolve_symbols.py --component 110310764189
  python scripts/step5_resolve_symbols.py --allow-llm --all       # actually call Codex
  python scripts/step5_resolve_symbols.py --json                  # machine-readable summary
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPECS_DIR = REPO_ROOT / "output" / "resolved_specs"
DEFAULT_OUTPUT_FILE = REPO_ROOT / "output" / "symbol_bindings.json"
DEFAULT_DOWNLOADS_DIR = REPO_ROOT / "downloads"
DEFAULT_CODEX_MODEL = "gpt-5.4-mini"


# ----------------------------------------------------------------------------
# Data model
# ----------------------------------------------------------------------------
@dataclass
class ResolveResult:
    component_id: str
    part_number: str
    spec_path: str
    unresolved_symbols: list[str] = field(default_factory=list)
    resolved_symbols: dict[str, float] = field(default_factory=dict)
    unresolved_after: list[str] = field(default_factory=list)
    codex_called: bool = False
    codex_rc: int | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _is_unresolved(value: Any) -> bool:
    """A value is 'unresolved' if it's a string that isn't a plain number
    (allow negative, decimal, scientific notation). Common unresolved forms:
    'D', 'F25', 'M', 'L+J', 'SC + SX + ℓ1', 'P10'."""
    if not isinstance(value, str):
        return False
    s = value.strip()
    if not s:
        return False
    # Plain number string?
    try:
        float(s)
        return False
    except ValueError:
        pass
    return True


def _collect_unresolved(spec: dict) -> list[dict]:
    """Return [{feature_id, name, value, role}] for every unresolved param."""
    out: list[dict] = []
    for f in spec.get("resolved_features", []):
        for p in f.get("parameters", []):
            if _is_unresolved(p.get("value")):
                out.append(
                    {
                        "feature_id": f.get("id"),
                        "feature_type": f.get("feature_type"),
                        "modeling_primitive": f.get("modeling_primitive"),
                        "name": p["name"],
                        "value": p["value"],
                        "role": p.get("role"),
                    }
                )
    return out


def _catalog_excerpt(component_id: str, downloads_dir: Path, max_chars: int = 12_000) -> str:
    """Return a small text excerpt of the catalog table relevant to the
    component. Truncated to max_chars to keep the Codex prompt small."""
    spec_path = downloads_dir / component_id / "json" / "specs.json"
    if not spec_path.exists():
        return f"(no catalog data at {spec_path})"
    try:
        d = json.loads(spec_path.read_text())
    except Exception as e:
        return f"(error reading {spec_path}: {e})"
    tables = d.get("tables") or []
    excerpt_lines: list[str] = [
        f"Title: {d.get('title', '?')}",
        f"Number of tables: {len(tables)}",
        "",
    ]
    for i, t in enumerate(tables[:8]):  # cap at 8 tables
        if not isinstance(t, list) or not t:
            continue
        excerpt_lines.append(f"--- table[{i}] ({len(t)} rows) ---")
        for row in t[:25]:  # cap at 25 rows/table
            excerpt_lines.append("  | " + " | ".join(str(c) for c in row))
        excerpt_lines.append("")
    s = "\n".join(excerpt_lines)
    if len(s) > max_chars:
        s = s[:max_chars] + "\n... (truncated)"
    return s


def _build_codex_prompt(spec: dict, unresolved: list[dict], catalog_excerpt: str) -> str:
    """Build a tight prompt asking Codex to resolve each unresolved symbol."""
    cid = spec.get("catalog_id", "?")
    pn = spec.get("part_number", "?")
    bindings = spec.get("parameter_bindings", {})
    coord = spec.get("coordinate_system", {})

    # Compact, structured
    unresolved_lines = "\n".join(
        f"  - feature_id={u['feature_id']!r}  type={u['feature_type']!r}  "
        f"prim={u['modeling_primitive']!r}  symbol={u['name']!r}  raw_value={u['value']!r}  "
        f"role={u['role']!r}"
        for u in unresolved
    )

    binding_lines = "\n".join(f"  - {k!r}: {v!r}" for k, v in bindings.items())

    return f"""You are resolving symbolic dimension references for a CAD
spec to concrete numbers. The part is {cid} / {pn}.

Coordinate system:
  origin: {coord.get('origin', '?')}
  x_axis: {coord.get('x_axis', '?')}

Already-resolved parameter bindings (snake_case name -> value):
{binding_lines}

Unresolved symbols (you need to produce a numeric value for each):
{unresolved_lines}

Catalog excerpt for this part family:
{catalog_excerpt}

TASK: For EACH unresolved symbol, produce a numeric value in millimeters
that matches the catalog for this specific part (family + size). Use the
already-resolved bindings as anchors. If a symbol is a thread size like
'M' for an M20 thread, the numeric value is the major diameter in mm
(20.0). If a symbol is a standard catalog dimension (P, F, J, K, C, R
etc. for MISUMI linear shafts), use the catalog's value for this size_no.

OUTPUT FORMAT (must be valid JSON, nothing else):
{{
  "<symbol_name>": <numeric_value>,
  ...
}}

For example:
{{"D": 20.0, "M": 20.0, "P": 17.5, "F": 15.0, "J": 10.0, "C": 1.0, "R": 0.5, "B": 30.0}}

Do NOT include commentary, explanations, or markdown fences. Only the JSON object."""


def _call_codex(prompt: str, model: str) -> tuple[int, str, str]:
    """Run `codex exec` with the given prompt. Return (rc, stdout, stderr)."""
    cmd = [
        "codex", "exec",
        "-m", model,
        "-c", "model_reasoning_effort=low",
        "-s", "workspace-write",
        "--ephemeral",
        prompt,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    return proc.returncode, proc.stdout, proc.stderr


def _extract_json(text: str) -> dict | None:
    """Best-effort: find the first {...} JSON object in the Codex response
    and parse it. Codex usually returns the JSON cleanly but sometimes
    wraps it in a 'final' message."""
    # Try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # Find first '{' and last '}'
    first = text.find("{")
    last = text.rfind("}")
    if first == -1 or last == -1 or last < first:
        return None
    candidate = text[first : last + 1]
    try:
        return json.loads(candidate)
    except Exception:
        pass
    # Tolerate trailing commas
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
    try:
        return json.loads(candidate)
    except Exception:
        return None


# ----------------------------------------------------------------------------
# Per-spec driver
# ----------------------------------------------------------------------------
def process_spec(
    spec_path: Path,
    output_bindings: dict[str, dict[str, float]],
    downloads_dir: Path,
    *,
    allow_llm: bool = False,
    codex_model: str = DEFAULT_CODEX_MODEL,
) -> ResolveResult:
    cid = spec_path.stem.split("__")[0]
    pn = spec_path.stem.split("__", 1)[1].removesuffix(".resolved_cad_spec")
    key = f"{cid}__{pn}"
    res = ResolveResult(component_id=cid, part_number=pn, spec_path=str(spec_path))

    try:
        spec = json.loads(spec_path.read_text())
    except Exception as e:
        res.error = f"read/parse: {e}"
        return res

    unresolved = _collect_unresolved(spec)
    res.unresolved_symbols = [u["name"] for u in unresolved]

    # Already have a binding for this spec? Use it.
    existing = output_bindings.get(key, {})
    if existing and not allow_llm:
        # Skip; existing bindings file is authoritative
        res.resolved_symbols = existing
        res.unresolved_after = [u for u in res.unresolved_symbols if u not in existing]
        return res

    if not unresolved:
        res.resolved_symbols = {}
        res.unresolved_after = []
        return res

    if not allow_llm:
        # No existing bindings, no LLM call allowed — just report what we couldn't resolve
        res.resolved_symbols = {}
        res.unresolved_after = res.unresolved_symbols[:]
        return res

    # Call Codex
    catalog_excerpt = _catalog_excerpt(cid, downloads_dir)
    prompt = _build_codex_prompt(spec, unresolved, catalog_excerpt)
    rc, stdout, stderr = _call_codex(prompt, codex_model)
    res.codex_called = True
    res.codex_rc = rc
    if rc != 0:
        res.error = f"codex: {stderr.strip()[:200] or 'non-zero exit'}"
        return res

    parsed = _extract_json(stdout)
    if not parsed or not isinstance(parsed, dict):
        res.error = f"codex: could not parse JSON from response (stdout len={len(stdout)})"
        return res

    # Coerce values to float; reject non-numerics
    cleaned: dict[str, float] = {}
    for k, v in parsed.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            cleaned[str(k)] = float(v)
        elif isinstance(v, str):
            try:
                cleaned[str(k)] = float(v)
            except ValueError:
                pass
    res.resolved_symbols = cleaned
    res.unresolved_after = [u for u in res.unresolved_symbols if u not in cleaned]
    return res


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "Step 4.5")
    p.add_argument("--component", help="Filter: component_id (e.g. 110310764189)")
    p.add_argument("--part-number", dest="part_number")
    p.add_argument("--specs-dir", type=Path, default=DEFAULT_SPECS_DIR)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE)
    p.add_argument("--downloads-dir", type=Path, default=DEFAULT_DOWNLOADS_DIR)
    p.add_argument("--allow-llm", action="store_true", help="Call Codex for unresolved symbols")
    p.add_argument("--codex-model", default=DEFAULT_CODEX_MODEL)
    p.add_argument("--all", dest="select_all", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--json", dest="json_out", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    # Load existing bindings (we append/update, never destroy)
    if args.output.exists():
        try:
            output_bindings: dict[str, dict[str, float]] = json.loads(args.output.read_text())
        except Exception:
            output_bindings = {}
    else:
        output_bindings = {}

    # Discover specs
    specs: list[Path] = sorted(args.specs_dir.glob("*.resolved_cad_spec.json"))
    if args.component:
        specs = [s for s in specs if s.stem.split("__")[0] == args.component]
    if args.part_number:
        specs = [s for s in specs if s.stem.split("__", 1)[1].removesuffix(".resolved_cad_spec") == args.part_number]
    if args.limit:
        specs = specs[: args.limit]
    if not args.select_all and not args.component and not args.part_number:
        if not specs:
            print(f"ERROR: no specs found in {args.specs_dir}", file=sys.stderr)
            return 1

    if args.verbose:
        print(
            f"==> Step 4.5: {len(specs)} spec(s), allow_llm={args.allow_llm}, "
            f"output={args.output}",
            file=sys.stderr,
        )

    results: list[ResolveResult] = []
    for spec_path in specs:
        r = process_spec(
            spec_path, output_bindings, args.downloads_dir,
            allow_llm=args.allow_llm, codex_model=args.codex_model,
        )
        results.append(r)
        # Update the in-memory bindings dict so subsequent specs in the same
        # run could theoretically reuse them
        if r.ok and r.resolved_symbols:
            key = f"{r.component_id}__{r.part_number}"
            output_bindings[key] = {**output_bindings.get(key, {}), **r.resolved_symbols}
        if args.verbose and not args.json_out:
            status = "OK" if r.ok and not r.unresolved_after else f"PARTIAL ({len(r.unresolved_after)} still unresolved)" if r.ok else f"ERR ({r.error})"
            print(
                f"  [{status}] {r.part_number:20s} "
                f"unresolved={len(r.unresolved_symbols):2d}  "
                f"resolved={len(r.resolved_symbols):2d}  "
                f"codex={'Y' if r.codex_called else 'N'}",
                file=sys.stderr,
            )

    # Persist the bindings file (only if anything was resolved)
    if any(r.resolved_symbols for r in results):
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output_bindings, indent=2, sort_keys=True))
        if args.verbose and not args.json_out:
            print(f"==> Wrote {args.output}", file=sys.stderr)

    if args.json_out:
        print(json.dumps([asdict(r) for r in results], indent=2))
    else:
        ok = sum(1 for r in results if r.ok)
        total_unresolved = sum(len(r.unresolved_symbols) for r in results)
        total_resolved = sum(len(r.resolved_symbols) for r in results)
        codex_calls = sum(1 for r in results if r.codex_called)
        print(
            f"Step 4.5: {ok}/{len(results)} ok, {total_resolved}/{total_unresolved} symbols resolved, "
            f"{codex_calls} codex calls",
            file=sys.stderr,
        )

    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
