#!/usr/bin/env python3
"""Run Step 5 (deterministic CadQuery code generation) and capture every
output into a fresh ``output/runs/step5/<run_id>/`` folder so you can
review the run in the browser via ``bin/review_server.py``.

This is a thin wrapper around ``scripts/step5_codegen.py``. It does NOT
modify step5_codegen itself — the goal is to keep that script's test
contract intact and add a run-lifecycle layer on top.

Output tree (per run)::

    output/runs/step5/<run_id>/
        index.html
        run.json
        components.json
        components/<cid>/
            index.html
            meta.json
            <part_number>.model.py
            <part_number>.step
            <part_number>.stl
        cad_models/                    # the legacy flat layout, mirrored

Usage::

    python bin/run_step5.py --all
    python bin/run_step5.py --component 110300324920
    python bin/run_step5.py --all --no-smoke
    python bin/run_step5.py --all --dry-run
    python bin/run_step5.py --list
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))  # so 'pipeline' and 'scripts' are importable

from pipeline.output import Run  # noqa: E402

# Import step5 lazily (cadquery may not be installed in every venv)
import importlib  # noqa: E402
step5 = importlib.import_module("scripts.step5_codegen")


def _format_run_summary_html(component, body_extra: str = "") -> str:
    """Per-component body for the run-layout page.

    Embeds the step5 stats in a small table and links to the artifacts
    (model.py / step / stl) that ``Component.add_artifact`` registered.
    """
    s = component.summary
    rows = "".join(
        f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in s.items()
    )
    return f"<h2>Step 5 result</h2><table>{rows}</table>{body_extra}"


def _resolve_specs(args) -> list[Path]:
    specs = sorted(args.specs_dir.glob("*.resolved_cad_spec.json"))
    if args.component:
        specs = [s for s in specs if step5.component_id_of(s) == args.component]
    if args.part_number:
        specs = [s for s in specs if step5.part_number_of(s) == args.part_number]
    if args.limit:
        specs = specs[: args.limit]
    return specs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "run_step5")
    p.add_argument("--all", dest="select_all", action="store_true", help="Process all specs")
    p.add_argument("--component", help="Filter: component_id")
    p.add_argument("--part-number", dest="part_number", help="Filter: part number")
    p.add_argument("--specs-dir", type=Path, default=step5.DEFAULT_SPECS_DIR)
    p.add_argument("--out-dir", type=Path,
                   help="Override base out dir (default: the run's own cad_models/)")
    p.add_argument("--schema", type=Path, default=step5.DEFAULT_SCHEMA)
    p.add_argument("--bindings-file", type=Path, default=step5.DEFAULT_BINDINGS_FILE)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-smoke", dest="smoke", action="store_false")
    p.add_argument("--list", action="store_true", help="List existing runs and exit")
    p.add_argument("--run-id", dest="explicit_run_id", help="Override auto-generated run id")
    args = p.parse_args()

    if args.list:
        from pipeline.output import _print_runs, RUNS_ROOT
        _print_runs(RUNS_ROOT)
        return 0

    specs = _resolve_specs(args)
    if not args.select_all and not args.component and not args.part_number:
        if not specs:
            print(f"ERROR: no specs in {args.specs_dir}", file=sys.stderr)
            return 1
        if not specs:
            return 1

    if not specs:
        print(f"ERROR: no specs matched", file=sys.stderr)
        return 1

    # The run's own output dir is inside the run folder; we always use it
    # so the per-component pages can find the .step/.stl/.model.py links
    # via relative path. If the user passed --out-dir, honor it but also
    # mirror to the run dir.
    if args.out_dir is not None and not args.out_dir.is_relative_to(REPO_ROOT / "output" / "runs"):
        # user wants an external out dir; respect it but warn
        print(f"NOTE: --out-dir={args.out_dir} is outside the run folder; "
              f"artifacts will NOT be discoverable from the run page", file=sys.stderr)
        out_dir = args.out_dir
    else:
        out_dir = None  # filled in after Run.open()

    run = Run.open(
        "step5",
        run_id=args.explicit_run_id,
        args={
            "all": args.select_all,
            "component": args.component,
            "part_number": args.part_number,
            "limit": args.limit,
            "dry_run": args.dry_run,
            "smoke": args.smoke,
        },
    )
    if out_dir is None:
        out_dir = run.path / "cad_models"

    print(f"[run_step5] run_id={run.run_id} path={run.path}", file=sys.stderr)
    print(f"[run_step5] {len(specs)} spec(s) -> {out_dir}", file=sys.stderr)

    schema = json.loads(args.schema.read_text())
    all_bindings = step5.load_symbol_bindings(args.bindings_file)

    n_ok = 0
    n_err = 0
    for spec_path in specs:
        cid = step5.component_id_of(spec_path)
        pn = step5.part_number_of(spec_path)
        c = run.component(cid)
        key = f"{cid}__{pn}"
        spec_bindings = all_bindings.get(key, {})

        try:
            r = step5.process_spec(
                spec_path, out_dir, schema,
                dry_run=args.dry_run, smoke=args.smoke, bindings=spec_bindings,
            )
        except Exception as e:
            c.fail(f"unhandled: {e}")
            c.json()
            c.html(f"<p class='err'>Unhandled exception: {e}</p>",
                   title=f"{cid} / {pn}")
            n_err += 1
            continue

        # Register artifacts (check .model.py first since it ends with .py)
        if not args.dry_run and r.ok:
            artifact_specs = [
                (".model.py", "Generated model.py", "python"),
                (".step",     "STEP geometry",      "step"),
                (".stl",      "STL mesh",           "stl"),
            ]
            for ext, label, kind in artifact_specs:
                f = out_dir / f"{pn}{ext}"
                if f.is_file():
                    c.add_artifact(label, f, kind=kind)

        if r.ok:
            c.finish(
                part_number=pn,
                features_total=r.features_total,
                features_implemented=len(r.features_implemented),
                features_skipped=len(r.features_skipped),
                parameters=r.parameters,
                parameters_null=r.parameters_null,
                bindings_applied=r.bindings_applied,
                schema_valid=r.schema_valid,
                syntax_valid=r.syntax_valid,
                runtime_smoke=r.runtime_smoke,
            )
            c.json()
            c.html(_format_run_summary_html(c))
            n_ok += 1
        else:
            c.fail(r.error or "unknown")
            c.json()
            c.html(_format_run_summary_html(c, body_extra=f"<p class='err'>Error: {c.error}</p>"),
                   title=f"{cid} / {pn} (ERR)")
            n_err += 1

        print(
            f"  [{'OK' if r.ok else 'ERR'}] {pn:20s} "
            f"feat={r.features_total}/{len(r.features_implemented)} "
            f"params={r.parameters}/{r.parameters_null} null "
            f"bindings={r.bindings_applied} "
            f"smoke={r.runtime_smoke}",
            file=sys.stderr,
        )

    run.finish(ok=n_ok, errors=n_err)
    print(
        f"[run_step5] done — {n_ok} ok, {n_err} error. Run page: "
        f"output/runs/step5/{run.run_id}/index.html",
        file=sys.stderr,
    )
    print(f"[run_step5] open via: python bin/review_server.py", file=sys.stderr)
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
