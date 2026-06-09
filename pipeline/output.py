"""Run layout helpers.

Every step in the pipeline writes its outputs into::

    output/runs/<step>/<run_id>/
        index.html              — run summary dashboard
        run.json                — machine metadata for the whole run
        components.json         — flat index of {component_id -> summary}
        components/<cid>/
            index.html          — per-component detail page
            meta.json           — per-component machine metadata

A run_id is a timestamp + short random suffix, e.g.
``2026-06-09T08-30-12_a3f9``. Sortable and unique within a step.

This module is the single entry point — steps call :func:`open_run` to
get a :class:`Run` object, then per-component via :meth:`Run.component`
which yields a :class:`Component` that exposes ``.html()`` and
``.json()`` writers. Don't write to the layout directly from steps.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import html
import json
import os
import secrets
import shutil
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = REPO_ROOT / "output" / "runs"


def _slugify(s: str) -> str:
    return s.replace(":", "-").replace("+00-00", "").strip("-")


# Module-level monotonic counter so two IDs in the same second stay
# sortable in call order. Resets when the second changes.
_last_second: str | None = None
_subsec_counter: int = 0


def _next_subsec() -> str:
    """Return a 4-hex-digit counter that increments within a single second
    so ``make_run_id()`` is sortable by call order across rapid invocations.
    """
    global _last_second, _subsec_counter
    now = _dt.datetime.now()
    sec = now.strftime("%Y-%m-%dT%H-%M-%S")
    if sec != _last_second:
        _last_second = sec
        _subsec_counter = 0
    else:
        _subsec_counter += 1
    return f"{_subsec_counter:04x}"


def make_run_id(now: _dt.datetime | None = None) -> str:
    """``2026-06-09T08-30-12_a3f9`` — sortable + collision-resistant.

    Within a single second the suffix is a monotonic per-process counter
    so two ``make_run_id()`` calls return lexicographically ordered IDs.
    Collisions across processes are unlikely (16-bit space per second);
    ``Run.open`` detects an existing dir and appends another suffix.
    """
    if now is None:
        return f"{_dt.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}_{_next_subsec()}"
    suffix = secrets.token_hex(2)  # when caller pins a fixed time, randomness is fine
    return f"{now.strftime('%Y-%m-%dT%H-%M-%S')}_{suffix}"


def step_dir(step: str) -> Path:
    """``output/runs/<step>/`` — created on first use."""
    p = RUNS_ROOT / step
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass
class Component:
    component_id: str
    path: Path
    summary: dict[str, Any] = field(default_factory=dict)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    started_at: str = field(default_factory=lambda: _dt.datetime.now().isoformat(timespec="seconds"))
    finished_at: str | None = None

    # ---- writers --------------------------------------------------------

    def html(self, body_html: str, title: str | None = None) -> Path:
        title = title or self.component_id
        full = self._render_html(title, body_html)
        out = self.path / "index.html"
        out.write_text(full)
        return out

    def json(self, extra: dict[str, Any] | None = None) -> Path:
        data = {
            "component_id": self.component_id,
            "summary": self.summary,
            "artifacts": self.artifacts,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at or _dt.datetime.now().isoformat(timespec="seconds"),
        }
        if extra:
            data.update(extra)
        out = self.path / "meta.json"
        out.write_text(json.dumps(data, indent=2, default=str))
        return out

    def add_artifact(self, label: str, path: str | Path, kind: str | None = None) -> None:
        """Register a file for the per-component page to link to."""
        rel = os.path.relpath(Path(path), self.path)
        self.artifacts.append({"label": label, "path": rel, "kind": kind})

    def fail(self, message: str) -> None:
        self.error = message
        self.finished_at = _dt.datetime.now().isoformat(timespec="seconds")

    def finish(self, **summary: Any) -> None:
        self.summary.update(summary)
        self.finished_at = _dt.datetime.now().isoformat(timespec="seconds")

    # ---- internals ------------------------------------------------------

    def _render_html(self, title: str, body_html: str) -> str:
        art_html = ""
        if self.artifacts:
            rows = "".join(
                f"<tr><td>{html.escape(a.get('kind') or '')}</td>"
                f"<td><a href={html.escape(a['path'], quote=True)}>{html.escape(a['label'])}</a></td></tr>"
                for a in self.artifacts
            )
            art_html = f"<h2>Artifacts</h2><table><tr><th>Kind</th><th>File</th></tr>{rows}</table>"
        err_html = (
            f"<div class='err'><b>Error:</b> {html.escape(self.error)}</div>"
            if self.error
            else ""
        )
        return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
  body{{font-family:system-ui;max-width:920px;margin:24px auto;padding:0 16px;color:#222}}
  h1{{border-bottom:2px solid #444;padding-bottom:6px}}
  table{{border-collapse:collapse;width:100%;margin:12px 0}}
  th,td{{border:1px solid #ddd;padding:6px 10px;text-align:left;font-size:14px}}
  th{{background:#f4f4f4}}
  .err{{background:#fee;border-left:4px solid #c33;padding:8px 12px;margin:12px 0}}
  .meta{{color:#666;font-size:13px}}
  a{{color:#06c;text-decoration:none}} a:hover{{text-decoration:underline}}
</style></head><body>
<h1>{html.escape(title)}</h1>
<div class="meta">component: <code>{html.escape(self.component_id)}</code>
 · started: {html.escape(self.started_at)}
 {f' · finished: {html.escape(self.finished_at)}' if self.finished_at else ''}
</div>
{err_html}
{art_html}
{body_html}
</body></html>"""


@dataclass
class Run:
    step: str
    run_id: str
    path: Path
    args: dict[str, Any] = field(default_factory=dict)
    started_at: str = field(default_factory=lambda: _dt.datetime.now().isoformat(timespec="seconds"))
    finished_at: str | None = None
    components: dict[str, Component] = field(default_factory=dict)
    status: str = "running"
    summary: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def open(cls, step: str, run_id: str | None = None, *, args: dict[str, Any] | None = None, **flags: Any) -> "Run":
        run_id = run_id or make_run_id()
        path = step_dir(step) / run_id
        if path.exists():
            # collision is theoretically possible; nudge the suffix
            run_id = run_id + "_" + secrets.token_hex(2)
            path = step_dir(step) / run_id
        path.mkdir(parents=True, exist_ok=True)
        (path / "components").mkdir(exist_ok=True)
        # Merge explicit args dict with **flags (which usually carries CLI flags
        # as --key=val strings; we convert to bool/native where obvious).
        merged: dict[str, Any] = dict(args or {})
        for k, v in flags.items():
            merged[k.replace("_", "-")] = v
        return cls(step=step, run_id=run_id, path=path, args=merged)

    def component(self, component_id: str) -> Component:
        if component_id in self.components:
            return self.components[component_id]
        cdir = self.path / "components" / component_id
        cdir.mkdir(parents=True, exist_ok=True)
        c = Component(component_id=component_id, path=cdir)
        self.components[component_id] = c
        return c

    def finish(self, **summary: Any) -> None:
        self.finished_at = _dt.datetime.now().isoformat(timespec="seconds")
        self.summary.update(summary)
        self.status = "completed"
        self._write_components_index()
        self._write_run_json()
        self._write_run_index_html()

    def abort(self, error: str) -> None:
        self.finished_at = _dt.datetime.now().isoformat(timespec="seconds")
        self.status = "aborted"
        self.summary["error"] = error
        self._write_components_index()
        self._write_run_json()
        self._write_run_index_html()

    # ---- index writers --------------------------------------------------

    def _write_components_index(self) -> None:
        items = []
        for cid, c in self.components.items():
            items.append({
                "component_id": cid,
                "status": "error" if c.error else "ok",
                "summary": c.summary,
                "path": f"components/{cid}/index.html",
            })
        (self.path / "components.json").write_text(
            json.dumps({"run_id": self.run_id, "step": self.step, "components": items}, indent=2)
        )

    def _write_run_json(self) -> None:
        data = {
            "run_id": self.run_id,
            "step": self.step,
            "status": self.status,
            "args": self.args,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "n_components": len(self.components),
            "n_errors": sum(1 for c in self.components.values() if c.error),
            "summary": self.summary,
        }
        (self.path / "run.json").write_text(json.dumps(data, indent=2, default=str))

    def _write_run_index_html(self) -> None:
        rows = []
        for cid, c in self.components.items():
            cls = "err" if c.error else "ok"
            sym = "✗" if c.error else "✓"
            summ = ", ".join(f"{k}={v}" for k, v in c.summary.items())
            rows.append(
                f"<tr class='{cls}'><td>{sym}</td>"
                f"<td><a href=components/{html.escape(cid)}/index.html>{html.escape(cid)}</a></td>"
                f"<td>{html.escape(summ)}</td>"
                f"<td>{html.escape(c.error) if c.error else ''}</td></tr>"
            )
        n_ok = sum(1 for c in self.components.values() if not c.error)
        n_err = len(self.components) - n_ok
        # Render args cleanly: True/False for bools, no quotes on strings,
        # drop None
        def _fmt(v: Any) -> str:
            if isinstance(v, bool):
                return "" if v else "=false"  # show only if False to reduce noise
            return f"={v}"
        arg_rows = []
        for k, v in self.args.items():
            display = f"<code>--{html.escape(str(k))}{html.escape(_fmt(v))}</code>"
            if v is not None:
                arg_rows.append(display)
        args_html = "<br>".join(arg_rows)
        body = f"""
<h2>Run summary</h2>
<table>
  <tr><th>Step</th><td>{html.escape(self.step)}</td></tr>
  <tr><th>Run id</th><td><code>{html.escape(self.run_id)}</code></td></tr>
  <tr><th>Started</th><td>{html.escape(self.started_at)}</td></tr>
  <tr><th>Finished</th><td>{html.escape(self.finished_at or '')}</td></tr>
  <tr><th>Status</th><td>{html.escape(self.status)}</td></tr>
  <tr><th>Components</th><td>{len(self.components)} ({n_ok} ok, {n_err} error)</td></tr>
  <tr><th>Args</th><td>{args_html or '<i>(none)</i>'}</td></tr>
</table>
<h2>Components</h2>
<table>
  <tr><th></th><th>ID</th><th>Summary</th><th>Error</th></tr>
  {''.join(rows) or '<tr><td colspan=4><i>none</i></td></tr>'}
</table>
"""
        full = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(self.step)} / {html.escape(self.run_id)}</title>
<style>
  body{{font-family:system-ui;max-width:1000px;margin:24px auto;padding:0 16px;color:#222}}
  h1{{border-bottom:2px solid #444;padding-bottom:6px}}
  h2{{margin-top:24px;border-bottom:1px solid #ccc;padding-bottom:4px}}
  table{{border-collapse:collapse;width:100%;margin:8px 0}}
  th,td{{border:1px solid #ddd;padding:6px 10px;text-align:left;font-size:14px}}
  th{{background:#f4f4f4}}
  tr.err td{{background:#fee}}
  a{{color:#06c;text-decoration:none}} a:hover{{text-decoration:underline}}
  code{{background:#f4f4f4;padding:1px 4px;border-radius:3px}}
</style></head><body>
<h1>{html.escape(self.step)} <span style="color:#888">·</span> {html.escape(self.run_id)}</h1>
{body}
<p style="margin-top:32px"><a href="../../index.html">← all steps</a></p>
</body></html>"""
        (self.path / "index.html").write_text(full)


# ---------------------------------------------------------------------------
# Tiny CLI: list runs for a given step (or all steps)
# ---------------------------------------------------------------------------

def _print_runs(root: Path) -> None:
    if not root.exists():
        print(f"(no runs yet — {root} does not exist)")
        return
    for step_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        runs = sorted(p for p in step_dir.iterdir() if p.is_dir())
        print(f"\n[{step_dir.name}]  ({len(runs)} runs)")
        for r in runs[-5:]:
            rj = r / "run.json"
            if rj.exists():
                try:
                    d = json.loads(rj.read_text())
                    print(f"  {d['run_id']:30s}  {d['status']:10s}  "
                          f"n={d.get('n_components', 0)} err={d.get('n_errors', 0)}")
                except Exception:
                    print(f"  {r.name}  (run.json unreadable)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "pipeline.output")
    p.add_argument("--list", action="store_true", help="List all runs across all steps")
    p.add_argument("--root", type=Path, default=RUNS_ROOT, help="Override runs root")
    args = p.parse_args()
    if args.list:
        _print_runs(args.root)
        return 0
    p.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
