#!/usr/bin/env python3
"""Run-review frontend server.

A single-file stdlib HTTP server that serves the run output tree at
``output/runs/`` so you can review step runs in a browser.

Layout it serves::

    /                              → output/runs/index.html (steps list)
    /runs/                         → output/runs/index.html
    /runs/<step>/                  → output/runs/<step>/index.html (run picker)
    /runs/<step>/<run_id>/         → output/runs/<step>/<run_id>/index.html
    /runs/<step>/<run_id>/...      → any file under that run dir
    /api/runs                      → JSON: full tree of {step -> [run_ids...]}
    /api/runs/<step>/<run_id>      → JSON: the run.json contents
    /api/components/<step>/<run>/<cid>
                                  → JSON: components/<cid>/meta.json

Usage::

    python bin/review_server.py            # binds 127.0.0.1:8765
    python bin/review_server.py --port 9000
    python bin/review_server.py --root /some/other/output/runs

Stdlib only. No pip install, no virtualenv juggling. Stops with Ctrl-C.

Open http://127.0.0.1:8765/ in a browser.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = REPO_ROOT / "output" / "runs"


# ---------------------------------------------------------------------------
# Index pages (the per-step picker and the root steps list)
# ---------------------------------------------------------------------------

INDEX_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Run review</title>
<style>
  body{{font-family:system-ui;max-width:960px;margin:24px auto;padding:0 16px;color:#222}}
  h1{{border-bottom:2px solid #444;padding-bottom:6px}}
  h2{{margin-top:24px;border-bottom:1px solid #ccc;padding-bottom:4px}}
  table{{border-collapse:collapse;width:100%;margin:8px 0}}
  th,td{{border:1px solid #ddd;padding:6px 10px;text-align:left;font-size:14px}}
  th{{background:#f4f4f4}}
  a{{color:#06c;text-decoration:none}} a:hover{{text-decoration:underline}}
  code{{background:#f4f4f4;padding:1px 4px;border-radius:3px}}
  .err{{color:#c33}}
  .ok{{color:#060}}
  .meta{{color:#666;font-size:13px}}
</style></head><body>
{body}
</body></html>"""


def _render_root(root: Path) -> str:
    if not root.exists():
        body = f"<h1>Run review</h1><p class='meta'>No runs yet. {root} does not exist.</p>"
        return INDEX_TEMPLATE.format(body=body)
    steps = sorted(p for p in root.iterdir() if p.is_dir())
    if not steps:
        body = f"<h1>Run review</h1><p class='meta'>No steps yet. Run a step to create one.</p>"
        return INDEX_TEMPLATE.format(body=body)
    rows = []
    for s in steps:
        runs = sorted((p for p in s.iterdir() if p.is_dir()), reverse=True)
        latest = runs[0] if runs else None
        latest_cell = (
            f"<a href=runs/{s.name}/{latest.name}/index.html><code>{latest.name}</code></a>"
            if latest
            else "<i>none</i>"
        )
        rows.append(
            f"<tr><td><a href=runs/{s.name}/index.html><b>{s.name}</b></a></td>"
            f"<td>{len(runs)}</td><td>{latest_cell}</td></tr>"
        )
    body = f"""
<h1>Run review</h1>
<p class="meta">Serving <code>{root}</code></p>
<h2>Steps</h2>
<table>
  <tr><th>Step</th><th>Runs</th><th>Latest</th></tr>
  {''.join(rows)}
</table>
<p style="margin-top:32px"><a href="api/runs">JSON: /api/runs</a></p>
"""
    return INDEX_TEMPLATE.format(body=body)


def _render_step(root: Path, step: str) -> str:
    sdir = root / step
    if not sdir.is_dir():
        body = f"<h1>{step}</h1><p class='meta'>No runs for this step yet.</p>"
        return INDEX_TEMPLATE.format(body=body)
    runs = sorted((p for p in sdir.iterdir() if p.is_dir()), reverse=True)
    rows = []
    for r in runs:
        rj = r / "run.json"
        meta = ""
        status = ""
        n = ""
        nerr = ""
        if rj.exists():
            try:
                d = json.loads(rj.read_text())
                status = d.get("status", "")
                n = d.get("n_components", 0)
                nerr = d.get("n_errors", 0)
                meta = d.get("started_at", "")
            except Exception:
                meta = "(unreadable)"
        cls = "err" if status == "aborted" else "ok"
        rows.append(
            f"<tr class='{cls}'><td><a href=runs/{step}/{r.name}/index.html><code>{r.name}</code></a></td>"
            f"<td>{status}</td><td>{n}</td><td>{nerr}</td><td>{meta}</td></tr>"
        )
    body = f"""
<h1>step: <code>{step}</code></h1>
<p><a href="../../index.html">← all steps</a></p>
<table>
  <tr><th>Run id</th><th>Status</th><th>#comp</th><th>#err</th><th>Started</th></tr>
  {''.join(rows) or '<tr><td colspan=5><i>none</i></td></tr>'}
</table>
"""
    return INDEX_TEMPLATE.format(body=body)


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class ReviewHandler(BaseHTTPRequestHandler):
    root: Path = DEFAULT_ROOT  # set by main()

    # ---- quiet logging ----
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003, A002
        sys.stderr.write(f"[review] {self.address_string()} {format % args}\n")

    # ---- routing ----
    def do_GET(self) -> None:  # noqa: N802
        try:
            self._handle()
        except BrokenPipeError:
            pass
        except Exception as e:  # last-ditch
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))

    def _handle(self) -> None:
        url = urlparse(self.path)
        path = unquote(url.path)
        # strip leading slashes
        rel = path.lstrip("/")
        # API routes
        if rel == "api/runs":
            return self._json_runs()
        if rel.startswith("api/runs/"):
            parts = rel.split("/")
            # api/runs/<step>/<run_id>
            if len(parts) == 4:
                _, _, step, run_id = parts
                return self._json_run(step, run_id)
        if rel.startswith("api/components/"):
            parts = rel.split("/")
            # api/components/<step>/<run>/<cid>
            if len(parts) == 5:
                _, _, step, run_id, cid = parts
                return self._json_component(step, run_id, cid)
        # Page routes
        if rel in ("", "index.html"):
            body = _render_root(self.root).encode()
            return self._send(200, "text/html; charset=utf-8", body)
        if rel == "runs" or rel == "runs/index.html":
            body = _render_root(self.root).encode()
            return self._send(200, "text/html; charset=utf-8", body)
        if rel.startswith("runs/"):
            parts = rel.split("/")
            # runs/<step>     (parts == 2)            or  runs/<step>/  (parts == 3 with empty)
            if len(parts) == 2 or (len(parts) == 3 and parts[2] == ""):
                step = parts[1]
                target = self.root / step / "index.html"
                if target.is_file():
                    return self._send_file(target)
                return self._send(200, "text/html; charset=utf-8",
                                  _render_step(self.root, step).encode())
            # runs/<step>/<run_id>  (parts == 3)  or  runs/<step>/<run_id>/  (parts == 4)
            if len(parts) in (3, 4) and parts[2] != "":
                step, run_id = parts[1], parts[2]
                target = self.root / step / run_id / "index.html"
                if target.is_file():
                    return self._send_file(target)
                # if no index.html, synthesize a placeholder
                return self._send(404, "text/html; charset=utf-8",
                                  f"<h1>404</h1><p>No run at {step}/{run_id}</p>".encode())
        # Direct file lookup: the URL path is virtual (starts with "runs/"),
        # so strip that prefix before mapping to <root>/<step>/<run_id>/...
        file_rel = rel[len("runs/"):] if rel.startswith("runs/") else rel
        candidate = (self.root / file_rel).resolve()
        try:
            candidate.relative_to(self.root.resolve())
        except ValueError:
            return self._send(403, "text/plain", b"forbidden")
        if candidate.is_file():
            return self._send_file(candidate)
        return self._send(404, "text/html; charset=utf-8",
                          f"<h1>404</h1><p>not found: {rel}</p>".encode())

    # ---- response helpers ----
    def _send(self, code: int, ctype: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        ctype, _ = mimetypes.guess_type(str(path))
        ctype = ctype or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    # ---- API responses ----
    def _json_runs(self) -> None:
        if not self.root.exists():
            return self._send(200, "application/json", b'{"steps":{}}')
        out: dict = {"steps": {}}
        for s in sorted(p for p in self.root.iterdir() if p.is_dir()):
            runs = []
            for r in sorted((p for p in s.iterdir() if p.is_dir()), reverse=True):
                rj = r / "run.json"
                entry: dict = {"run_id": r.name, "url": f"runs/{s.name}/{r.name}/index.html"}
                if rj.exists():
                    try:
                        d = json.loads(rj.read_text())
                        entry["status"] = d.get("status")
                        entry["started_at"] = d.get("started_at")
                        entry["n_components"] = d.get("n_components")
                        entry["n_errors"] = d.get("n_errors")
                    except Exception:
                        entry["status"] = "corrupt"
                runs.append(entry)
            out["steps"][s.name] = runs
        self._send(200, "application/json", json.dumps(out, indent=2).encode())

    def _json_run(self, step: str, run_id: str) -> None:
        rj = self.root / step / run_id / "run.json"
        if not rj.is_file():
            return self._send(404, "application/json", b'{"error":"not found"}')
        try:
            data = json.loads(rj.read_text())
        except Exception as e:
            return self._send(500, "application/json", json.dumps({"error": str(e)}).encode())
        self._send(200, "application/json", json.dumps(data, indent=2).encode())

    def _json_component(self, step: str, run_id: str, cid: str) -> None:
        mj = self.root / step / run_id / "components" / cid / "meta.json"
        if not mj.is_file():
            return self._send(404, "application/json", b'{"error":"not found"}')
        try:
            data = json.loads(mj.read_text())
        except Exception as e:
            return self._send(500, "application/json", json.dumps({"error": str(e)}).encode())
        self._send(200, "application/json", json.dumps(data, indent=2).encode())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "")
    p.add_argument("--host", default="127.0.0.1", help="Bind host (default 127.0.0.1)")
    p.add_argument("--port", type=int, default=8765, help="Bind port (default 8765)")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                   help=f"Runs root directory (default {DEFAULT_ROOT})")
    args = p.parse_args()

    if not args.root.is_dir():
        # don't fail — the user might start the server before any runs exist
        args.root.mkdir(parents=True, exist_ok=True)

    ReviewHandler.root = args.root
    server = ThreadingHTTPServer((args.host, args.port), ReviewHandler)
    print(f"[review] serving {args.root} on http://{args.host}:{args.port}/", file=sys.stderr)
    print(f"[review] open: http://{args.host}:{args.port}/runs/", file=sys.stderr)
    print(f"[review] press Ctrl-C to stop", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[review] shutting down", file=sys.stderr)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
