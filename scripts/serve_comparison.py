#!/usr/bin/env python3
"""Serve a lightweight frontend for CadQuery drawing comparisons."""

from __future__ import annotations

import argparse
import html
import json
import mimetypes
import os
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

IMAGE_SUFFIXES = {".gif", ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".svg"}
ROOT = Path.cwd().resolve()


def read_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def safe_rel(path: Path) -> str:
    return os.path.relpath(path.resolve(), ROOT)


def asset_url(path: Path) -> str:
    return "/asset?path=" + quote(safe_rel(path))


def source_component_id(component: dict, fallback: str) -> str:
    attrs = component.get("attributes") or {}
    for value in (attrs.get("series_code"), attrs.get("component_id"), attrs.get("misumi_id")):
        if isinstance(value, str) and re.fullmatch(r"\d{6,}", value):
            return value
    source_url = attrs.get("source_url")
    if isinstance(source_url, str):
        match = re.search(r"/detail/(\d{6,})/", source_url)
        if match:
            return match.group(1)
    matches = re.findall(r"\d{6,}", fallback)
    return matches[-1] if matches else fallback


def select_source_images(downloads: Path, source_id: str, max_images: int = 3) -> list[Path]:
    drawing_dir = downloads / source_id / "drawings"
    if not drawing_dir.exists():
        return []
    candidates = []
    for path in sorted(drawing_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES or path.name.lower() == "product_photo.jpg":
            continue
        name = path.name.lower()
        priority = 0
        if name.startswith("drw"):
            priority = 5
        elif name.startswith("alt"):
            priority = 4
        elif "spec" in name or "table" in name:
            priority = 3
        elif name.startswith("oth"):
            priority = 2
        candidates.append((priority, path.stat().st_size, path))
    return [path for _priority, _size, path in sorted(candidates, reverse=True)[:max_images]]


def discover_runs(outputs: Path) -> list[Path]:
    if not outputs.exists():
        return []
    return sorted([path for path in outputs.iterdir() if path.is_dir()])


def discover_components(run_dir: Path) -> list[Path]:
    return sorted([path for path in run_dir.iterdir() if path.is_dir() and (path / "component.json").exists()])


def badge(text: str, kind: str = "") -> str:
    return f'<span class="badge {html.escape(kind)}">{html.escape(text)}</span>'


def page_shell(title: str, body: str) -> bytes:
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #16202f;
      --muted: #667085;
      --line: #d8dde6;
      --accent: #116a7b;
      --fail: #b42318;
      --review: #9a5b00;
      --pass: #087443;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; background: var(--bg); color: var(--ink); }}
    header {{ display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 14px 20px; border-bottom: 1px solid var(--line); background: #fff; position: sticky; top: 0; z-index: 2; }}
    header h1 {{ font-size: 18px; margin: 0; letter-spacing: 0; }}
    main {{ padding: 18px 20px 28px; }}
    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .toolbar {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
    select, input {{ height: 34px; border: 1px solid var(--line); border-radius: 6px; padding: 0 10px; background: #fff; color: var(--ink); }}
    button, .button {{ height: 34px; display: inline-flex; align-items: center; gap: 6px; border: 1px solid var(--line); border-radius: 6px; padding: 0 10px; background: #fff; color: var(--ink); cursor: pointer; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid var(--line); }}
    th, td {{ padding: 9px 10px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; font-size: 14px; }}
    th {{ font-size: 12px; text-transform: uppercase; color: var(--muted); background: #fbfcfd; }}
    tr:hover td {{ background: #f9fbfc; }}
    .badge {{ display: inline-flex; align-items: center; min-height: 22px; padding: 2px 8px; border-radius: 999px; font-size: 12px; border: 1px solid var(--line); color: var(--muted); }}
    .badge.pass {{ color: var(--pass); border-color: #9bd7bd; background: #ecfdf3; }}
    .badge.review {{ color: var(--review); border-color: #f2c572; background: #fff8e8; }}
    .badge.fail {{ color: var(--fail); border-color: #f5b5ae; background: #fff1f0; }}
    .layout {{ display: grid; grid-template-columns: 320px 1fr; gap: 16px; align-items: start; }}
    .side {{ background: #fff; border: 1px solid var(--line); padding: 12px; max-height: calc(100vh - 90px); overflow: auto; }}
    .component-link {{ display: flex; justify-content: space-between; gap: 10px; padding: 8px; border-radius: 6px; }}
    .component-link.active, .component-link:hover {{ background: #edf7f9; text-decoration: none; }}
    .compare {{ display: grid; grid-template-columns: minmax(360px, 1fr) minmax(360px, 1fr); gap: 16px; }}
    .section {{ background: #fff; border: 1px solid var(--line); padding: 12px; }}
    .section h2 {{ margin: 0 0 10px; font-size: 15px; }}
    .media-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }}
    figure {{ margin: 0; border: 1px solid var(--line); background: #fff; }}
    figure img {{ width: 100%; height: auto; display: block; }}
    figcaption {{ padding: 8px 9px; color: var(--muted); font-size: 12px; border-top: 1px solid var(--line); }}
    pre {{ margin: 0; white-space: pre-wrap; background: #f3f5f7; border: 1px solid var(--line); padding: 10px; overflow: auto; }}
    .muted {{ color: var(--muted); }}
    @media (max-width: 980px) {{ .layout, .compare {{ grid-template-columns: 1fr; }} header {{ position: static; }} }}
  </style>
</head>
<body>
{body}
</body>
</html>
""".encode("utf-8")


class ComparisonHandler(BaseHTTPRequestHandler):
    outputs: Path
    downloads: Path

    def send_html(self, title: str, body: str) -> None:
        payload = page_shell(title, body)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def send_not_found(self, message: str = "Not found") -> None:
        payload = message.encode("utf-8")
        self.send_response(HTTPStatus.NOT_FOUND)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API.
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if parsed.path == "/asset":
            self.serve_asset(query.get("path", [""])[0])
            return
        if parsed.path == "/":
            self.serve_index(query)
            return
        self.send_not_found()

    def serve_asset(self, raw_path: str) -> None:
        if not raw_path:
            self.send_not_found()
            return
        path = (ROOT / unquote(raw_path)).resolve()
        allowed = [self.outputs.resolve(), self.downloads.resolve()]
        if not any(path == base or base in path.parents for base in allowed):
            self.send_not_found()
            return
        if not path.exists() or not path.is_file():
            self.send_not_found()
            return
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        payload = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def serve_index(self, query: dict[str, list[str]]) -> None:
        runs = discover_runs(self.outputs)
        if not runs:
            self.send_html("CadQuery Comparison", "<header><h1>CadQuery Comparison</h1></header><main>No runs found.</main>")
            return
        run_name = query.get("run", [runs[-1].name])[0]
        run_dir = next((path for path in runs if path.name == run_name), runs[-1])
        components = discover_components(run_dir)
        component_name = query.get("component", [components[0].name if components else ""])[0]
        component_dir = next((path for path in components if path.name == component_name), components[0] if components else None)
        run_options = "\n".join(
            f'<option value="{html.escape(run.name)}" {"selected" if run == run_dir else ""}>{html.escape(run.name)}</option>'
            for run in runs
        )
        header = f"""
<header>
  <h1>CadQuery Comparison</h1>
  <form class="toolbar" method="get">
    <label class="muted">Run</label>
    <select name="run" onchange="this.form.submit()">{run_options}</select>
  </form>
</header>
"""
        if component_dir is None:
            self.send_html("CadQuery Comparison", header + "<main>No components in selected run.</main>")
            return
        side = self.component_sidebar(run_dir, components, component_dir)
        detail = self.component_detail(run_dir, component_dir)
        self.send_html("CadQuery Comparison", header + f"<main><div class='layout'>{side}{detail}</div></main>")

    def component_sidebar(self, run_dir: Path, components: list[Path], active: Path) -> str:
        links = []
        for component_dir in components:
            qc = read_json(component_dir / "qc.json")
            status = str(qc.get("overall_status") or "unknown")
            active_class = " active" if component_dir == active else ""
            links.append(
                f'<a class="component-link{active_class}" href="/?run={quote(run_dir.name)}&component={quote(component_dir.name)}">'
                f'<span>{html.escape(component_dir.name)}</span>{badge(status, status)}</a>'
            )
        return "<aside class='side'>" + "\n".join(links) + "</aside>"

    def component_detail(self, run_dir: Path, component_dir: Path) -> str:
        component = read_json(component_dir / "component.json")
        qc = read_json(component_dir / "qc.json")
        source_id = source_component_id(component, component_dir.name)
        source_images = select_source_images(self.downloads, source_id)
        cad_svgs = sorted((component_dir / "cad_svg_views").glob("*.svg"))
        vision = qc.get("drawing_match") or {}
        reasons = qc.get("reasons") or []
        source_figs = self.figures(source_images, component_dir)
        cad_figs = self.figures(cad_svgs, component_dir)
        if not cad_figs:
            script_exec = qc.get("script_exec") or {}
            cad_figs = f"<pre>{html.escape(str(script_exec.get('error_trace') or 'No SVG views rendered.'))}</pre>"
        reason_html = "<br>".join(html.escape(str(reason)) for reason in reasons[:8]) or "<span class='muted'>No reasons recorded.</span>"
        return f"""
<section>
  <div class="section" style="margin-bottom: 16px">
    <h2>{html.escape(component_dir.name)} {badge(str(qc.get('overall_status', 'unknown')), str(qc.get('overall_status', '')))}</h2>
    <div class="muted">{html.escape(str(component.get('name') or ''))}</div>
    <p><strong>Vision:</strong> {html.escape(str(vision.get('status', '')))} / {html.escape(str(vision.get('verdict', '')))}
       <strong>Confidence:</strong> {html.escape(str(vision.get('confidence', '')))}</p>
    <p>{reason_html}</p>
  </div>
  <div class="compare">
    <div class="section"><h2>Engineering Drawing</h2><div class="media-grid">{source_figs}</div></div>
    <div class="section"><h2>CadQuery SVG Views</h2><div class="media-grid">{cad_figs}</div></div>
  </div>
</section>
"""

    def figures(self, paths: list[Path], base: Path) -> str:
        if not paths:
            return "<p class='muted'>No images found.</p>"
        parts = []
        for path in paths:
            parts.append(
                f'<figure><img src="{asset_url(path)}"><figcaption>{html.escape(path.name)}</figcaption></figure>'
            )
        return "\n".join(parts)

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"{self.address_string()} - {fmt % args}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--downloads", type=Path, default=Path("downloads"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    handler = type("BoundComparisonHandler", (ComparisonHandler,), {"outputs": args.outputs, "downloads": args.downloads})
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"http://{args.host}:{args.port}/")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
