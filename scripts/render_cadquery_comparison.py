#!/usr/bin/env python3
"""Render generated CadQuery outputs to SVG comparison sheets."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import traceback
from pathlib import Path
from typing import Any

import cadquery as cq
from cadquery import exporters

IMAGE_SUFFIXES = {".gif", ".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def source_component_id(component: dict[str, Any], fallback: str) -> str:
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


def select_source_images(downloads: Path, source_id: str, max_images: int) -> list[Path]:
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


def execute_script(script_path: Path) -> Any:
    ns = {"__name__": "__cadquery_render__", "cq": cq}
    exec(compile(script_path.read_text(encoding="utf-8"), str(script_path), "exec"), ns)
    result = ns.get("result")
    if result is None:
        raise RuntimeError("missing result")
    return result


def renderable_shape(result: Any) -> Any:
    if isinstance(result, cq.Assembly):
        return result.toCompound()
    if isinstance(result, cq.Workplane):
        vals = result.vals()
        if not vals:
            raise RuntimeError("empty Workplane result")
        return result.val()
    if isinstance(result, cq.Shape):
        return result
    raise RuntimeError(f"unsupported result type: {type(result).__name__}")


def render_views(script_path: Path, out_dir: Path) -> tuple[dict[str, Path], str]:
    views = {
        "front": (0, -1, 0),
        "top": (0, 0, 1),
        "right": (1, 0, 0),
        "isometric": (1, -1, 0.75),
    }
    try:
        shape = renderable_shape(execute_script(script_path))
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = {}
        for name, projection in views.items():
            svg = exporters.getSVG(
                shape,
                {
                    "width": 900,
                    "height": 650,
                    "marginLeft": 20,
                    "marginTop": 20,
                    "projectionDir": projection,
                    "showHidden": True,
                    "strokeWidth": 1.4,
                    "strokeColor": (20, 80, 140),
                    "hiddenColor": (170, 170, 170),
                },
            )
            path = out_dir / f"{name}.svg"
            path.write_text(svg, encoding="utf-8")
            paths[name] = path
        return paths, ""
    except Exception:  # noqa: BLE001 - report render failures in comparison sheet.
        return {}, traceback.format_exc()[-3000:]


def rel(path: Path, base: Path) -> str:
    return html.escape(os.path.relpath(path.resolve(), base.resolve()))


def write_component_page(
    run_dir: Path,
    component_dir: Path,
    component: dict[str, Any],
    source_images: list[Path],
    view_paths: dict[str, Path],
    render_error: str,
) -> Path:
    page = component_dir / "comparison.html"
    title = f"{component_dir.name} - {component.get('name', '')}"
    source_html = "\n".join(
        f'<figure><img src="{rel(path, component_dir)}"><figcaption>{html.escape(path.name)}</figcaption></figure>'
        for path in source_images
    ) or "<p>No source drawings found.</p>"
    cad_html = "\n".join(
        f'<figure><img src="{rel(path, component_dir)}"><figcaption>{html.escape(name)}</figcaption></figure>'
        for name, path in view_paths.items()
    ) or f"<pre>{html.escape(render_error or 'No CAD views rendered.')}</pre>"
    qc_path = component_dir / "qc.json"
    qc = read_json(qc_path) if qc_path.exists() else {}
    vision = qc.get("drawing_match") or {}
    page.write_text(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #172033; }}
    h1 {{ font-size: 22px; margin-bottom: 4px; }}
    h2 {{ font-size: 16px; margin-top: 24px; }}
    .meta {{ color: #5b6472; margin-bottom: 18px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 18px; align-items: start; }}
    figure {{ margin: 0; border: 1px solid #d7dce3; padding: 10px; background: #fff; }}
    img {{ max-width: 100%; height: auto; display: block; }}
    figcaption {{ margin-top: 8px; font-size: 13px; color: #475163; }}
    pre {{ white-space: pre-wrap; background: #f5f7fa; padding: 12px; border: 1px solid #d7dce3; }}
  </style>
</head>
<body>
  <h1>{html.escape(component_dir.name)}</h1>
  <div class="meta">{html.escape(str(component.get('name') or ''))}</div>
  <p><strong>QC:</strong> {html.escape(str(qc.get('overall_status', 'not run')))}
     <strong>Vision:</strong> {html.escape(str(vision.get('status', '')))} / {html.escape(str(vision.get('verdict', '')))}
     <strong>Confidence:</strong> {html.escape(str(vision.get('confidence', '')))}</p>
  <h2>Engineering Drawing</h2>
  <div class="grid">{source_html}</div>
  <h2>CadQuery SVG Views</h2>
  <div class="grid">{cad_html}</div>
</body>
</html>
""",
        encoding="utf-8",
    )
    return page


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--downloads", type=Path, default=Path("downloads"))
    parser.add_argument("--max-source-images", type=int, default=2)
    args = parser.parse_args()

    pages = []
    for component_dir in sorted(path for path in args.run_dir.iterdir() if path.is_dir()):
        component_path = component_dir / "component.json"
        script_path = component_dir / "cadquery_script.py"
        if not component_path.exists() or not script_path.exists():
            continue
        component = read_json(component_path)
        source_id = source_component_id(component, component_dir.name)
        source_images = select_source_images(args.downloads, source_id, args.max_source_images)
        views_dir = component_dir / "cad_svg_views"
        view_paths, render_error = render_views(script_path, views_dir)
        pages.append(write_component_page(args.run_dir, component_dir, component, source_images, view_paths, render_error))

    index_links = "\n".join(f'<li><a href="{html.escape(str(page.relative_to(args.run_dir)))}">{html.escape(page.parent.name)}</a></li>' for page in pages)
    (args.run_dir / "comparison_index.html").write_text(
        f"<!doctype html><html><head><meta charset='utf-8'><title>CadQuery Comparison</title></head><body><h1>CadQuery Comparison</h1><ul>{index_links}</ul></body></html>\n",
        encoding="utf-8",
    )
    print(json.dumps({"components": len(pages), "index": str(args.run_dir / "comparison_index.html")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
