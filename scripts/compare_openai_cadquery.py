#!/usr/bin/env python3
"""Compare OpenAI models on component-to-CadQuery generation."""

from __future__ import annotations

import argparse
import ast
import base64
import csv
import importlib.util
import json
import mimetypes
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


MODELS = ("gpt-5.4-nano", "gpt-5.4-mini")
PRICES_PER_MTOK = {
    "gpt-5.4-nano": {"input": 0.20, "output": 1.25},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
}


def load_etc_var(*names: str) -> str:
    values = {}
    path = Path("/etc/environment")
    if path.exists():
        for line in path.read_text(errors="ignore").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            values[key.strip()] = value.strip().strip("\"'")
    for name in names:
        if values.get(name):
            return values[name]
    return ""


def load_converter() -> Any:
    module_path = Path("scripts/convert_downloads.py")
    spec = importlib.util.spec_from_file_location("convert_downloads", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["convert_downloads"] = module
    spec.loader.exec_module(module)
    return module


def select_image(component_dir: Path, converter: Any) -> dict[str, str] | None:
    drawing_dir = component_dir / "drawings"
    candidates = []
    for path in sorted(drawing_dir.glob("*")):
        if not path.is_file() or path.name.lower() == "product_photo.jpg":
            continue
        size = converter.image_size(path)
        if size and size[0] <= 32 and size[1] <= 32:
            continue
        priority = 0
        name = path.name.lower()
        if name.startswith("drw"):
            priority = 4
        elif name.startswith("alt"):
            priority = 3
        elif "spec" in name or "table" in name:
            priority = 2
        elif name.startswith("oth"):
            priority = 1
        area = size[0] * size[1] if size else 0
        candidates.append((priority, area, path))
    if not candidates:
        return None
    path = sorted(candidates, reverse=True)[0][2]
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"path": str(path), "url": f"data:{mime};base64,{encoded}"}


def build_context(component_id: str, converter: Any, downloads: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    component_dir = downloads / component_id
    component, review, evidence = converter.build_component(component_dir, None)
    specs, _ = converter.choose_specs(component_dir)
    tables = specs.get("tables") or []
    context = {
        "component_id": component_id,
        "name": component.get("name"),
        "category": (component.get("attributes") or {}).get("category", ""),
        "category_code": component.get("category_code"),
        "values": component.get("values") or {},
        "drawings": component.get("drawings") or [],
        "review_flags": review.flags,
        "missing": review.missing,
        "table_excerpts": converter.excerpt_tables(tables, max_tables=6, max_rows=12, max_cells=16),
        "materials": evidence.get("materials", [])[:8],
        "finishes": evidence.get("finishes", [])[:8],
        "gdt_lines": evidence.get("gdt_lines", [])[:8],
    }
    component["cadquery_script"] = ""
    return component, context


def make_prompt(context: dict[str, Any]) -> str:
    return (
        "Return JSON only with keys cadquery_script, review_flags, unknowns, modeling_notes.\n"
        "Generate a conservative executable CadQuery 2.x Python script for this MISUMI component.\n"
        "Requirements:\n"
        "- import cadquery as cq\n"
        "- define numeric variables at the top\n"
        "- use only dimensions supported by context; use TODO comments for placeholders\n"
        "- assign final geometry to result as cq.Workplane or cq.Assembly\n"
        "- avoid unsupported CadQuery APIs, cq.math, named colors, and show_object calls\n"
        "- keep the script executable in headless CadQuery 2.x\n"
        "- preserve uncertainty in review_flags instead of inventing manufacturing-critical dimensions\n"
        "Context JSON:\n"
        + json.dumps(context, ensure_ascii=False, indent=2)[:28000]
    )


def call_openai(model: str, prompt: str, image: dict[str, str] | None, api_key: str) -> tuple[dict[str, Any], dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    if image:
        content.append({"type": "image_url", "image_url": {"url": image["url"], "detail": "low"}})
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "response_format": {"type": "json_object"},
        "temperature": 0,
        "max_completion_tokens": 4096,
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        raw = json.loads(response.read().decode("utf-8"))
    content_text = raw["choices"][0]["message"].get("content") or "{}"
    return json.loads(content_text), raw.get("usage") or {}


def syntax_ok(script: str, path: Path) -> tuple[bool, str]:
    try:
        ast.parse(script, filename=str(path))
        return True, ""
    except SyntaxError as exc:
        return False, str(exc)


def run_cadquery(script_path: Path) -> tuple[bool, dict[str, Any]]:
    checker = f"""
import json
import cadquery as cq
ns={{'__name__':'__cadquery_check__'}}
exec(compile(open({str(script_path)!r}, encoding='utf-8').read(), {str(script_path)!r}, 'exec'), ns)
result=ns.get('result')
if result is None:
    raise RuntimeError('missing result')
out={{'type': type(result).__name__}}
if isinstance(result, cq.Assembly):
    out['children']=len(result.children)
elif isinstance(result, cq.Workplane):
    solids=result.solids().vals()
    out['solids']=len(solids)
    if solids:
        bb=solids[0].BoundingBox()
        out['bbox']=[round(bb.xlen,3), round(bb.ylen,3), round(bb.zlen,3)]
print(json.dumps(out, sort_keys=True))
"""
    proc = subprocess.run(
        [".venv/bin/python", "-c", checker],
        check=False,
        text=True,
        capture_output=True,
        timeout=30,
    )
    if proc.returncode != 0:
        return False, {"error": (proc.stderr or proc.stdout).strip()[-1200:]}
    return True, json.loads(proc.stdout.strip().splitlines()[-1])


def validate_schema(output_dir: Path) -> bool:
    proc = subprocess.run(
        ["npx", "--yes", "ajv-cli", "validate", "--spec=draft7", "-s", "component_schema.json", "-d", str(output_dir / "*" / "component.json")],
        check=False,
        text=True,
        capture_output=True,
        timeout=60,
    )
    return proc.returncode == 0


def estimate_cost(model: str, usage: dict[str, Any]) -> float:
    prices = PRICES_PER_MTOK[model]
    input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    output_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0
    return input_tokens / 1_000_000 * prices["input"] + output_tokens / 1_000_000 * prices["output"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-id", action="append", required=True)
    parser.add_argument("--model", action="append", choices=MODELS, help="Model to run. Repeatable. Defaults to both comparison models.")
    parser.add_argument("--downloads", type=Path, default=Path("downloads"))
    parser.add_argument("--output", type=Path, default=Path("outputs_openai_compare_10"))
    parser.add_argument("--flat-output", action="store_true", help="Write output/<component_id> instead of output/<model>/<component_id>. Intended for single-model runs.")
    args = parser.parse_args()

    api_key = load_etc_var("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not found in /etc/environment")

    converter = load_converter()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    models = tuple(args.model or MODELS)
    if args.flat_output and len(models) != 1:
        raise SystemExit("--flat-output requires exactly one --model")
    for model in models:
        model_dir = args.output if args.flat_output else args.output / model
        model_dir.mkdir(parents=True, exist_ok=True)
        for component_id in args.component_id:
            component_dir = model_dir / component_id
            component_dir.mkdir(parents=True, exist_ok=True)
            component, context = build_context(component_id, converter, args.downloads)
            image = select_image(args.downloads / component_id, converter)
            prompt = make_prompt(context)
            row = {"model": model, "component_id": component_id, "image": image["path"] if image else ""}
            try:
                data, usage = call_openai(model, prompt, image, api_key)
                script = converter.normalize_cadquery_script(data.get("cadquery_script"))
                component["cadquery_script"] = script
                (component_dir / "component.json").write_text(
                    json.dumps(component, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                (component_dir / "cadquery_script.py").write_text(script, encoding="utf-8")
                (component_dir / "openai_result.json").write_text(
                    json.dumps({"response": data, "usage": usage}, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                ok_syntax, syntax_error = syntax_ok(script, component_dir / "cadquery_script.py")
                ok_cq, cq_result = run_cadquery(component_dir / "cadquery_script.py") if ok_syntax and script.strip() else (False, {"error": "syntax_failed_or_empty"})
                row.update(
                    {
                        "ok_api": True,
                        "script_len": len(script),
                        "syntax_ok": ok_syntax,
                        "syntax_error": syntax_error,
                        "cadquery_ok": ok_cq,
                        "cadquery_result": json.dumps(cq_result, sort_keys=True),
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                        "estimated_cost_usd": round(estimate_cost(model, usage), 6),
                        "review_flags": len(data.get("review_flags") or []),
                        "unknowns": len(data.get("unknowns") or []),
                    }
                )
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, RuntimeError, subprocess.TimeoutExpired) as exc:
                row.update({"ok_api": False, "error": repr(exc)})
                (component_dir / "error.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    fieldnames = sorted({key for row in rows for key in row})
    with (args.output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for model in models:
        model_rows = [row for row in rows if row["model"] == model]
        summary[model] = {
            "requests": len(model_rows),
            "api_ok": sum(bool(row.get("ok_api")) for row in model_rows),
            "syntax_ok": sum(bool(row.get("syntax_ok")) for row in model_rows),
            "cadquery_ok": sum(bool(row.get("cadquery_ok")) for row in model_rows),
            "prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in model_rows),
            "completion_tokens": sum(int(row.get("completion_tokens") or 0) for row in model_rows),
            "total_tokens": sum(int(row.get("total_tokens") or 0) for row in model_rows),
            "estimated_cost_usd": round(sum(float(row.get("estimated_cost_usd") or 0) for row in model_rows), 6),
            "batch_estimated_cost_usd": round(sum(float(row.get("estimated_cost_usd") or 0) for row in model_rows) * 0.5, 6),
        }
        summary[model]["schema_ok"] = validate_schema(args.output if args.flat_output else args.output / model)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
