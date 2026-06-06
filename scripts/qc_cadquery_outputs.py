#!/usr/bin/env python3
"""QC generated CadQuery scripts against source component evidence."""

from __future__ import annotations

import argparse
import ast
import base64
import csv
import json
import math
import mimetypes
import os
import re
import subprocess
import sys
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from PIL import Image, ImageDraw, ImageFont
except ModuleNotFoundError:  # pragma: no cover - handled at runtime.
    Image = None
    ImageDraw = None
    ImageFont = None


STATUS_PASS = "pass"
STATUS_REVIEW = "review"
STATUS_FAIL = "fail"

PLACEHOLDER_RE = re.compile(r"\b(todo|placeholder|assum(?:e|ed|ption)|verify|replace with|unknown|tbd)\b", re.I)
RANGE_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*(?:~|-|to)\s*(-?\d+(?:\.\d+)?)\s*$", re.I)
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
IMAGE_SUFFIXES = {".gif", ".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class ComponentInput:
    component_id: str
    component_dir: Path
    component_json: Path
    script_path: Path
    relative_key: str


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


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
        if os.environ.get(name):
            return os.environ[name]
        if values.get(name):
            return values[name]
    return ""


def discover_components(outputs: Path) -> list[ComponentInput]:
    found: list[ComponentInput] = []
    for component_json in sorted(outputs.rglob("component.json")):
        component_dir = component_json.parent
        script_path = component_dir / "cadquery_script.py"
        if not script_path.exists():
            continue
        component_id = component_dir.name
        try:
            relative_key = str(component_dir.relative_to(outputs))
        except ValueError:
            relative_key = component_id
        found.append(ComponentInput(component_id, component_dir, component_json, script_path, relative_key))
    return found


def validate_component_schema(component: dict[str, Any], schema_path: Path) -> dict[str, Any]:
    errors = []
    for key in ("name", "category_code", "attributes"):
        if key not in component:
            errors.append(f"missing required field: {key}")
    if not isinstance(component.get("attributes"), dict):
        errors.append("attributes must be an object")
    try:
        import jsonschema  # type: ignore

        schema = read_json(schema_path)
        jsonschema.validate(component, schema)
        schema_checked = True
    except ModuleNotFoundError:
        schema_checked = False
    except Exception as exc:  # noqa: BLE001 - report validation failures in scorecard.
        schema_checked = True
        errors.append(f"json_schema_validation_failed: {exc}")
    return {"status": STATUS_PASS if not errors else STATUS_FAIL, "schema_checked": schema_checked, "errors": errors}


def call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def static_script_checks(script: str, script_path: Path) -> dict[str, Any]:
    errors = []
    warnings = []
    result_assignments = 0
    result_none_assignments = 0
    try:
        tree = ast.parse(script, filename=str(script_path))
    except SyntaxError as exc:
        return {"status": STATUS_FAIL, "errors": [f"syntax_error: {exc}"], "warnings": warnings}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = call_name(node.func)
            if name == "show_object":
                errors.append("show_object call is not allowed in headless QC")
            if name == "cq.Color" and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                errors.append(f"unsupported named color: {node.args[0].value}")
        if isinstance(node, ast.Attribute) and call_name(node) == "cq.math":
            errors.append("cq.math is unsupported")
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = []
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
                value = node.value
            else:
                targets = [node.target]
                value = node.value
            if any(isinstance(target, ast.Name) and target.id == "result" for target in targets):
                result_assignments += 1
                if isinstance(value, ast.Constant) and value.value is None:
                    result_none_assignments += 1
    if not script.strip():
        errors.append("empty CadQuery script")
    if result_assignments == 0:
        errors.append("script does not assign result")
    if result_none_assignments and result_none_assignments == result_assignments:
        errors.append("result is only assigned None")
    if PLACEHOLDER_RE.search(script):
        warnings.append("script contains placeholder/TODO/assumption language")
    return {"status": STATUS_FAIL if errors else STATUS_PASS, "errors": errors, "warnings": warnings}


def run_cadquery(script_path: Path, python_bin: Path, timeout: int) -> dict[str, Any]:
    checker = f"""
import json
import traceback
import cadquery as cq

def bbox_list(shape):
    bb = shape.BoundingBox()
    return {{
        "xlen": round(float(bb.xlen), 6),
        "ylen": round(float(bb.ylen), 6),
        "zlen": round(float(bb.zlen), 6),
        "xmin": round(float(bb.xmin), 6),
        "xmax": round(float(bb.xmax), 6),
        "ymin": round(float(bb.ymin), 6),
        "ymax": round(float(bb.ymax), 6),
        "zmin": round(float(bb.zmin), 6),
        "zmax": round(float(bb.zmax), 6),
    }}

def shape_metrics(shape):
    out = {{}}
    try:
        out["bbox"] = bbox_list(shape)
    except Exception as exc:
        out["bbox_error"] = repr(exc)
    try:
        out["volume"] = round(float(shape.Volume()), 6)
    except Exception as exc:
        out["volume_error"] = repr(exc)
    return out

try:
    ns = {{"__name__": "__cadquery_qc__"}}
    with open({str(script_path)!r}, encoding="utf-8") as handle:
        exec(compile(handle.read(), {str(script_path)!r}, "exec"), ns)
    result = ns.get("result")
    if result is None:
        raise RuntimeError("missing result")
    out = {{"pass": True, "type": type(result).__name__}}
    if isinstance(result, cq.Assembly):
        out["assembly_children"] = len(result.children)
        compound = result.toCompound()
        out.update(shape_metrics(compound))
        out["solids_count"] = len(compound.Solids())
    elif isinstance(result, cq.Workplane):
        solids = result.solids().vals()
        out["solids_count"] = len(solids)
        vals = result.vals()
        shape = result.val() if vals else None
        if shape is not None:
            out.update(shape_metrics(shape))
    elif isinstance(result, cq.Shape):
        out["solids_count"] = len(result.Solids())
        out.update(shape_metrics(result))
    else:
        raise RuntimeError(f"unsupported result type: {{type(result).__name__}}")
    print(json.dumps(out, sort_keys=True))
except Exception:
    print(json.dumps({{"pass": False, "error_trace": traceback.format_exc()[-4000:]}}, sort_keys=True))
"""
    try:
        proc = subprocess.run(
            [str(python_bin), "-c", checker],
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return {"pass": False, "error_trace": f"cadquery execution timed out after {timeout}s: {exc}"}
    if proc.returncode != 0:
        return {"pass": False, "error_trace": (proc.stderr or proc.stdout).strip()[-4000:]}
    try:
        result = json.loads(proc.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return {"pass": False, "error_trace": (proc.stderr + proc.stdout).strip()[-4000:]}
    return result


def cad_metrics_status(metrics: dict[str, Any]) -> tuple[str, list[str]]:
    reasons = []
    if not metrics.get("pass"):
        return STATUS_FAIL, ["script execution failed"]
    bbox = metrics.get("bbox") or {}
    axes = [bbox.get("xlen"), bbox.get("ylen"), bbox.get("zlen")]
    if not all(isinstance(axis, (int, float)) and math.isfinite(axis) for axis in axes):
        reasons.append("missing finite bounding box")
    elif any(abs(axis) < 1e-6 for axis in axes):
        reasons.append("zero or near-zero bounding box axis")
    if int(metrics.get("solids_count") or 0) <= 0:
        reasons.append("no solids detected")
    volume = metrics.get("volume")
    if isinstance(volume, (int, float)) and volume <= 0:
        reasons.append("non-positive volume")
    return (STATUS_FAIL if reasons else STATUS_PASS), reasons


def parse_numeric_value(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, (int, float)) and math.isfinite(raw):
        return {"kind": "number", "value": float(raw)}
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    match = RANGE_RE.match(text)
    if match:
        lo = float(match.group(1))
        hi = float(match.group(2))
        return {"kind": "range", "min": min(lo, hi), "max": max(lo, hi)}
    nums = NUMBER_RE.findall(text)
    if len(nums) == 1:
        return {"kind": "number", "value": float(nums[0])}
    return None


def extract_source_metrics(component: dict[str, Any]) -> dict[str, Any]:
    metrics = {"values": {}, "placeholders": [], "geometry_types": [], "review_flags": []}
    for key, entry in (component.get("values") or {}).items():
        value = entry.get("value") if isinstance(entry, dict) else entry
        parsed = parse_numeric_value(value)
        unit = entry.get("unit") if isinstance(entry, dict) else ""
        if parsed and (not unit or str(unit).lower() in {"mm", "millimeter", "millimeters", "text"}):
            metrics["values"][key] = {"raw": value, "unit": unit, **parsed}
        elif isinstance(value, str) and PLACEHOLDER_RE.search(value):
            metrics["placeholders"].append(key)
    for drawing in component.get("drawings") or []:
        geometry_type = drawing.get("geometry_type")
        if geometry_type and geometry_type not in metrics["geometry_types"]:
            metrics["geometry_types"].append(geometry_type)
    for item in component.get("status") or []:
        if isinstance(item, str) and item.startswith("needs_review"):
            metrics["review_flags"].append(item)
    return metrics


def classify_dimension_key(key: str) -> str:
    text = key.lower()
    if any(token in text for token in ("diameter", "outer_d", "od", "bore", "hole")) or re.fullmatch(r"d\d*|dr|dh|id", text):
        return "diameter"
    if text in {"l", "length"} or "length" in text:
        return "length"
    if text in {"h", "height"} or "height" in text:
        return "height"
    if text in {"w", "width", "b"} or "width" in text:
        return "width"
    if text in {"t", "thickness"} or "thick" in text:
        return "thickness"
    return "generic"


def expected_matches_observed(expected: dict[str, Any], observed: float, tolerance_ratio: float) -> bool:
    tol = max(0.5, abs(observed) * tolerance_ratio)
    if expected["kind"] == "range":
        return expected["min"] - tol <= observed <= expected["max"] + tol
    return abs(float(expected["value"]) - observed) <= max(0.5, abs(float(expected["value"])) * tolerance_ratio)


def compare_dimensions(source_metrics: dict[str, Any], cad_metrics: dict[str, Any], tolerance_ratio: float) -> list[dict[str, Any]]:
    bbox = cad_metrics.get("bbox") or {}
    axes = {
        "x": float(bbox.get("xlen") or 0),
        "y": float(bbox.get("ylen") or 0),
        "z": float(bbox.get("zlen") or 0),
    }
    axis_values = [value for value in axes.values() if value > 0]
    checks = []
    if not axis_values:
        return checks
    for key, expected in sorted((source_metrics.get("values") or {}).items()):
        role = classify_dimension_key(key)
        observed_candidates: dict[str, float]
        if role == "length":
            max_axis = max(axes, key=axes.get)
            observed_candidates = {max_axis: axes[max_axis]}
        elif role in {"diameter", "width", "height", "thickness"}:
            observed_candidates = axes
        else:
            observed_candidates = axes
        matches = [
            {"axis": axis, "observed": value}
            for axis, value in observed_candidates.items()
            if value > 0 and expected_matches_observed(expected, value, tolerance_ratio)
        ]
        status = STATUS_PASS if matches else STATUS_REVIEW
        checks.append(
            {
                "key": key,
                "role": role,
                "expected": expected,
                "observed_candidates": observed_candidates,
                "status": status,
                "evidence": "bbox axis match" if matches else "no bbox axis matched expected source value/range",
                "matches": matches,
            }
        )
    return checks


def select_source_images(downloads: Path, component_id: str, max_images: int) -> list[Path]:
    drawing_dir = downloads / component_id / "drawings"
    candidates = []
    if not drawing_dir.exists():
        return []
    for path in sorted(drawing_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES or path.name.lower() == "product_photo.jpg":
            continue
        priority = 0
        name = path.name.lower()
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


def render_metric_views(component_id: str, cad_metrics: dict[str, Any], output_dir: Path) -> list[Path]:
    if Image is None or ImageDraw is None:
        return []
    bbox = cad_metrics.get("bbox") or {}
    dims = [float(bbox.get(axis) or 0) for axis in ("xlen", "ylen", "zlen")]
    if not all(dim > 0 for dim in dims):
        return []
    render_dir = output_dir / component_id / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    views = {
        "front": (dims[0], dims[2]),
        "top": (dims[0], dims[1]),
        "right": (dims[1], dims[2]),
        "isometric": (math.hypot(dims[0], dims[1]) * 0.85, dims[2] + min(dims[0], dims[1]) * 0.35),
    }
    paths = []
    for name, (width_mm, height_mm) in views.items():
        img = Image.new("RGB", (640, 480), "white")
        draw = ImageDraw.Draw(img)
        margin = 72
        scale = min((640 - 2 * margin) / max(width_mm, 1), (480 - 2 * margin) / max(height_mm, 1))
        w = max(4, width_mm * scale)
        h = max(4, height_mm * scale)
        x0 = (640 - w) / 2
        y0 = (480 - h) / 2
        x1 = x0 + w
        y1 = y0 + h
        draw.rectangle([x0, y0, x1, y1], outline=(20, 70, 120), width=4, fill=(222, 235, 246))
        draw.line([x0, y1 + 18, x1, y1 + 18], fill=(70, 70, 70), width=2)
        draw.line([x0, y1 + 12, x0, y1 + 24], fill=(70, 70, 70), width=2)
        draw.line([x1, y1 + 12, x1, y1 + 24], fill=(70, 70, 70), width=2)
        draw.text((24, 20), f"{component_id} CAD {name}", fill=(0, 0, 0))
        draw.text((24, 44), f"bbox X/Y/Z: {dims[0]:.3g} / {dims[1]:.3g} / {dims[2]:.3g} mm", fill=(0, 0, 0))
        draw.text((int(x0), min(456, int(y1 + 28))), f"{width_mm:.3g} mm", fill=(0, 0, 0))
        path = render_dir / f"{name}.png"
        img.save(path)
        paths.append(path)
    return paths


def image_payload(path: Path, detail: str = "low") -> dict[str, Any]:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}", "detail": detail}}


def parse_json_object(content: str) -> dict[str, Any]:
    try:
        data = json.loads(content)
        return data if isinstance(data, dict) else {"raw": data}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            return {"error": "model did not return JSON", "raw": content[:1000]}
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else {"raw": data}
        except json.JSONDecodeError:
            return {"error": "model JSON parse failed", "raw": content[:1000]}


def call_openai_drawing_match(
    model: str,
    api_key: str,
    component: dict[str, Any],
    source_images: list[Path],
    cad_images: list[Path],
    cad_metrics: dict[str, Any],
    source_metrics: dict[str, Any],
    detail: str,
) -> dict[str, Any]:
    prompt = (
        "Compare the supplier engineering drawing images against the CAD render/metric images. "
        "Return strict JSON only with keys geometry_match, verdict, mismatches, missing_features, "
        "extra_features, orientation_mismatch, dimension_label_mismatches, confidence, rationale. "
        "verdict must be pass, review, or fail. Be strict: missing flanges, hole patterns, shaft/block "
        "type mismatches, assembly/single-solid mismatches, or gross envelope errors should be fail.\n"
        "Component context:\n"
        + json.dumps(
            {
                "name": component.get("name"),
                "category": (component.get("attributes") or {}).get("category"),
                "drawing_geometry_types": source_metrics.get("geometry_types"),
                "source_values": source_metrics.get("values"),
                "cad_metrics": cad_metrics,
            },
            ensure_ascii=False,
            sort_keys=True,
        )[:12000]
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for path in source_images:
        content.append({"type": "text", "text": f"Supplier drawing: {path.name}"})
        content.append(image_payload(path, detail))
    for path in cad_images:
        content.append({"type": "text", "text": f"CAD render/metric view: {path.name}"})
        content.append(image_payload(path, detail))
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "response_format": {"type": "json_object"},
        "temperature": 0,
        "max_completion_tokens": 1600,
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        raw = json.loads(response.read().decode("utf-8"))
    content_text = raw["choices"][0]["message"].get("content") or "{}"
    result = parse_json_object(content_text)
    result["usage"] = raw.get("usage") or {}
    result["model"] = model
    result["source_images"] = [str(path) for path in source_images]
    result["cad_images"] = [str(path) for path in cad_images]
    return result


def drawing_match_review(
    args: argparse.Namespace,
    component: dict[str, Any],
    component_id: str,
    cad_metrics: dict[str, Any],
    source_metrics: dict[str, Any],
) -> dict[str, Any]:
    source_images = select_source_images(args.downloads, component_id, args.max_source_images)
    cad_images = render_metric_views(component_id, cad_metrics, args.output)
    base = {
        "model": args.model,
        "source_images": [str(path) for path in source_images],
        "cad_images": [str(path) for path in cad_images],
    }
    if args.skip_vision:
        return {**base, "status": "skipped", "verdict": STATUS_REVIEW, "confidence": 0.0, "reason": "vision review skipped"}
    if not source_images:
        return {**base, "status": "skipped", "verdict": STATUS_REVIEW, "confidence": 0.0, "reason": "no source drawing images found"}
    if not cad_images:
        return {**base, "status": "skipped", "verdict": STATUS_REVIEW, "confidence": 0.0, "reason": "CAD render generation unavailable"}
    api_key = load_etc_var("OPENAI_API_KEY")
    if not api_key:
        return {**base, "status": "skipped", "verdict": STATUS_REVIEW, "confidence": 0.0, "reason": "OPENAI_API_KEY not found"}
    try:
        result = call_openai_drawing_match(args.model, api_key, component, source_images, cad_images, cad_metrics, source_metrics, "low")
        confidence = float(result.get("confidence") or 0)
        if 0.45 <= confidence <= 0.75 and not args.no_high_detail_retry:
            result = call_openai_drawing_match(args.model, api_key, component, source_images[:1], cad_images, cad_metrics, source_metrics, "high")
            result["retried_high_detail"] = True
        result["status"] = "complete"
        return result
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
        return {**base, "status": "error", "verdict": STATUS_REVIEW, "confidence": 0.0, "reason": repr(exc)}


def gate_status(
    schema_check: dict[str, Any],
    static_check: dict[str, Any],
    script_exec: dict[str, Any],
    cad_status: str,
    cad_reasons: list[str],
    source_metrics: dict[str, Any],
    dimension_checks: list[dict[str, Any]],
    drawing_match: dict[str, Any],
) -> tuple[str, list[str]]:
    fail_reasons = []
    review_reasons = []
    if schema_check["status"] == STATUS_FAIL:
        fail_reasons.extend(schema_check["errors"])
    if static_check["status"] == STATUS_FAIL:
        fail_reasons.extend(static_check["errors"])
    if not script_exec.get("pass"):
        fail_reasons.append("CadQuery script did not execute")
    if cad_status == STATUS_FAIL:
        fail_reasons.extend(cad_reasons)
    if drawing_match.get("verdict") == STATUS_FAIL:
        fail_reasons.append("vision drawing-match verdict failed")
    if source_metrics.get("placeholders"):
        review_reasons.append("source values contain placeholders")
    if source_metrics.get("review_flags"):
        review_reasons.append("component source status requires review")
    if static_check.get("warnings"):
        review_reasons.extend(static_check["warnings"])
    if not dimension_checks:
        review_reasons.append("no source-backed numeric dimensions could be checked")
    elif any(check["status"] != STATUS_PASS for check in dimension_checks):
        review_reasons.append("one or more source dimensions did not match CAD bbox checks")
    if drawing_match.get("status") != "complete":
        review_reasons.append(f"drawing-match vision review {drawing_match.get('status', 'unknown')}")
    else:
        confidence = float(drawing_match.get("confidence") or 0)
        if drawing_match.get("verdict") != STATUS_PASS or confidence < 0.75:
            review_reasons.append("drawing-match vision review is not a high-confidence pass")
    if fail_reasons:
        return STATUS_FAIL, fail_reasons + review_reasons
    if review_reasons:
        return STATUS_REVIEW, review_reasons
    return STATUS_PASS, []


def repair_prompt(component: dict[str, Any], qc: dict[str, Any]) -> str:
    failures = qc.get("reasons") or []
    dim_failures = [
        f"{check['key']} expected {check['expected']} but observed bbox candidates {check['observed_candidates']}"
        for check in qc.get("dimension_checks") or []
        if check.get("status") != STATUS_PASS
    ][:8]
    drawing = qc.get("drawing_match") or {}
    vision_bits = []
    for key in ("mismatches", "missing_features", "extra_features", "dimension_label_mismatches"):
        value = drawing.get(key)
        if value:
            vision_bits.append(f"{key}: {value}")
    return (
        "Regenerate/fix the CadQuery 2.x script for this component. "
        "Define result as cq.Workplane or cq.Assembly, avoid show_object/cq.math/named colors, and use only source-backed dimensions. "
        f"Component: {component.get('name')}. "
        f"QC status: {qc.get('overall_status')}. Reasons: {'; '.join(map(str, failures[:10]))}. "
        f"Dimension issues: {'; '.join(dim_failures) if dim_failures else 'none reported'}. "
        f"Drawing-match issues: {'; '.join(map(str, vision_bits)) if vision_bits else 'none reported or vision skipped'}."
    )


def qc_component(args: argparse.Namespace, item: ComponentInput) -> dict[str, Any]:
    component = read_json(item.component_json)
    script = item.script_path.read_text(encoding="utf-8")
    schema_check = validate_component_schema(component, args.schema)
    static_check = static_script_checks(script, item.script_path)
    if static_check["status"] == STATUS_FAIL and any("syntax_error" in error for error in static_check["errors"]):
        script_exec = {"pass": False, "error_trace": "skipped because AST parse failed"}
    else:
        script_exec = run_cadquery(item.script_path, args.python_bin, args.timeout)
    cad_status, cad_reasons = cad_metrics_status(script_exec)
    source_metrics = extract_source_metrics(component)
    dimension_checks = compare_dimensions(source_metrics, script_exec, args.dimension_tolerance)
    drawing_match = drawing_match_review(args, component, item.component_id, script_exec, source_metrics)
    overall_status, reasons = gate_status(
        schema_check,
        static_check,
        script_exec,
        cad_status,
        cad_reasons,
        source_metrics,
        dimension_checks,
        drawing_match,
    )
    qc = {
        "component_id": item.component_id,
        "component_dir": str(item.component_dir),
        "relative_key": item.relative_key,
        "schema_check": schema_check,
        "static_checks": static_check,
        "script_exec": {"pass": bool(script_exec.get("pass")), "error_trace": script_exec.get("error_trace", "")},
        "cad_metrics": {key: value for key, value in script_exec.items() if key not in {"pass", "error_trace"}},
        "source_metrics": source_metrics,
        "dimension_checks": dimension_checks,
        "drawing_match": drawing_match,
        "overall_status": overall_status,
        "reasons": reasons,
    }
    qc["repair_prompt"] = repair_prompt(component, qc)
    return qc


def output_path_for(args: argparse.Namespace, item: ComponentInput) -> Path:
    if item.relative_key == item.component_id:
        return args.output / item.component_id / "qc.json"
    return args.output / item.relative_key / "qc.json"


def write_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    write_json(output_dir / "qc_summary.json", rows)
    fieldnames = [
        "component_id",
        "relative_key",
        "overall_status",
        "schema_status",
        "static_status",
        "script_exec_pass",
        "bbox",
        "solids_count",
        "assembly_children",
        "dimension_pass",
        "dimension_review",
        "drawing_verdict",
        "drawing_confidence",
        "reason_count",
    ]
    with (output_dir / "qc_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            cad = row.get("cad_metrics") or {}
            dim_checks = row.get("dimension_checks") or []
            drawing = row.get("drawing_match") or {}
            writer.writerow(
                {
                    "component_id": row.get("component_id"),
                    "relative_key": row.get("relative_key"),
                    "overall_status": row.get("overall_status"),
                    "schema_status": (row.get("schema_check") or {}).get("status"),
                    "static_status": (row.get("static_checks") or {}).get("status"),
                    "script_exec_pass": (row.get("script_exec") or {}).get("pass"),
                    "bbox": json.dumps(cad.get("bbox") or {}, sort_keys=True),
                    "solids_count": cad.get("solids_count", ""),
                    "assembly_children": cad.get("assembly_children", ""),
                    "dimension_pass": sum(1 for check in dim_checks if check.get("status") == STATUS_PASS),
                    "dimension_review": sum(1 for check in dim_checks if check.get("status") != STATUS_PASS),
                    "drawing_verdict": drawing.get("verdict", ""),
                    "drawing_confidence": drawing.get("confidence", ""),
                    "reason_count": len(row.get("reasons") or []),
                }
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, required=True, help="Directory containing generated component.json/cadquery_script.py outputs.")
    parser.add_argument("--downloads", type=Path, default=Path("downloads"), help="Directory containing source drawings/spec tables.")
    parser.add_argument("--model", default="gpt-5.4-mini", help="Vision model for drawing-match review.")
    parser.add_argument("--output", type=Path, required=True, help="QC output directory.")
    parser.add_argument("--schema", type=Path, default=Path("component_schema.json"))
    parser.add_argument("--python-bin", type=Path, default=Path(".venv/bin/python"))
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--dimension-tolerance", type=float, default=0.05)
    parser.add_argument("--max-source-images", type=int, default=3)
    parser.add_argument("--skip-vision", action="store_true", help="Skip OpenAI drawing-match review and only run deterministic QC.")
    parser.add_argument("--no-high-detail-retry", action="store_true", help="Disable high-detail retry for borderline vision confidence.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    components = discover_components(args.outputs)
    if args.limit is not None:
        components = components[: args.limit]
    if not components:
        raise SystemExit(f"no component outputs found under {args.outputs}")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in components:
        try:
            qc = qc_component(args, item)
        except Exception:  # noqa: BLE001 - keep batch going and emit actionable failure.
            qc = {
                "component_id": item.component_id,
                "component_dir": str(item.component_dir),
                "relative_key": item.relative_key,
                "overall_status": STATUS_FAIL,
                "reasons": ["qc pipeline exception"],
                "script_exec": {"pass": False, "error_trace": traceback.format_exc()[-4000:]},
                "cad_metrics": {},
                "source_metrics": {},
                "dimension_checks": [],
                "drawing_match": {"status": "skipped", "verdict": STATUS_REVIEW, "confidence": 0.0},
                "repair_prompt": "Fix the CadQuery script after resolving the QC pipeline exception in script_exec.error_trace.",
            }
        write_json(output_path_for(args, item), qc)
        rows.append(qc)
        print(json.dumps({"component_id": item.component_id, "relative_key": item.relative_key, "overall_status": qc["overall_status"]}, sort_keys=True), flush=True)
    write_summary(args.output, rows)
    counts = {status: sum(1 for row in rows if row.get("overall_status") == status) for status in (STATUS_PASS, STATUS_REVIEW, STATUS_FAIL)}
    print(json.dumps({"components": len(rows), "counts": counts, "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
