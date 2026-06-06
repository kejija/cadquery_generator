#!/usr/bin/env python3
"""Convert downloaded supplier component folders into schema JSON and CAD briefs."""

from __future__ import annotations

import argparse
import csv
import base64
import html
import json
import os
import random
import re
import struct
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
TOLERANCE_RE = re.compile(
    r"^\s*(?P<nominal>[-+]?\d+(?:\.\d+)?)?\s*(?:±|\+/-|\+-)\s*(?P<pm>\d+(?:\.\d+)?)\s*$"
)
LIMIT_RE = re.compile(
    r"^\s*(?P<upper>[-+]?\d+(?:\.\d+)?)\s+(?P<lower>[-+]?\d+(?:\.\d+)?)\s*$"
)
FIT_RE = re.compile(r"\b[A-Za-z]{0,2}[HhGgFfJjKkNnPp][0-9]{1,2}(?:/[A-Za-z]{0,2}[A-Za-z][0-9]{1,2})?\b")
THREAD_RE = re.compile(r"^M\d+(?:\.\d+)?(?:\s*[xX]\s*\d+(?:\.\d+)?)?$")
DIMENSION_KEYS = {
    "a",
    "b",
    "c",
    "d",
    "d1",
    "d2",
    "dr",
    "e",
    "f",
    "h",
    "l",
    "l1",
    "l2",
    "p",
    "pcd",
    "p.c.d.",
    "t",
    "w",
    "id",
    "od",
    "diameter",
    "length",
    "width",
    "height",
    "thickness",
}
MATERIAL_WORDS = ("material", "steel", "stainless", "aluminum", "brass", "plastic", "rubber", "resin")
SURFACE_WORDS = ("surface", "plating", "coating", "finish", "treatment", "black oxide", "anodize")
GDT_WORDS = (
    "perpendicularity",
    "parallelism",
    "concentricity",
    "runout",
    "flatness",
    "profile",
    "position",
    "eccentricity",
)


@dataclass
class Review:
    component_id: str
    confidence: float
    ready: bool
    flags: list[str]
    missing: list[str]


@dataclass
class OllamaConfig:
    model: str
    host: str
    max_images: int
    timeout: int


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def choose_specs(component_dir: Path) -> tuple[dict[str, Any], str]:
    candidates = [
        component_dir / "json" / "specs_after_click.json",
        component_dir / "json" / "specs_after.json",
        component_dir / "json" / "specs.json",
    ]
    best: tuple[dict[str, Any], str] | None = None
    best_score = -1
    for path in candidates:
        if not path.exists():
            continue
        data = load_json(path)
        score = len(data.get("tables") or []) * 10 + len(data.get("configFields") or []) + len(data.get("specImages") or [])
        if score > best_score:
            best = (data, str(path.relative_to(component_dir)))
            best_score = score
    if best is None:
        return {}, ""
    return best


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_source_url(page_html: str, component_id: str) -> str:
    patterns = [
        r'<link rel="canonical" href="([^"]+)"',
        r'<meta property="og:url"[^>]+content="([^"]+)"',
    ]
    for pattern in patterns:
        match = re.search(pattern, page_html)
        if match:
            return html.unescape(match.group(1))
    return f"https://us.misumi-ec.com/vona2/detail/{component_id}/"


def extract_category_code(specs: dict[str, Any], page_html: str) -> str:
    category_match = re.search(r"[?&]categoryCode=([A-Z0-9]+)", page_html)
    if category_match:
        return category_match.group(1)
    for crumb in reversed(specs.get("breadcrumbs") or []):
        url = crumb.get("url") or ""
        match = re.search(r"/([A-Z]\d{10})/?$", url)
        if match:
            return match.group(1)
    return "unknown"


def clean_cell(value: Any) -> str:
    text = html.unescape(str(value or "")).replace("\u00a0", " ")
    text = (
        text.replace("\u2009", " ")
        .replace("\u202f", " ")
        .replace("\u2212", "-")
        .replace("\u2010", "-")
        .replace("\u2011", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\uff0b", "+")
    )
    text = re.sub(r"([+-])\s+(\d)", r"\1\2", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_key(text: str) -> str:
    text = clean_cell(text)
    text = re.sub(r"\[[^\]]+\]", "", text)
    text = text.replace("φ", "D").replace("ø", "D")
    text = re.sub(r"[^A-Za-z0-9.]+", "_", text).strip("_")
    return text or "value"


def is_dimension_header(text: str) -> bool:
    cleaned = clean_cell(text).lower()
    key = normalize_key(cleaned).lower()
    return (
        key in DIMENSION_KEYS
        or "tolerance" in cleaned
        or "mass" in cleaned
        or "load rating" in cleaned
        or any(word in cleaned for word in MATERIAL_WORDS + SURFACE_WORDS + GDT_WORDS)
    )


def table_score(table: dict[str, Any]) -> int:
    rows = table.get("rows") or []
    if not rows:
        return 0
    header_hits = sum(is_dimension_header(cell) for cell in rows[0])
    numeric_rows = sum(any(NUMBER_RE.search(clean_cell(cell)) for cell in row) for row in rows[1:])
    width = max((len(row) for row in rows), default=0)
    return header_hits * 5 + numeric_rows + width


def best_dimension_table(tables: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not tables:
        return None
    scored = sorted(tables, key=table_score, reverse=True)
    return scored[0] if table_score(scored[0]) > 0 else None


def parse_number(text: str) -> float | None:
    match = NUMBER_RE.search(clean_cell(text))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def parse_value(cell: str) -> dict[str, Any] | None:
    text = clean_cell(cell)
    if not text or text in {"-", "ー", "—"}:
        return None
    if THREAD_RE.match(text):
        return {"value": text, "unit": "thread"}
    fit_match = FIT_RE.search(text)
    if fit_match:
        return {
            "value": text,
            "unit": "fit",
            "tolerance": {"type": "fit", "fit_class": fit_match.group(0), "fit_standard": "ISO 286"},
        }
    tolerance_match = TOLERANCE_RE.match(text)
    if tolerance_match:
        nominal = tolerance_match.group("nominal")
        value: float | str = float(nominal) if nominal is not None else text
        return {
            "value": value,
            "unit": "mm",
            "tolerance": {"type": "plus_minus", "value": float(tolerance_match.group("pm"))},
        }
    limit_match = LIMIT_RE.match(text)
    if limit_match:
        upper = float(limit_match.group("upper"))
        lower = float(limit_match.group("lower"))
        if upper >= lower:
            return {"value": upper, "unit": "mm", "tolerance": {"type": "limit", "min": lower, "max": upper}}
    number = parse_number(text)
    if number is not None:
        return {"value": number, "unit": "mm"}
    return {"value": text, "unit": "text"}


def extract_values(tables: list[dict[str, Any]]) -> dict[str, Any]:
    table = best_dimension_table(tables)
    if not table:
        return {}
    rows = table.get("rows") or []
    if len(rows) < 2:
        return {}
    headers = infer_headers(rows)
    data_row = next(
        (
            row
            for row in rows[1:]
            if not is_header_like_row(row) and any(parse_number(clean_cell(cell)) is not None for cell in row)
        ),
        None,
    )
    if data_row is None:
        return {}
    values: dict[str, Any] = {}
    for index, raw_cell in enumerate(data_row):
        if index >= len(headers):
            continue
        key = headers[index]
        parsed = parse_value(clean_cell(raw_cell))
        if parsed is None:
            continue
        parsed["source"] = {
            "spec_name": "dimension_table",
            "spec_code": f"table_{table.get('idx', 0)}",
            "asset_kind": "scraped_table",
        }
        values[key] = parsed
    enrich_tolerance_values(tables, values)
    return values


def infer_headers(rows: list[list[Any]]) -> list[str]:
    primary = [normalize_key(cell) for cell in rows[0]]
    if len(rows) < 2:
        return primary
    secondary = [clean_cell(cell) for cell in rows[1]]
    if not any("tolerance" in cell.lower() or clean_cell(cell).lower() in {"type", "dr"} for cell in secondary):
        return primary
    headers = []
    previous_dimension = ""
    for index, header in enumerate(primary):
        sub = secondary[index] if index < len(secondary) else ""
        sub_key = normalize_key(sub)
        sub_lower = sub.lower()
        if sub_lower == "dr":
            header = "D"
        elif "tolerance" in sub_lower and previous_dimension:
            header = f"{previous_dimension}_Tolerance"
        elif sub and sub_lower not in {"type"} and not parse_number(sub) and "treatment" in sub_lower and previous_dimension:
            header = f"{previous_dimension}_{sub_key}"
        headers.append(header)
        if header and "Tolerance" not in header and header not in {"Part_Number", "Type"}:
            previous_dimension = header
    return headers


def is_header_like_row(row: list[Any]) -> bool:
    first = clean_cell(row[0] if row else "").lower()
    joined = " ".join(clean_cell(cell).lower() for cell in row[:4])
    return first in {"type", "part number"} or joined.startswith("type ")


def enrich_tolerance_values(tables: list[dict[str, Any]], values: dict[str, Any]) -> None:
    fit_classes: set[str] = set()
    for table in tables:
        rows = table.get("rows") or []
        unit = "um" if any("unit:" in clean_cell(cell).lower() and "μm" in clean_cell(cell) for row in rows for cell in row) else "mm"
        for row in rows:
            cells = [clean_cell(cell) for cell in row]
            joined = " | ".join(cells)
            for fit in FIT_RE.findall(joined):
                if fit.upper() == "L1":
                    continue
                fit_classes.add(fit)
                merge_fit_tolerance(values, fit, joined)
            first = cells[0] if cells else ""
            if "tolerance" in first.lower():
                parsed = next((parse_value(cell) for cell in cells[1:] if parse_value(cell)), None)
                if parsed and parsed.get("tolerance"):
                    key = normalize_key(first)
                    parsed["unit"] = unit
                    parsed.setdefault(
                        "source",
                        {
                            "spec_name": first,
                            "spec_code": f"table_{table.get('idx', 0)}",
                            "asset_kind": "scraped_table",
                        },
                    )
                    values.setdefault(key, parsed)
    useful_fits = sorted(fit for fit in fit_classes if fit.lower() in {"h9", "h7", "g6", "g7", "h6", "h8", "dh7"} or fit.upper() == "DH7")
    if useful_fits and "D_Tolerance_Options" not in values:
        values["D_Tolerance_Options"] = {
            "value": "select_by_type",
            "unit": "fit",
            "enum": useful_fits,
            "enum_details": {
                fit: {"tolerance": {"type": "fit", "fit_class": fit, "fit_standard": "ISO 286"}}
                for fit in useful_fits
            },
            "source": {
                "spec_name": "catalog fit tolerance options",
                "asset_kind": "scraped_table",
            },
        }


def merge_fit_tolerance(values: dict[str, Any], fit_class: str, source_text: str) -> None:
    key = ""
    if fit_class.upper().startswith("D") and len(fit_class) > 2:
        key = "D"
        fit_class = fit_class[1:]
    elif "d tolerance" in source_text.lower() or " d " in f" {source_text.lower()} ":
        key = "D_Tolerance"
    if not key:
        return
    value = values.get(key)
    if not isinstance(value, dict):
        value = {"value": f"{{{{{key}}}}}", "unit": "mm"}
        values[key] = value
    value.setdefault("tolerance", {"type": "fit", "fit_class": fit_class, "fit_standard": "ISO 286"})
    value.setdefault(
        "source",
        {
            "spec_name": "catalog fit tolerance",
            "asset_kind": "scraped_table",
        },
    )


def extract_configuration_schema(specs: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    title = specs.get("title") or ""
    tables = specs.get("tables") or []
    parameters = []
    for key, value in values.items():
        unit = value.get("unit") if isinstance(value, dict) else None
        parameters.append(
            {
                "key": key,
                "kind": "derived",
                "unit": unit or "mm",
                "required_for_part_number": key.lower() in {"d", "dr", "l", "type", "material"},
                "required_for_cad": True,
                "source": "representative dimension row",
            }
        )
    family_code = ""
    for table in tables:
        for row in table.get("rows") or []:
            joined = " ".join(clean_cell(cell) for cell in row)
            if "Part Number" in joined:
                continue
            match = re.search(r"\b[A-Z][A-Z0-9-]{2,}\b", joined)
            if match:
                family_code = match.group(0)
                break
        if family_code:
            break
    if not parameters and not family_code:
        return {}
    return {
        "family_code": family_code or normalize_key(title).upper()[:32],
        "parameters": parameters,
        "representative_configuration": {
            "model_number": family_code or "representative",
            "source": "first numeric dimension row",
            "values": values,
        },
    }


def image_size(path: Path) -> tuple[int, int] | None:
    try:
        with path.open("rb") as f:
            header = f.read(32)
        if header[:6] in (b"GIF87a", b"GIF89a"):
            return struct.unpack("<HH", header[6:10])
        if header.startswith(b"\xff\xd8"):
            with path.open("rb") as f:
                f.read(2)
                while True:
                    marker_start = f.read(1)
                    if not marker_start:
                        return None
                    if marker_start != b"\xff":
                        continue
                    marker = f.read(1)
                    while marker == b"\xff":
                        marker = f.read(1)
                    if marker in (b"\xc0", b"\xc1", b"\xc2", b"\xc3", b"\xc5", b"\xc6", b"\xc7", b"\xc9", b"\xca", b"\xcb", b"\xcd", b"\xce", b"\xcf"):
                        length = struct.unpack(">H", f.read(2))[0]
                        data = f.read(length - 2)
                        height, width = struct.unpack(">HH", data[1:5])
                        return width, height
                    length_bytes = f.read(2)
                    if len(length_bytes) != 2:
                        return None
                    length = struct.unpack(">H", length_bytes)[0]
                    f.seek(length - 2, os.SEEK_CUR)
    except OSError:
        return None
    return None


def drawing_type(path: Path) -> str:
    name = path.name.lower()
    if name.startswith("drw") or "drw" in name:
        return "orthographic_multiview"
    if name.startswith("alt") or "alter" in name:
        return "detail_view"
    if "table" in name or name.startswith("spc") or "spec" in name:
        return "datasheet_table"
    if name.startswith("oth"):
        return "other"
    return "other"


def extract_drawings(component_dir: Path) -> list[dict[str, Any]]:
    drawing_dir = component_dir / "drawings"
    candidates = []
    for path in sorted(drawing_dir.glob("*")):
        if not path.is_file():
            continue
        name = path.name.lower()
        if name == "product_photo.jpg":
            continue
        size = image_size(path)
        if size and size[0] <= 32 and size[1] <= 32:
            continue
        kind = drawing_type(path)
        if kind == "other" and size and size[0] * size[1] < 20_000:
            continue
        candidates.append((path, size, kind))
    drawings = []
    for index, (path, size, kind) in enumerate(candidates, start=1):
        width, height = size or (None, None)
        drawings.append(
            {
                "index": index,
                "drawing_type": kind,
                "geometry_type": infer_geometry_type(path, component_dir),
                "geometry_summary": drawing_summary(path, width, height),
                "variables": [],
                "gdt": [],
                "chamfers": [],
                "surface_notes": [],
            }
        )
    return drawings


def infer_geometry_type(path: Path, component_dir: Path) -> str:
    title = ""
    specs_path = component_dir / "json" / "specs.json"
    if specs_path.exists():
        try:
            title = str(load_json(specs_path).get("title") or "").lower()
        except (OSError, json.JSONDecodeError):
            title = ""
    filename = path.name.lower()
    text = f"{title} {filename}"
    if any(word in text for word in ("bearing", "bushing", "pulley", "shaft", "collar", "pin")):
        return "solid_of_revolution"
    if any(word in text for word in ("bracket", "plate", "block", "flange")):
        return "extruded_prism"
    if any(word in text for word in ("assembly", "unit", "stage")):
        return "assembly"
    return "other"


def drawing_summary(path: Path, width: int | None, height: int | None) -> str:
    size = f"{width}x{height}" if width and height else "unknown size"
    return (
        f"Candidate engineering asset {path.name} ({size}). "
        "Requires visual review to map drawing labels, section views, and hidden features to CAD operations."
    )


def flatten_table_text(tables: list[dict[str, Any]]) -> str:
    chunks = []
    for table in tables:
        for row in table.get("rows") or []:
            chunks.append(" | ".join(clean_cell(cell) for cell in row if clean_cell(cell)))
    return "\n".join(chunks)


def extract_materials_and_finishes(tables: list[dict[str, Any]]) -> tuple[list[str], list[str], list[str]]:
    text = flatten_table_text(tables)
    materials = sorted({line for line in text.splitlines() if any(word in line.lower() for word in MATERIAL_WORDS)})[:12]
    finishes = sorted({line for line in text.splitlines() if any(word in line.lower() for word in SURFACE_WORDS)})[:12]
    gdt = sorted({line for line in text.splitlines() if any(word in line.lower() for word in GDT_WORDS)})[:12]
    return materials, finishes, gdt


def build_component(component_dir: Path, ollama: OllamaConfig | None = None) -> tuple[dict[str, Any], Review, dict[str, Any]]:
    component_id = component_dir.name
    specs, specs_source = choose_specs(component_dir)
    page_html = read_text(component_dir / "html" / "page.html")
    source_url = extract_source_url(page_html, component_id)
    title = clean_cell(specs.get("title") or component_id)
    breadcrumbs = specs.get("breadcrumbs") or []
    category = clean_cell(breadcrumbs[-1].get("text")) if breadcrumbs else ""
    category_code = extract_category_code(specs, page_html)
    tables = specs.get("tables") or []
    values = extract_values(tables)
    drawings = extract_drawings(component_dir)
    materials, finishes, gdt_lines = extract_materials_and_finishes(tables)

    attributes: dict[str, Any] = {
        "brand": "MISUMI",
        "supplier": "MISUMI",
        "source_url": source_url,
        "series_code": component_id,
        "category": category,
        "department": "mech" if "/mech/" in source_url or any("/mech/" in (c.get("url") or "") for c in breadcrumbs) else "unknown",
        "configuration_count": len(tables),
    }
    if drawings:
        attributes["drawing_count"] = len(drawings)
    if values:
        attributes["representative_value_count"] = len(values)

    component: dict[str, Any] = {
        "name": title,
        "description": description_for(title, category, values, drawings),
        "category_code": category_code,
        "status": [],
        "drawings": drawings,
        "attributes": attributes,
        "standards": {
            "unit_system": "metric",
            "export_format": "STEP AP242",
        },
    }
    if values:
        component["values"] = values
        configuration = extract_configuration_schema(specs, values)
        if configuration:
            component["configuration_schema"] = configuration
    if finishes:
        component["surface_finish"] = [
            {
                "surface": "unspecified catalog surface",
                "symbol": "basic",
                "production_method": finish,
            }
            for finish in finishes[:5]
        ]

    flags, missing = review_flags(tables, values, drawings, materials, finishes, gdt_lines)
    confidence = confidence_score(tables, values, drawings, materials, finishes, gdt_lines, flags)
    ready = confidence >= 0.9 and not flags
    component["status"] = ["ready_for_cad_instruction" if ready else "needs_review"] + flags

    evidence = {
        "component_id": component_id,
        "specs_source": specs_source,
        "source_url": source_url,
        "tables": summarize_tables(tables),
        "table_excerpts": excerpt_tables(tables),
        "materials": materials,
        "finishes": finishes,
        "gdt_lines": gdt_lines,
        "drawing_files": [p.name for p in sorted((component_dir / "drawings").glob("*")) if p.is_file()],
    }
    review = Review(component_id, confidence, ready, flags, missing)
    if ollama:
        llm_result = analyze_with_ollama(component_dir, component, evidence, ollama)
        evidence["llm"] = llm_result
        apply_llm_result(component, llm_result)
        flags, missing = review_flags(tables, component.get("values") or {}, drawings, materials, finishes, gdt_lines)
        llm_flags, llm_notes = split_llm_flags(llm_result.get("review_flags", []))
        flags = sorted(set(flags + llm_flags))
        if llm_notes:
            component.setdefault("annotations", []).extend(
                {"type": "note", "text": note, "scope": "component"} for note in llm_notes
            )
        missing = sorted(set(item for item in missing if item not in llm_result.get("resolved_missing", [])))
        if llm_result.get("drawing_variables") and (llm_result.get("modeling_operations") or llm_result.get("geometry_summary")):
            flags = [flag for flag in flags if flag != "needs_review:drawing_geometry_requires_visual_mapping"]
            missing = [item for item in missing if item != "visual mapping of drawing labels to CAD features"]
        if llm_result.get("gdt_callouts"):
            flags = [flag for flag in flags if flag != "needs_review:missing_gdt_or_form_control_evidence"]
            missing = [item for item in missing if item != "GD&T/form controls"]
        confidence = max(confidence, float(llm_result.get("confidence", 0) or 0))
        if llm_result.get("manufacturing_ready") is True and confidence >= 0.9:
            flags = [flag for flag in flags if flag != "needs_review:drawing_geometry_requires_visual_mapping"]
        ready = confidence >= 0.9 and not flags
        component["status"] = ["ready_for_cad_instruction" if ready else "needs_review"] + flags
        review = Review(component_id, round(confidence, 3), ready, flags, missing)
    return component, review, evidence


def split_llm_flags(raw: Any) -> tuple[list[str], list[str]]:
    flags = []
    notes = []
    if not isinstance(raw, list):
        return flags, notes
    for item in raw:
        text = clean_cell(item)
        if not text:
            continue
        if text.startswith("needs_review:"):
            flags.append(text)
        else:
            notes.append(text)
    return flags, notes


def description_for(title: str, category: str, values: dict[str, Any], drawings: list[dict[str, Any]]) -> str:
    bits = [title]
    if category:
        bits.append(f"in {category}")
    if values:
        bits.append(f"with {len(values)} representative extracted parameters")
    if drawings:
        bits.append(f"with {len(drawings)} candidate engineering drawing assets")
    return ", ".join(bits) + "."


def review_flags(
    tables: list[dict[str, Any]],
    values: dict[str, Any],
    drawings: list[dict[str, Any]],
    materials: list[str],
    finishes: list[str],
    gdt_lines: list[str],
) -> tuple[list[str], list[str]]:
    flags = []
    missing = []
    if not tables:
        flags.append("needs_review:missing_spec_tables")
        missing.append("dimension/specification tables")
    if not values:
        flags.append("needs_review:missing_representative_dimensions")
        missing.append("representative dimensions")
    if not drawings:
        flags.append("needs_review:missing_engineering_drawing")
        missing.append("engineering drawing")
    if not materials:
        flags.append("needs_review:missing_material_evidence")
        missing.append("material")
    if not finishes:
        flags.append("needs_review:missing_surface_finish_evidence")
        missing.append("surface finish")
    if not has_tolerance_evidence(values):
        flags.append("needs_review:missing_tolerance")
        missing.append("dimension tolerances")
    if not gdt_lines:
        flags.append("needs_review:missing_gdt_or_form_control_evidence")
        missing.append("GD&T/form controls")
    if drawings:
        flags.append("needs_review:drawing_geometry_requires_visual_mapping")
        missing.append("visual mapping of drawing labels to CAD features")
    return sorted(set(flags)), sorted(set(missing))


def confidence_score(
    tables: list[dict[str, Any]],
    values: dict[str, Any],
    drawings: list[dict[str, Any]],
    materials: list[str],
    finishes: list[str],
    gdt_lines: list[str],
    flags: list[str],
) -> float:
    score = 0.0
    if tables:
        score += 0.15
    if values:
        score += min(0.25, len(values) * 0.02)
    if drawings:
        score += 0.15
    if materials:
        score += 0.1
    if finishes:
        score += 0.1
    if gdt_lines:
        score += 0.1
    if has_tolerance_evidence(values):
        score += 0.15
    score -= min(0.2, len(flags) * 0.02)
    return max(0.0, min(1.0, round(score, 3)))


def has_tolerance_evidence(values: dict[str, Any]) -> bool:
    return any(
        isinstance(value, dict) and ("tolerance" in value or "enum_details" in value)
        for value in values.values()
    )


def summarize_tables(tables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for table in tables:
        rows = table.get("rows") or []
        summary.append(
            {
                "idx": table.get("idx"),
                "row_count": len(rows),
                "first_row": [clean_cell(cell) for cell in rows[0][:12]] if rows else [],
            }
        )
    return summary


def excerpt_tables(tables: list[dict[str, Any]], max_tables: int = 8, max_rows: int = 16, max_cells: int = 18) -> list[dict[str, Any]]:
    excerpts = []
    for table in tables[:max_tables]:
        rows = table.get("rows") or []
        excerpts.append(
            {
                "idx": table.get("idx"),
                "rows": [
                    [clean_cell(cell) for cell in row[:max_cells] if clean_cell(cell)]
                    for row in rows[:max_rows]
                ],
            }
        )
    return excerpts


def analyze_with_ollama(
    component_dir: Path,
    component: dict[str, Any],
    evidence: dict[str, Any],
    config: OllamaConfig,
) -> dict[str, Any]:
    images = selected_image_payloads(component_dir, config.max_images)
    prompt = build_llm_prompt(component, evidence)
    message: dict[str, Any] = {"role": "user", "content": prompt}
    if images:
        message["images"] = images
    payload = {
        "model": config.model,
        "messages": [message],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0,
            "top_p": 0.2,
            "num_ctx": 32768,
        },
    }
    req = urllib.request.Request(
        config.host.rstrip("/") + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=config.timeout) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {
            "error": repr(exc),
            "confidence": 0,
            "review_flags": ["needs_review:ollama_analysis_failed"],
            "cad_instruction_sections": {},
        }
    content = ((raw.get("message") or {}).get("content") or "").strip()
    parsed = parse_json_object(content)
    if not isinstance(parsed, dict):
        return {
            "error": "ollama_response_not_json",
            "raw_content": content[:2000],
            "confidence": 0,
            "review_flags": ["needs_review:ollama_analysis_failed"],
            "cad_instruction_sections": {},
        }
    parsed["_ollama_model"] = config.model
    parsed["_image_count"] = len(images)
    return parsed


def selected_image_payloads(component_dir: Path, max_images: int) -> list[str]:
    drawing_dir = component_dir / "drawings"
    candidates = []
    for path in sorted(drawing_dir.glob("*")):
        if not path.is_file() or path.name.lower() == "product_photo.jpg":
            continue
        size = image_size(path)
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
        area = (size[0] * size[1]) if size else 0
        candidates.append((priority, area, path))
    images = []
    for _priority, _area, path in sorted(candidates, reverse=True)[:max_images]:
        try:
            images.append(base64.b64encode(path.read_bytes()).decode("ascii"))
        except OSError:
            continue
    return images


def build_llm_prompt(component: dict[str, Any], evidence: dict[str, Any]) -> str:
    compact_tables = []
    for table in evidence.get("tables", [])[:8]:
        compact_tables.append(table)
    values = component.get("values") or {}
    source_context = {
        "component_id": evidence["component_id"],
        "name": component["name"],
        "category": component["attributes"].get("category", ""),
        "category_code": component["category_code"],
        "source_url": evidence.get("source_url", ""),
        "existing_values": values,
        "table_summaries": compact_tables,
        "table_excerpts": evidence.get("table_excerpts", [])[:8],
        "material_evidence": evidence.get("materials", [])[:10],
        "finish_evidence": evidence.get("finishes", [])[:10],
        "gdt_text_evidence": evidence.get("gdt_lines", [])[:10],
        "drawing_files": evidence.get("drawing_files", [])[:12],
    }
    return (
        "You are converting MISUMI component catalog data and engineering drawings into a manufacturing CAD brief.\n"
        "Use the attached drawing images plus the JSON context. Return JSON only. Do not include markdown.\n"
        "Do not invent hidden dimensions; mark unknowns. Prefer exact drawing labels like D, L, P.C.D., M, T.\n"
        "A shaft, bushing, bearing, pulley, collar, pin, or other lathe-like cylindrical part is usually solid_of_revolution, not extruded_prism.\n"
        "For exact manufacturing, preserve tolerance/GD&T/material/finish evidence when visible and flag missing evidence.\n\n"
        "Also return a conservative CadQuery 2.x Python script in cadquery_script. The script must define result as a cq.Workplane or cq.Assembly, "
        "use only numeric values supported by context, create named variables at the top, and add TODO comments for unknown dimensions instead of inventing them. "
        "Keep it syntactically valid Python and do not include markdown fences.\n\n"
        "Return this object shape:\n"
        "{\n"
        '  "geometry_type": "solid_of_revolution|extruded_prism|sheet_metal|assembly|other",\n'
        '  "geometry_summary": "1-3 sentence real part geometry summary",\n'
        '  "drawing_variables": [{"key":"D","label":"D","role":"outer diameter","geometry":"main cylinder OD"}],\n'
        '  "derived_values": {"D":{"value":"{{D}}","unit":"mm","source_note":"visible drawing label or table evidence"}},\n'
        '  "materials": ["evidence-backed material choices"],\n'
        '  "surface_finishes": ["evidence-backed surface treatments or roughness"],\n'
        '  "gdt_callouts": [{"symbol":"perpendicularity","value":"0.01","applies_to":"flange face"}],\n'
        '  "datums": ["primary datum/coordinate assumptions"],\n'
        '  "modeling_operations": ["ordered CAD operation using labels and reviewed dimensions"],\n'
        '  "validation_checks": ["checks before CAD export"],\n'
        '  "cadquery_script": "import cadquery as cq\\n...\\nresult = ...",\n'
        '  "unknowns": ["missing exact information"],\n'
        '  "review_flags": ["needs_review:..."],\n'
        '  "resolved_missing": ["dimension tolerances"],\n'
        '  "manufacturing_ready": false,\n'
        '  "confidence": 0.0\n'
        "}\n\n"
        "Context JSON:\n"
        + json.dumps(source_context, ensure_ascii=False, indent=2)[:24000]
    )


def parse_json_object(content: str) -> Any:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def apply_llm_result(component: dict[str, Any], llm: dict[str, Any]) -> None:
    if llm.get("error"):
        component.setdefault("annotations", []).append(
            {"type": "note", "text": f"Ollama analysis failed: {llm['error']}", "scope": "component"}
        )
        return
    geometry_type = normalized_geometry_type(llm.get("geometry_type"), llm.get("geometry_summary"), component["name"])
    geometry_summary = clean_cell(llm.get("geometry_summary") or "")
    variables = normalize_llm_variables(llm.get("drawing_variables"))
    for drawing in component.get("drawings") or []:
        if geometry_type in {"solid_of_revolution", "extruded_prism", "sheet_metal", "assembly", "other"}:
            drawing["geometry_type"] = geometry_type
        if geometry_summary:
            drawing["geometry_summary"] = geometry_summary
        if variables:
            drawing["variables"] = variables
        drawing["gdt"] = normalize_llm_drawing_gdt(llm.get("gdt_callouts"))
        drawing["surface_notes"] = [clean_cell(x) for x in llm.get("surface_finishes") or [] if clean_cell(x)]

    values = component.setdefault("values", {})
    for key, value in normalize_llm_values(llm.get("derived_values")).items():
        if key not in values:
            values[key] = value

    materials = [clean_cell(x) for x in llm.get("materials") or [] if clean_cell(x)]
    if materials:
        component["attributes"]["llm_material_summary"] = "; ".join(materials[:5])
    finishes = [clean_cell(x) for x in llm.get("surface_finishes") or [] if clean_cell(x)]
    if finishes:
        component["surface_finish"] = [
            {"surface": "LLM-identified catalog surface", "symbol": "basic", "production_method": finish}
            for finish in finishes[:8]
        ]

    if llm.get("datums"):
        component.setdefault("annotations", []).append(
            {
                "type": "note",
                "text": "Datum guidance: " + "; ".join(map(clean_cell, llm["datums"][:6])),
                "scope": "component",
            }
        )
    component["cadquery_script"] = normalize_cadquery_script(llm.get("cadquery_script"))
    structured_gdt = normalize_component_gdt(llm.get("gdt_callouts"))
    if structured_gdt:
        component["gdt_callouts"] = structured_gdt


def normalize_cadquery_script(raw: Any) -> str:
    if raw is None:
        return ""
    script = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False, indent=2)
    script = script.strip()
    script = re.sub(r"^```(?:python)?\s*", "", script)
    script = re.sub(r"\s*```$", "", script)
    return script.strip()


def normalized_geometry_type(raw: Any, summary: Any, name: str) -> str:
    allowed = {"solid_of_revolution", "extruded_prism", "sheet_metal", "assembly", "other"}
    text = f"{raw or ''} {summary or ''} {name}".lower()
    if any(word in text for word in ("shaft", "cylindrical", "bushing", "bearing", "pulley", "collar", "pin")):
        return "solid_of_revolution"
    raw_text = clean_cell(raw)
    return raw_text if raw_text in allowed else "other"


def normalize_llm_variables(raw: Any) -> list[dict[str, str]]:
    variables = []
    if not isinstance(raw, list):
        return variables
    for item in raw[:40]:
        if not isinstance(item, dict):
            continue
        key = clean_cell(item.get("key") or item.get("label") or "")
        if not key:
            continue
        variables.append(
            {
                "key": normalize_key(key),
                "label": key,
                "role": clean_cell(item.get("role") or "dimension/control variable"),
                "geometry": clean_cell(item.get("geometry") or "LLM-identified drawing feature"),
            }
        )
    return variables


def normalize_llm_drawing_gdt(raw: Any) -> list[dict[str, str]]:
    gdt = []
    if not isinstance(raw, list):
        return gdt
    for item in raw[:20]:
        if isinstance(item, str):
            gdt.append({"symbol": "other", "value_label": item, "applies_to": "unspecified feature"})
        elif isinstance(item, dict):
            gdt.append(
                {
                    "symbol": clean_cell(item.get("symbol") or "other"),
                    "value_label": clean_cell(item.get("value") or item.get("value_label") or ""),
                    "applies_to": clean_cell(item.get("applies_to") or item.get("feature") or "unspecified feature"),
                }
            )
    return gdt


def normalize_component_gdt(raw: Any) -> list[dict[str, Any]]:
    symbol_map = {
        "flatness": "flatness",
        "cylindricity": "cylindricity",
        "circularity": "circularity",
        "straightness": "straightness",
        "profile": "profile_surface",
        "profile_surface": "profile_surface",
        "profile_line": "profile_line",
        "perpendicularity": "perpendicularity",
        "parallelism": "parallelism",
        "angularity": "angularity",
        "position": "position",
        "concentricity": "concentricity",
        "symmetry": "symmetry",
        "runout": "circular_runout",
        "circular_runout": "circular_runout",
        "total_runout": "total_runout",
        "eccentricity": "concentricity",
    }
    result = []
    if not isinstance(raw, list):
        return result
    for item in raw[:20]:
        if not isinstance(item, dict):
            continue
        symbol = symbol_map.get(clean_cell(item.get("symbol")).lower())
        value = parse_number(clean_cell(item.get("value") or item.get("value_label") or ""))
        feature = clean_cell(item.get("applies_to") or item.get("feature") or "unspecified feature")
        if not symbol or value is None:
            continue
        result.append(
            {
                "feature": feature,
                "symbol": symbol,
                "tolerance_zone": {"value": value, "unit": "mm"},
            }
        )
    return result


def normalize_llm_values(raw: Any) -> dict[str, Any]:
    values = {}
    if not isinstance(raw, dict):
        return values
    for raw_key, raw_value in raw.items():
        key = normalize_key(raw_key)
        if isinstance(raw_value, dict):
            value = raw_value.get("value", f"{{{{{key}}}}}")
            unit = raw_value.get("unit") or "mm"
            source_note = clean_cell(raw_value.get("source_note") or "LLM extraction")
        else:
            value = raw_value
            unit = "mm"
            source_note = "LLM extraction"
        if isinstance(value, str):
            parsed_number = parse_number(value)
            has_range_or_variable = bool(re.search(r"\d\s*[-~]\s*\d|variable|varies|depends|option", value, re.I))
            value = parsed_number if parsed_number is not None and "{{" not in str(value) and not has_range_or_variable else value
        elif not isinstance(value, (int, float)):
            value = json.dumps(value, ensure_ascii=False, sort_keys=True)
        values[key] = {
            "value": value,
            "unit": clean_cell(unit) or "mm",
            "source": {
                "spec_name": source_note[:120],
                "asset_kind": "ollama_vision_extraction",
            },
        }
    return values


def build_instructions(component: dict[str, Any], review: Review, evidence: dict[str, Any]) -> str:
    values = component.get("values") or {}
    drawings = component.get("drawings") or []
    attrs = component["attributes"]
    lines = [
        f"# CAD Instructions: {component['name']}",
        "",
        "## Source",
        f"- Component id: `{review.component_id}`",
        f"- Supplier URL: {attrs.get('source_url')}",
        f"- Category: {attrs.get('category', 'unknown')} (`{component['category_code']}`)",
        f"- Primary scraped data: `{evidence.get('specs_source') or 'none'}`",
        "",
        "## Manufacturing Readiness",
        f"- Confidence: {review.confidence:.3f}",
        f"- Status: {'ready' if review.ready else 'review required'}",
    ]
    if review.flags:
        lines.extend(f"- {flag}" for flag in review.flags)
    lines.extend(["", "## Coordinate System And Datums"])
    llm_sections = (evidence.get("llm") or {}).get("cad_instruction_sections") or {}
    llm_datums = (evidence.get("llm") or {}).get("datums") or []
    lines.extend([f"- {clean_cell(item)}" for item in llm_datums] or coordinate_guidance(component))
    lines.extend(["", "## Modeling Operations"])
    llm_ops = (evidence.get("llm") or {}).get("modeling_operations") or []
    lines.extend([f"- {clean_cell(item)}" for item in llm_ops] or modeling_guidance(component))
    lines.extend(["", "## Representative Parameters"])
    if values:
        lines.extend(parameter_lines(values))
    else:
        lines.append("- No representative dimension row was extracted. Review source tables/drawings before CAD generation.")
    lines.extend(["", "## Drawings To Review"])
    if drawings:
        for drawing in drawings:
            lines.append(
                f"- Drawing {drawing['index']}: {drawing['drawing_type']}, {drawing['geometry_type']}. {drawing['geometry_summary']}"
            )
    else:
        lines.append("- No candidate engineering drawings were found after filtering thumbnails/product photos.")
    lines.extend(["", "## Materials, Finish, And Controls"])
    lines.extend(material_finish_lines(evidence))
    lines.extend(["", "## CAD Validation Checklist"])
    llm_checks = (evidence.get("llm") or {}).get("validation_checks") or []
    lines.extend(
        [f"- {clean_cell(item)}" for item in llm_checks]
        or [
            "- Confirm all dimensions against the referenced drawing before generating manufacturing CAD.",
            "- Confirm units are millimeters unless the supplier page explicitly states otherwise.",
            "- Apply tolerances, fits, GD&T, and surface finish only from source evidence.",
            "- Export the final reviewed model as STEP AP242 to preserve manufacturing annotations.",
            "- Do not treat this brief as production-ready while any `needs_review:*` status remains.",
        ]
    )
    llm_unknowns = (evidence.get("llm") or {}).get("unknowns") or []
    if llm_unknowns:
        lines.extend(["", "## LLM-Identified Unknowns"])
        lines.extend(f"- {clean_cell(item)}" for item in llm_unknowns)
    if review.missing:
        lines.extend(["", "## Unknowns Requiring Triage"])
        lines.extend(f"- {item}" for item in review.missing)
    lines.append("")
    return "\n".join(lines)


def coordinate_guidance(component: dict[str, Any]) -> list[str]:
    geometry_types = {drawing.get("geometry_type") for drawing in component.get("drawings", [])}
    if "solid_of_revolution" in geometry_types:
        return [
            "- Place the primary rotational axis on global Z.",
            "- Put the principal mounting or flange face on the XY plane where identifiable.",
            "- Use the largest coaxial bore/shaft feature as the secondary datum axis after review.",
        ]
    if "extruded_prism" in geometry_types:
        return [
            "- Place the broadest machined mounting face on the XY plane.",
            "- Align the longest orthogonal edge to global X.",
            "- Use the first confirmed hole pattern centerline as a secondary datum after review.",
        ]
    return [
        "- Establish datums from the supplier drawing before modeling.",
        "- Prefer a right-handed coordinate system with primary mounting face on XY and main length along X or Z.",
    ]


def modeling_guidance(component: dict[str, Any]) -> list[str]:
    geometry_types = {drawing.get("geometry_type") for drawing in component.get("drawings", [])}
    values = component.get("values") or {}
    ops = []
    if "solid_of_revolution" in geometry_types:
        ops.extend(
            [
                "- Build the base body from a revolved half-section using reviewed diameter and length dimensions.",
                "- Add bores, counterbores, grooves, flanges, and chamfers in axial order from the engineering drawing.",
                "- Pattern radial holes or notches only after the pitch circle and angular spacing are confirmed.",
            ]
        )
    elif "extruded_prism" in geometry_types:
        ops.extend(
            [
                "- Sketch the primary profile from reviewed width, height, and thickness dimensions.",
                "- Extrude the base solid, then add pockets, holes, slots, chamfers, and fillets from the drawing.",
                "- Mirror or pattern repeated mounting features only when spacing and count are explicit.",
            ]
        )
    else:
        ops.extend(
            [
                "- Classify the geometry from the ranked drawings before selecting revolve, extrude, sweep, or assembly operations.",
                "- Convert each visible dimension label into a named CAD parameter before creating features.",
            ]
        )
    if values:
        ops.append("- Use the representative parameters below as a starting configuration, then replace with the reviewed target SKU values.")
    return ops


def parameter_lines(values: dict[str, Any]) -> list[str]:
    lines = []
    for key in sorted(values):
        value = values[key]
        if not isinstance(value, dict):
            lines.append(f"- `{key}` = {value}")
            continue
        raw = value.get("value")
        unit = value.get("unit", "")
        tolerance = value.get("tolerance")
        suffix = f" {unit}" if unit and unit != "text" else ""
        if tolerance:
            suffix += f"; tolerance {json.dumps(tolerance, ensure_ascii=False)}"
        lines.append(f"- `{key}` = {raw}{suffix}")
    return lines


def material_finish_lines(evidence: dict[str, Any]) -> list[str]:
    lines = []
    llm = evidence.get("llm") or {}
    if llm.get("materials"):
        lines.append("- LLM material interpretation:")
        lines.extend(f"  - {clean_cell(value)}" for value in llm.get("materials", [])[:8])
    if llm.get("surface_finishes"):
        lines.append("- LLM surface/finish interpretation:")
        lines.extend(f"  - {clean_cell(value)}" for value in llm.get("surface_finishes", [])[:8])
    if llm.get("gdt_callouts"):
        lines.append("- LLM GD&T/form-control interpretation:")
        for item in llm.get("gdt_callouts", [])[:8]:
            if isinstance(item, dict):
                lines.append(f"  - {json.dumps(item, ensure_ascii=False)}")
            else:
                lines.append(f"  - {clean_cell(item)}")
    for label, key in (("Material evidence", "materials"), ("Surface/finish evidence", "finishes"), ("GD&T/form-control evidence", "gdt_lines")):
        values = evidence.get(key) or []
        if values:
            lines.append(f"- {label}:")
            lines.extend(f"  - {value}" for value in values[:8])
        else:
            lines.append(f"- {label}: not found in deterministic table extraction.")
    return lines


def validate_component(component: dict[str, Any]) -> list[str]:
    errors = []
    for key in ("name", "category_code", "attributes"):
        if key not in component:
            errors.append(f"missing required field: {key}")
    attrs = component.get("attributes")
    if not isinstance(attrs, dict):
        errors.append("attributes must be an object")
    try:
        import jsonschema  # type: ignore

        with Path("component_schema.json").open("r", encoding="utf-8") as f:
            schema = json.load(f)
        jsonschema.validate(component, schema)
    except ModuleNotFoundError:
        pass
    except Exception as exc:  # noqa: BLE001 - keep validation lightweight in batch output.
        errors.append(f"json_schema_validation_failed: {exc}")
    return errors


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")


def process_all(
    downloads: Path,
    output: Path,
    limit: int | None = None,
    sample: int | None = None,
    sample_seed: int | None = None,
    component_ids: list[str] | None = None,
    ollama: OllamaConfig | None = None,
    resume: bool = False,
    progress: bool = False,
) -> dict[str, Any]:
    component_dirs = [p for p in sorted(downloads.iterdir()) if p.is_dir()]
    if component_ids:
        wanted = set(component_ids)
        component_dirs = [p for p in component_dirs if p.name in wanted]
    if sample is not None:
        rng = random.Random(sample_seed)
        sample_size = min(sample, len(component_dirs))
        component_dirs = sorted(rng.sample(component_dirs, sample_size))
    if limit is not None:
        component_dirs = component_dirs[:limit]
    output.mkdir(parents=True, exist_ok=True)
    ready_rows = []
    review_rows = []
    summary_rows = []
    failures = []
    for offset, component_dir in enumerate(component_dirs, start=1):
        try:
            component_out = output / component_dir.name
            if resume and (component_out / "component.json").exists() and (component_out / "review.json").exists():
                if progress:
                    print(f"[{offset}/{len(component_dirs)}] skip {component_dir.name}", flush=True)
                try:
                    component = load_json(component_out / "component.json")
                    review_data = load_json(component_out / "review.json")
                    row = summary_row(component, review_data, component_out)
                    summary_rows.append(row)
                    (ready_rows if review_data.get("ready") else review_rows).append(row)
                except (OSError, json.JSONDecodeError, KeyError) as exc:
                    failures.append({"component_id": component_dir.name, "error": f"resume_load_failed: {exc!r}"})
                continue
            if progress:
                print(f"[{offset}/{len(component_dirs)}] process {component_dir.name}", flush=True)
            component, review, evidence = build_component(component_dir, ollama)
            validation_errors = validate_component(component)
            if validation_errors:
                review.flags.extend(f"schema_error:{error}" for error in validation_errors)
                review.ready = False
                review.confidence = min(review.confidence, 0.2)
            write_json(component_out / "component.json", component)
            write_json(
                component_out / "review.json",
                {
                    "component_id": review.component_id,
                    "confidence": review.confidence,
                    "ready": review.ready,
                    "flags": sorted(set(review.flags)),
                    "missing": review.missing,
                    "validation_errors": validation_errors,
                    "evidence": evidence,
                },
            )
            (component_out / "cad_instructions.md").write_text(
                build_instructions(component, review, evidence), encoding="utf-8"
            )
            row = summary_row(
                component,
                {
                    "component_id": review.component_id,
                    "confidence": review.confidence,
                    "ready": review.ready,
                    "flags": review.flags,
                    "missing": review.missing,
                },
                component_out,
            )
            summary_rows.append(row)
            (ready_rows if review.ready else review_rows).append(row)
        except Exception as exc:  # noqa: BLE001 - batch mode must continue and report component failures.
            failures.append({"component_id": component_dir.name, "error": repr(exc)})
    write_jsonl(output / "ready.jsonl", ready_rows)
    write_jsonl(output / "review_queue.jsonl", review_rows)
    write_json(output / "failures.json", failures)
    write_summary_csv(output / "summary.csv", summary_rows)
    return {
        "processed": len(component_dirs),
        "ready": len(ready_rows),
        "review": len(review_rows),
        "failures": len(failures),
        "output": str(output),
        "component_ids": [p.name for p in component_dirs],
    }


def summary_row(component: dict[str, Any], review_data: dict[str, Any], component_out: Path) -> dict[str, Any]:
    return {
        "component_id": review_data["component_id"],
        "name": component["name"],
        "category_code": component["category_code"],
        "category": component["attributes"].get("category", ""),
        "confidence": review_data.get("confidence", 0),
        "ready": review_data.get("ready", False),
        "flags": ";".join(sorted(set(review_data.get("flags") or []))),
        "missing": ";".join(review_data.get("missing") or []),
        "component_json": str((component_out / "component.json").as_posix()),
        "cad_instructions": str((component_out / "cad_instructions.md").as_posix()),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "component_id",
        "name",
        "category_code",
        "category",
        "confidence",
        "ready",
        "flags",
        "missing",
        "component_json",
        "cad_instructions",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads", type=Path, default=Path("downloads"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample", type=int, default=None, help="Randomly sample this many component folders before processing.")
    parser.add_argument("--sample-seed", type=int, default=None, help="Seed for reproducible --sample selection.")
    parser.add_argument("--component-id", action="append", default=None, help="Process a specific component id. Can be repeated.")
    parser.add_argument("--use-ollama", action="store_true", help="Run a local Ollama vision/text pass over tables and drawings.")
    parser.add_argument("--ollama-model", default="gemma4:latest")
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    parser.add_argument("--ollama-max-images", type=int, default=3)
    parser.add_argument("--ollama-timeout", type=int, default=180)
    parser.add_argument("--resume", action="store_true", help="Skip already generated component folders while rebuilding batch indexes.")
    parser.add_argument("--progress", action="store_true", help="Print per-component progress.")
    args = parser.parse_args()
    ollama = None
    if args.use_ollama:
        ollama = OllamaConfig(
            model=args.ollama_model,
            host=args.ollama_host,
            max_images=args.ollama_max_images,
            timeout=args.ollama_timeout,
        )
    result = process_all(
        args.downloads,
        args.output,
        args.limit,
        args.sample,
        args.sample_seed,
        args.component_id,
        ollama,
        args.resume,
        args.progress,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["failures"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
