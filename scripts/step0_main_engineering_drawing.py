#!/usr/bin/env python3
"""Step 0: identify and copy the main engineering drawing for each component."""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import sys
import tempfile
import textwrap
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Iterable

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw

from scripts.workflow_state import (
    DEFAULT_STATE_DB,
    component_dirs,
    connect_state_db,
    count_by_status,
    scan_downloads,
    selection_manifest_exists,
    update_component_status,
)


IMAGE_SUFFIXES = {".gif", ".png", ".jpg", ".jpeg", ".webp"}
DEFAULT_OLLAMA_MODEL = "qwen3.6:latest"
MODEL_SCORE_OVERRIDE_GAP = 0.75
MODEL_AREA_OVERRIDE_RATIO = 1.5
NOISE_NAME_PARTS = (
    "product_photo",
    "cleaning",
    "package",
    "bluecircle",
    "rightarrow",
    "icon",
    "anchor",
    "ancher",
)


@dataclass
class Step0Event:
    status: str
    component_id: str
    component_path: str
    message: str
    selected_drawing: str | None = None
    confidence: float | None = None
    method: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def drawing_files(component_dir: Path) -> list[Path]:
    drawings_dir = component_dir / "drawings"
    if not drawings_dir.exists():
        return []
    return sorted(
        path
        for path in drawings_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and path.stat().st_size > 0
        and not any(part in path.name.lower() for part in NOISE_NAME_PARTS)
    )


def image_size(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            return image.size
    except Exception:
        return (0, 0)


def heuristic_score(path: Path) -> float:
    name = path.name.lower()
    width, height = image_size(path)
    area = width * height
    score = area / 100_000

    if name.startswith(("drw", "draw")) or "_draw" in name:
        score += 7
    if "spec" in name:
        score += 3
    if name.startswith(("alt", "altr", "oth")):
        score -= 2
    if any(part in name for part in NOISE_NAME_PARTS):
        score -= 10
    if width < 120 or height < 90:
        score -= 6
    if width > 350 and height > 180:
        score += 2

    return score


def fallback_main_drawing(candidates: list[Path]) -> Path:
    if not candidates:
        raise FileNotFoundError("No drawing images found")
    return max(candidates, key=heuristic_score)


def ranked_candidates(component_dir: Path, max_candidates: int = 12) -> list[Path]:
    candidates = drawing_files(component_dir)
    return sorted(candidates, key=heuristic_score, reverse=True)[:max_candidates]


def build_contact_sheet(candidates: list[Path]) -> Path:
    thumb_width = 360
    label_height = 34
    padding = 14
    columns = 2 if len(candidates) > 1 else 1
    rows = (len(candidates) + columns - 1) // columns
    cell_width = thumb_width + padding * 2
    cell_height = 260 + label_height + padding * 2
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(sheet)

    for index, path in enumerate(candidates):
        col = index % columns
        row = index // columns
        x0 = col * cell_width + padding
        y0 = row * cell_height + padding
        label = f"{index + 1}. {path.name}"

        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                image.thumbnail((thumb_width, 250), Image.Resampling.LANCZOS)
                sheet.paste(image, (x0, y0 + label_height))
        except Exception:
            draw.rectangle((x0, y0 + label_height, x0 + thumb_width, y0 + 250), outline="red")
            draw.text((x0 + 8, y0 + label_height + 8), "unreadable image", fill="red")

        draw.text((x0, y0), label[:58], fill="black")

    tmp = tempfile.NamedTemporaryFile(prefix="cad_step0_contact_", suffix=".png", delete=False)
    tmp.close()
    sheet.save(tmp.name)
    return Path(tmp.name)


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                return {}
    return {}


def parse_ollama_generate_response(raw: dict[str, Any]) -> tuple[dict[str, Any], str]:
    for field in ("response", "thinking"):
        parsed = parse_json_object(str(raw.get(field, "")))
        if parsed:
            return parsed, field
    return {}, "none"


def ask_ollama_for_main_drawing(candidates: list[Path], model: str) -> dict[str, Any]:
    contact_sheet = build_contact_sheet(candidates)
    try:
        image_b64 = base64.b64encode(contact_sheet.read_bytes()).decode("ascii")
        candidate_lines = "\n".join(f"- {path.name}" for path in candidates)
        prompt = f"""
Choose the single image that is most likely the main dimensioned engineering drawing for CAD reconstruction.

Prefer the most complete catalog-family template drawing: multi-view orthographic mechanical drawings with dimensions,
section views, feature callouts, GD&T/surface symbols, and named dimensional parameters.
Do not choose detail-only, inset, alternate-shape, or option drawings when a larger full template drawing is present.
Reject product photos, cleaning process graphics, packaging examples, arrows/icons, and pure datasheet/table images.

Candidate filenames:
{candidate_lines}

Return only compact JSON with this exact shape:
{{"filename":"one candidate filename","confidence":0.0,"reason":"short reason"}}
"""
        payload = {
            "model": model,
            "prompt": textwrap.dedent(prompt).strip(),
            "images": [image_b64],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        }
        request = urllib.request.Request(
            "http://127.0.0.1:11434/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=180) as response:
            raw = json.loads(response.read().decode("utf-8"))
        parsed, parse_source = parse_ollama_generate_response(raw)
        parsed["_method"] = "ollama"
        parsed["_parse_source"] = parse_source
        return parsed
    finally:
        try:
            contact_sheet.unlink()
        except FileNotFoundError:
            pass


def match_candidate(filename: str, candidates: list[Path]) -> Path | None:
    normalized = filename.strip().lower()
    for candidate in candidates:
        if candidate.name.lower() == normalized:
            return candidate
    for candidate in candidates:
        if normalized and normalized in candidate.name.lower():
            return candidate
    return None


def image_area(path: Path) -> int:
    width, height = image_size(path)
    return width * height


def numeric_confidence(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def apply_model_selection_guard(selected: Path, candidates: list[Path], result: dict[str, Any]) -> Path:
    if not candidates:
        return selected
    top_candidate = candidates[0]
    if selected == top_candidate:
        return selected

    top_score = heuristic_score(top_candidate)
    selected_score = heuristic_score(selected)
    top_area = image_area(top_candidate)
    selected_area = max(1, image_area(selected))
    score_gap = top_score - selected_score
    area_ratio = top_area / selected_area

    if score_gap >= MODEL_SCORE_OVERRIDE_GAP and area_ratio >= MODEL_AREA_OVERRIDE_RATIO:
        result["_method"] = "ollama_guarded_heuristic"
        result["_ollama_filename"] = result.get("filename")
        result["_guard_reason"] = (
            f"Top ranked candidate was materially stronger: score_gap={score_gap:.2f}, "
            f"area_ratio={area_ratio:.2f}"
        )
        result["filename"] = top_candidate.name
        result["confidence"] = min(numeric_confidence(result.get("confidence")), 0.75)
        return top_candidate

    return selected


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")
        tmp_path = Path(handle.name)
    tmp_path.replace(path)


def clear_main_drawing_images(main_dir: Path) -> None:
    if not main_dir.exists():
        return
    for path in main_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            path.unlink()


def select_main_engineering_drawing(component_dir: Path, model: str, *, force: bool = False) -> dict[str, Any]:
    candidates = drawing_files(component_dir)
    if not candidates:
        raise FileNotFoundError(f"No usable drawing files in {component_dir / 'drawings'}")

    selected: Path | None = None
    result: dict[str, Any] = {}
    if len(candidates) == 1:
        selected = candidates[0]
        result = {"filename": selected.name, "confidence": 1.0, "reason": "Only usable drawing image", "_method": "single"}
    else:
        short_list = ranked_candidates(component_dir)
        try:
            result = ask_ollama_for_main_drawing(short_list, model)
            selected = match_candidate(str(result.get("filename", "")), short_list)
            if selected is not None:
                selected = apply_model_selection_guard(selected, short_list, result)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            result = {"reason": f"Ollama unavailable or failed: {exc}", "_method": "heuristic_after_ollama_failure"}

    if selected is None:
        selected = fallback_main_drawing(candidates)
        result.setdefault("filename", selected.name)
        result.setdefault("confidence", 0.0)
        result.setdefault("reason", "Selected by filename and image-size heuristic")
        result["_method"] = result.get("_method", "heuristic")
        result["_fallback_selected"] = selected.name

    main_dir = component_dir / "main_engineering_drawing"
    main_dir.mkdir(parents=True, exist_ok=True)
    if force:
        clear_main_drawing_images(main_dir)
    destination = main_dir / selected.name
    shutil.copy2(selected, destination)

    manifest = {
        "component_id": component_dir.name,
        "source_path": str(selected),
        "copied_path": str(destination),
        "ollama_model": model,
        "selection": result,
    }
    atomic_write_json(main_dir / "selection.json", manifest)
    return manifest


def select_component_subset(downloads_dir: Path, limit: int | None, component_dirs_arg: list[Path]) -> list[Path]:
    if component_dirs_arg:
        components = component_dirs_arg
    else:
        components = component_dirs(downloads_dir)
    if limit is not None:
        components = components[:limit]
    return components


def event_from_manifest(status: str, manifest: dict[str, Any], message: str) -> Step0Event:
    selection = manifest.get("selection", {})
    return Step0Event(
        status=status,
        component_id=str(manifest.get("component_id", "")),
        component_path=str(Path(manifest.get("copied_path", "")).parents[1]) if manifest.get("copied_path") else "",
        message=message,
        selected_drawing=Path(manifest["copied_path"]).name if manifest.get("copied_path") else None,
        confidence=selection.get("confidence"),
        method=selection.get("_method"),
        error=None,
    )


def read_selection_manifest(component: Path) -> dict[str, Any] | None:
    manifest_path = component / "main_engineering_drawing" / "selection.json"
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def process_component(component: Path, model: str, force: bool = False) -> Step0Event:
    if selection_manifest_exists(component) and not force:
        manifest = read_selection_manifest(component)
        if manifest:
            event = event_from_manifest("skipped", manifest, "Existing Step 0 output skipped")
            event.component_id = component.name
            event.component_path = str(component)
            return event
        return Step0Event(
            status="error",
            component_id=component.name,
            component_path=str(component),
            message="Existing selection.json is unreadable",
            error="Existing selection.json is unreadable",
        )

    try:
        manifest = select_main_engineering_drawing(component, model, force=force)
        return event_from_manifest("done", manifest, "Selected main engineering drawing")
    except Exception as exc:
        return Step0Event(
            status="error",
            component_id=component.name,
            component_path=str(component),
            message=str(exc),
            error=str(exc),
        )


def iter_step0_events(
    components: list[Path],
    model: str,
    *,
    state_db: str | Path = DEFAULT_STATE_DB,
    force: bool = False,
    workers: int = 1,
) -> Iterable[Step0Event]:
    conn = connect_state_db(state_db)
    scan_downloads(conn, Path("."), component_paths=components)

    def mark_running(component: Path) -> Step0Event:
        update_component_status(conn, component.name, "running")
        return Step0Event(
            status="running",
            component_id=component.name,
            component_path=str(component),
            message="Selecting main engineering drawing",
        )

    def persist(event: Step0Event) -> None:
        persistent_status = "skipped" if event.status == "skipped" else event.status
        update_component_status(
            conn,
            event.component_id,
            persistent_status,
            selected_drawing=event.selected_drawing,
            confidence=event.confidence,
            method=event.method,
            error=event.error,
        )

    workers = max(1, workers)
    if workers == 1:
        for component in components:
            yield mark_running(component)
            event = process_component(component, model, force=force)
            persist(event)
            yield event
        conn.close()
        return

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_component = {}
        for component in components:
            yield mark_running(component)
            future_to_component[executor.submit(process_component, component, model, force)] = component
        for future in as_completed(future_to_component):
            event = future.result()
            persist(event)
            yield event
    conn.close()


def run_step0_batch(
    components: list[Path],
    model: str,
    *,
    state_db: str | Path = DEFAULT_STATE_DB,
    force: bool = False,
    workers: int = 1,
    on_event: Callable[[Step0Event], None] | None = None,
) -> tuple[list[Step0Event], list[Step0Event]]:
    completed = []
    errors = []
    for event in iter_step0_events(
        components,
        model,
        state_db=state_db,
        force=force,
        workers=workers,
    ):
        if on_event:
            on_event(event)
        if event.status in {"done", "skipped"}:
            completed.append(event)
        elif event.status == "error":
            errors.append(event)
    return completed, errors


def run_step0(components: list[Path], model: str) -> tuple[list[dict[str, Any]], list[str]]:
    manifests = []
    errors = []
    for component in components:
        try:
            manifests.append(select_main_engineering_drawing(component, model, force=True))
        except Exception as exc:
            errors.append(f"{component.name}: {exc}")
    return manifests, errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 0: select each component's main engineering drawing with local Ollama.")
    parser.add_argument("--downloads-dir", default="downloads", type=Path)
    parser.add_argument("--component-dir", action="append", default=[], type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ollama-model", default=os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL))
    parser.add_argument("--state-db", default=DEFAULT_STATE_DB, type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--status", choices=["all", "pending", "error", "done"], default="all")
    parser.add_argument("--json", action="store_true", help="Emit JSONL progress events and final summary.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    conn = connect_state_db(args.state_db)
    components = select_component_subset(args.downloads_dir, args.limit, args.component_dir)
    scan_downloads(conn, args.downloads_dir, args.limit, args.component_dir)
    if args.status != "all":
        status_set = {"done", "skipped"} if args.status == "done" else {args.status}
        components = [
            component
            for component in components
            if conn.execute(
                "SELECT step0_status FROM components WHERE component_id = ?",
                (component.name,),
            ).fetchone()["step0_status"] in status_set
        ]
    conn.close()

    def print_event(event: Step0Event) -> None:
        if args.json:
            print(json.dumps({"type": "event", **event.to_dict()}), flush=True)
        elif event.status != "running":
            detail = event.selected_drawing or event.error or event.message
            print(f"{event.component_id}: {event.status} {detail}", flush=True)

    completed, errors = run_step0_batch(
        components,
        args.ollama_model,
        state_db=args.state_db,
        force=args.force,
        workers=args.workers,
        on_event=print_event,
    )

    conn = connect_state_db(args.state_db)
    summary = count_by_status(conn)
    conn.close()
    if args.json:
        print(json.dumps({"type": "summary", **summary}), flush=True)
    else:
        print(
            "summary: "
            + " ".join(f"{key}={summary[key]}" for key in ["total", "pending", "running", "done", "skipped", "error"])
        )

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
