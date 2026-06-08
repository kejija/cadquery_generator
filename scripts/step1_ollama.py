#!/usr/bin/env python3
"""Step 1 (Ollama backend): run the same drawing + datasheet -> engineering feature
template prompt as step1_drawing_datasheetjson.py, but call a local Ollama
model instead of OpenAI's Responses API.

Differences from the OpenAI path:
- No strict JSON schema enforcement; we use Ollama's `format: "json"` mode
  (loose JSON object, not a schema).
- No prompt caching, no reasoning-effort control, no usage telemetry.
- One request at a time (Ollama's /api/generate is sync, not batch).
- The image is sent as a base64 string in the `images` field.

After generation, this script applies the same structural_issues checks the
upstream eval uses (bad_parameter_refs, bad_dependency_refs, bad_body_refs,
null_driving_dimensions) so results are directly comparable to a gpt-5.4-mini
batch output.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Reuse the exact prompt + helpers from the OpenAI-side script so the
# comparison is fair.
from scripts.step1_drawing_datasheetjson import (  # noqa: E402
    DEVELOPER_MESSAGE,
    USER_PROMPT,
    find_main_drawing,
    find_specs_json,
)

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3.6:latest"
DEFAULT_OUTPUT_DIR = Path("output/feature_templates_ollama")
DEFAULT_EVAL_DIR = Path("output")
REQUEST_TIMEOUT_S = 600


# ---------------------------------------------------------------------------
# Ollama call
# ---------------------------------------------------------------------------


def _encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def call_ollama(
    *,
    ollama_url: str,
    model: str,
    drawing_path: Path,
    specs_json_text: str,
    temperature: float = 0.2,
    timeout_s: int = REQUEST_TIMEOUT_S,
) -> dict[str, Any]:
    """Call Ollama /api/generate with the developer+user prompt and the drawing.

    Returns the parsed JSON body. Raises on transport or non-2xx.
    """
    # The Ollama chat template does the role separation. We pass the developer
    # block in the system slot and merge user_prompt + datasheet into the
    # user slot. Drawing is attached as `images`.
    user_text = f"{USER_PROMPT.strip()}\n\nDatasheet JSON:\n{specs_json_text}"

    payload = {
        "model": model,
        "prompt": user_text,
        "system": DEVELOPER_MESSAGE,
        "images": [_encode_image(drawing_path)],
        "format": "json",
        "stream": False,
        "options": {
            "temperature": temperature,
        },
    }

    url = f"{ollama_url.rstrip('/')}/api/generate"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body_text = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {e.code} from {url}: {body_text[:500]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama unreachable at {url}: {e}") from e

    if "error" in body:
        raise RuntimeError(f"Ollama returned error: {body['error']}")

    return body


def extract_template(response_body: dict[str, Any]) -> dict[str, Any]:
    """Pull the JSON template out of an Ollama /api/generate response.

    Some reasoning-capable Ollama models (qwen3.6 in particular) emit a
    `thinking` block but leave `response` empty when the model decides the
    reasoning budget consumed the whole turn. We treat `response` as the
    primary payload and fall back to `thinking` only if it parses as JSON.
    """
    text = response_body.get("response") or ""
    if not text:
        thinking = response_body.get("thinking", "")
        # Try the thinking block as a last resort
        if thinking:
            text = thinking
        else:
            raise RuntimeError(
                f"Ollama returned empty response. Keys: {list(response_body.keys())}"
            )
    # Be defensive: strip code fences if the model emitted them despite the
    # format constraint.
    text = text.strip()
    if text.startswith("```"):
        # ```json ... ```
        first_nl = text.find("\n")
        text = text[first_nl + 1 :] if first_nl != -1 else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    return json.loads(text)


# ---------------------------------------------------------------------------
# Structural eval (mirrors the deleted step1_batch_eval_*.json schema)
# ---------------------------------------------------------------------------


REQUIRED_TOP_KEYS = {
    "schema_version",
    "template_id",
    "part_family",
    "units",
    "modeling_mode",
    "coordinate_system",
    "components",
    "parameters",
    "feature_graph",
    "catalog_metadata",
    "assumptions",
    "missing_information",
    "validation_requirements",
}


def evaluate_template(template: dict[str, Any]) -> dict[str, Any]:
    """Apply structural checks to a feature template.

    We compute two parallel sets of measurements:

    schema_strict: exactly what the upstream gpt-5.4-mini eval measures
                   (only counts features in `feature_graph[]`, params in
                   `parameters[]` with the strict shape, etc.). Local models
                   routinely use different key names (`features` vs
                   `feature_graph`, `validation_checks` vs
                   `validation_requirements`) so this number is often 0.

    engineering:  a tolerant pass that also looks at the model-native keys.
                  This is the number that actually reflects whether the
                  model understood the engineering task.

    Returns a dict with both views, plus structural_issues and cad_ready.
    """
    # ----- resolve candidate key sets ----------------------------------
    # Helper: coerce a field that "should" be a list to a list. Some models
    # emit feature_graph as a dict (e.g. {"F1": {...}, "F2": {...}}); others
    # emit `parameters` as a dict keyed by symbol. We unwrap to the values
    # in that case. Missing/None becomes [].
    def _as_list(v: Any) -> list[Any]:
        if v is None:
            return []
        if isinstance(v, list):
            return v
        if isinstance(v, dict):
            return [item for item in v.values() if item is not None]
        return []

    # Some models wrap the whole template under a single key (e.g.
    # gemini-3-flash-preview on part 110302255160 emitted
    # {"part_specification": {parameters, features, ...}}). Detect that and
    # look one level deep.
    _WRAPPER_KEYS = ("part_specification", "specification", "template", "result")
    def _unwrap(t: dict[str, Any]) -> dict[str, Any]:
        for k in _WRAPPER_KEYS:
            inner = t.get(k)
            if isinstance(inner, dict) and any(
                fld in inner for fld in ("parameters", "features", "feature_graph",
                                          "geometric_features", "components",
                                          "validation_checks", "validation_requirements",
                                          "part_family_metadata", "metadata",
                                          "missing_information", "assumptions")
            ):
                # Merge wrapper-into-top-level: prefer the wrapper for any
                # field it provides, fall back to the original top-level.
                merged = dict(t)
                merged.update(inner)
                # Preserve the wrapper itself for debugging/inspection
                merged["_wrapper_detected"] = k
                return merged
        return t

    template = _unwrap(template)

    schema_params_list = _as_list(template.get("parameters"))
    schema_features_list = _as_list(template.get("feature_graph"))
    schema_components_list = _as_list(template.get("components"))
    schema_validation_list = _as_list(template.get("validation_requirements"))
    schema_catalog = template.get("catalog_metadata", {}) or {}
    schema_missing = _as_list(template.get("missing_information"))
    schema_assumptions = _as_list(template.get("assumptions"))

    # Native shapes observed on local Ollama models:
    #   parameters[{symbol, description, unit|units, ...}]
    #   features[{feature_id, feature_type, ...}]   (qwen3.6 / minimax-m3)
    #   geometric_features[{feature_id, ...}]       (gemini-3-flash-preview)
    #   validation_checks[{check_type, ...}]
    #   metadata{...} / part_family_metadata{...}
    native_params_list = _as_list(template.get("parameters"))
    # Union of all observed feature-list key names. Pick the first non-empty.
    native_features_list = (
        _as_list(template.get("geometric_features"))
        or _as_list(template.get("features"))
    )
    native_validation_list = _as_list(template.get("validation_checks"))
    native_catalog = (
        template.get("part_family_metadata")
        or template.get("metadata")
        or {}
    )
    # `assumptions` is sometimes the same; check both
    native_assumptions = native_catalog.get("assumptions", []) if isinstance(native_catalog, dict) else []
    if not native_assumptions:
        native_assumptions = template.get("assumptions", []) or []

    # ----- parameter symbol/feature id extraction ----------------------
    def param_symbol(p: Any) -> str | None:
        if not isinstance(p, dict):
            return None
        # schema uses 'symbol' like "$D"; native also uses 'symbol'
        s = p.get("symbol")
        if s:
            return s
        # fallback: native-style name
        n = p.get("name")
        if n:
            return f"${n}"
        return None

    def feature_id(f: Any) -> str | None:
        if not isinstance(f, dict):
            return None
        return f.get("id") or f.get("feature_id")

    def feature_dep(f: Any) -> list[Any]:
        if not isinstance(f, dict):
            return []
        raw = f.get("depends_on") or f.get("dependencies") or []
        out: list[Any] = []
        for item in raw:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, list):
                for sub in item:
                    if isinstance(sub, str):
                        out.append(sub)
            elif item is not None:
                out.append(str(item))
        return out

    def feature_target_bodies(f: Any) -> list[Any]:
        if not isinstance(f, dict):
            return []
        raw = f.get("target_bodies") or ([f.get("target_body")] if f.get("target_body") else [])
        # Flatten and stringify: some models nest lists, or mix strings+lists
        out: list[Any] = []
        for item in raw:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, list):
                for sub in item:
                    if isinstance(sub, str):
                        out.append(sub)
            elif item is not None:
                out.append(str(item))
        return out

    def feature_output_bodies(f: Any) -> list[Any]:
        if not isinstance(f, dict):
            return []
        return f.get("output_bodies") or []

    def feature_driving_dim(f: Any) -> list[dict[str, Any]]:
        if not isinstance(f, dict):
            return []
        out: list[dict[str, Any]] = []
        # schema: parameters: [{name, value, role, ...}]
        for p in f.get("parameters", []) or []:
            if isinstance(p, dict) and p.get("role") == "driving_dimension":
                out.append(p)
        # native: symbolic_driving_parameters: ["$D", ...]
        for s in f.get("symbolic_driving_parameters", []) or []:
            if isinstance(s, str):
                out.append({"name": s, "value": None, "role": "driving_dimension"})
        return out

    def needs_review_flag(f: Any) -> bool:
        if not isinstance(f, dict):
            return False
        v = f.get("needs_review")
        if v is not None:
            return bool(v)
        return False

    def confidence_value(f: Any) -> float | None:
        if not isinstance(f, dict):
            return None
        v = f.get("confidence")
        if v is None:
            v = f.get("confidence_score")
        if isinstance(v, (int, float)):
            return float(v)
        return None

    # ----- count helpers ------------------------------------------------
    def count_view(
        param_list: list[Any],
        feat_list: list[Any],
        val_list: list[Any],
        missing_list: list[Any],
        assumption_list: list[Any],
    ) -> dict[str, Any]:
        params_with_default = sum(
            1
            for p in param_list
            if isinstance(p, dict)
            and (p.get("default_value") is not None
                 or p.get("value") is not None)
            and p.get("default_value") != ""
            and p.get("value") != ""
        )
        needs_review_features = sum(1 for f in feat_list if needs_review_flag(f))
        confs = [confidence_value(f) for f in feat_list]
        confs = [c for c in confs if c is not None]
        avg_conf = round(sum(confs) / len(confs), 3) if confs else 0.0
        return {
            "parameters": len(param_list),
            "parameters_with_default": params_with_default,
            "features": len(feat_list),
            "needs_review_features": needs_review_features,
            "validation_requirements": len(val_list),
            "missing_information": len(missing_list),
            "assumptions": len(assumption_list),
            "avg_feature_confidence": avg_conf,
        }

    schema_counts = count_view(
        schema_params_list, schema_features_list, schema_validation_list,
        schema_missing, schema_assumptions,
    )
    engineering_counts = count_view(
        native_params_list, native_features_list, native_validation_list,
        schema_missing, native_assumptions,
    )
    # engineering view: prefer native feature list when it has content
    if native_features_list and not schema_features_list:
        engineering_counts["features"] = len(native_features_list)
    elif schema_features_list and not native_features_list:
        engineering_counts["features"] = len(schema_features_list)
    # engineering view: prefer native validation list when it has content;
    # fall back to the schema's validation_requirements
    if not native_validation_list and schema_validation_list:
        engineering_counts["validation_requirements"] = len(schema_validation_list)

    # ----- structural issues (combined view) ----------------------------
    component_ids: set[Any] = set()
    for c in schema_components_list + (template.get("components", []) or []):
        if isinstance(c, dict) and c.get("id"):
            component_ids.add(c["id"])

    parameter_symbols: set[str] = set()
    parameter_symbols_bare: set[str] = set()
    for p in schema_params_list + native_params_list:
        s = param_symbol(p)
        if s:
            parameter_symbols.add(s)
            # also track the bare form (without leading $) for ref lookups
            if s.startswith("$"):
                parameter_symbols_bare.add(s[1:])
            else:
                parameter_symbols_bare.add(s)

    feature_ids: set[Any] = set()
    for f in schema_features_list + native_features_list:
        fid = feature_id(f)
        if fid:
            feature_ids.add(fid)

    bad_parameter_refs: list[dict[str, str]] = []
    for f in schema_features_list + native_features_list:
        fid = feature_id(f) or ""
        # schema-style: parameters: [{value: "$X", ...}]
        for p in f.get("parameters", []) or []:
            if not isinstance(p, dict):
                continue
            v = p.get("value")
            if isinstance(v, str) and v.startswith("$"):
                bare = v[1:]
                if bare and bare not in parameter_symbols_bare and bare not in feature_ids:
                    bad_parameter_refs.append({"feature": fid, "symbol": v})
        # native-style: symbolic_driving_parameters: ["$X", ...]
        for s in f.get("symbolic_driving_parameters", []) or []:
            if isinstance(s, str) and s.startswith("$"):
                bare = s[1:]
                if bare and bare not in parameter_symbols_bare and bare not in feature_ids:
                    bad_parameter_refs.append({"feature": fid, "symbol": s})

    bad_dependency_refs: list[dict[str, str]] = []
    for f in schema_features_list + native_features_list:
        fid = feature_id(f) or ""
        for dep in feature_dep(f):
            if dep and dep not in feature_ids and dep != fid:
                bad_dependency_refs.append({"feature": fid, "dependency": str(dep)})

    bad_body_refs: list[dict[str, str]] = []
    for f in schema_features_list + native_features_list:
        fid = feature_id(f) or ""
        for body in feature_target_bodies(f):
            if body and body not in component_ids:
                bad_body_refs.append({"feature": fid, "body": str(body)})

    null_driving_dimensions: list[dict[str, str]] = []
    for f in schema_features_list + native_features_list:
        fid = feature_id(f) or ""
        for dp in feature_driving_dim(f):
            v = dp.get("value")
            if v is None or v == "":
                null_driving_dimensions.append({
                    "feature": fid, "name": str(dp.get("name", "")),
                })

    structural_issues = {
        "bad_parameter_refs": bad_parameter_refs,
        "bad_dependency_refs": bad_dependency_refs,
        "bad_body_refs": bad_body_refs,
        "null_driving_dimensions": null_driving_dimensions,
    }
    issue_count = sum(len(v) for v in structural_issues.values())

    # CAD-ready is a function of the engineering view: features exist, no
    # dangling refs, no missing information, no review flags.
    eng_features = engineering_counts["features"]
    eng_validation = engineering_counts["validation_requirements"]
    cad_ready = (
        eng_features > 0
        and issue_count == 0
        and engineering_counts["missing_information"] == 0
        and engineering_counts["needs_review_features"] == 0
        and eng_validation > 0
    )

    return {
        "schema_strict": schema_counts,
        "engineering": engineering_counts,
        "structural_issues": structural_issues,
        "issue_count": issue_count,
        "cad_ready": cad_ready,
        "schema_compliant": bool(
            schema_features_list
            and schema_params_list
            and "feature_graph" in template
            and "validation_requirements" in template
        ),
    }


# ---------------------------------------------------------------------------
# Per-component driver
# ---------------------------------------------------------------------------


@dataclass
class ComponentResult:
    component_id: str
    drawing_path: str
    specs_path: str
    status: str  # "ok" | "json_error" | "transport_error" | "missing_inputs"
    template: dict | None = None
    raw_response: str | None = None
    error: str | None = None
    eval: dict | None = None
    duration_s: float = 0.0
    load_duration_s: float = 0.0
    eval_count: int | None = None
    prompt_eval_count: int | None = None


def process_component(
    component_dir: Path,
    *,
    ollama_url: str,
    model: str,
    temperature: float,
    output_dir: Path,
) -> ComponentResult:
    cid = component_dir.name
    result = ComponentResult(
        component_id=cid,
        drawing_path="",
        specs_path="",
        status="missing_inputs",
    )

    try:
        drawing = find_main_drawing(component_dir)
        specs = find_specs_json(component_dir)
    except FileNotFoundError as e:
        result.error = f"missing input: {e}"
        return result

    result.drawing_path = str(drawing)
    result.specs_path = str(specs)

    specs_json_text = json.dumps(json.loads(specs.read_text(encoding="utf-8")), ensure_ascii=False)

    t0 = time.time()
    try:
        body = call_ollama(
            ollama_url=ollama_url,
            model=model,
            drawing_path=drawing,
            specs_json_text=specs_json_text,
            temperature=temperature,
        )
    except Exception as e:
        result.status = "transport_error"
        result.error = str(e)
        result.duration_s = time.time() - t0
        return result
    result.duration_s = time.time() - t0
    result.load_duration_s = (body.get("load_duration") or 0) / 1e9
    result.eval_count = body.get("eval_count")
    result.prompt_eval_count = body.get("prompt_eval_count")

    try:
        template = extract_template(body)
    except json.JSONDecodeError as e:
        result.status = "json_error"
        result.error = f"could not parse template JSON: {e}"
        result.raw_response = body.get("response", "")[:2000]
        return result

    result.template = template
    result.eval = evaluate_template(template)
    result.status = "ok"

    # Persist
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{cid}.feature_template.json").write_text(
        json.dumps(template, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def discover_component_dirs(downloads_dir: Path, limit: int | None = None) -> list[Path]:
    out: list[Path] = []
    for child in sorted(downloads_dir.iterdir()):
        if not child.is_dir():
            continue
        if not (child / "main_engineering_drawing").exists():
            continue
        out.append(child)
        if limit and len(out) >= limit:
            break
    return out


def render_summary_table(results: list[ComponentResult]) -> str:
    cols = ("id", "status", "schema", "params", "feat", "review", "val", "miss", "issues", "cad_rdy", "secs")
    rows: list[tuple[str, str, str, str, str, str, str, str, str, str, str]] = []
    for r in results:
        ev = r.eval
        if ev is not None:
            eng = ev["engineering"]
            rows.append((
                r.component_id,
                r.status,
                "Y" if ev["schema_compliant"] else "n",
                str(eng["parameters"]),
                str(eng["features"]),
                str(eng["needs_review_features"]),
                str(eng["validation_requirements"]),
                str(eng["missing_information"]),
                str(ev["issue_count"]),
                str(ev["cad_ready"]),
                f"{r.duration_s:.1f}",
            ))
        else:
            rows.append((r.component_id, r.status, "-", "-", "-", "-", "-", "-", "-", "-", f"{r.duration_s:.1f}"))

    widths = [max(len(r[i]) for r in rows + [cols]) for i in range(len(cols))]
    lines = []
    lines.append("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    lines.append("  ".join("-" * w for w in widths))
    for row in rows:
        lines.append("  ".join(c.ljust(w) for c, w in zip(row, widths)))
    return "\n".join(lines)


def main() -> int:
    doc_first_line = (__doc__ or "").splitlines()[0] if __doc__ else "Step 1 (Ollama backend)"
    parser = argparse.ArgumentParser(description=doc_first_line)
    parser.add_argument("--downloads-dir", default="downloads", type=Path)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--eval-path", type=Path, default=None,
                        help="Path to write a combined eval JSON (default: <output-dir>/../step1_ollama_eval_<model>.json)")
    args = parser.parse_args()

    downloads_dir: Path = args.downloads_dir
    if not downloads_dir.is_dir():
        print(f"downloads dir not found: {downloads_dir}", file=sys.stderr)
        return 2

    component_dirs = discover_component_dirs(downloads_dir, args.limit)
    if not component_dirs:
        print(f"no components with main_engineering_drawing/ under {downloads_dir}", file=sys.stderr)
        return 2

    print(f"model:      {args.model}")
    print(f"ollama url: {args.ollama_url}")
    print(f"components: {len(component_dirs)}")
    print(f"output dir: {args.output_dir}")
    print()

    results: list[ComponentResult] = []
    for i, cdir in enumerate(component_dirs, 1):
        print(f"[{i}/{len(component_dirs)}] {cdir.name} ...", end=" ", flush=True)
        r = process_component(
            cdir,
            ollama_url=args.ollama_url,
            model=args.model,
            temperature=args.temperature,
            output_dir=args.output_dir,
        )
        if r.status == "ok" and r.eval is not None:
            eng = r.eval["engineering"]
            print(
                f"ok  schema={r.eval['schema_compliant']} "
                f"params={eng['parameters']} "
                f"features={eng['features']} "
                f"issues={r.eval['issue_count']} "
                f"cad_ready={r.eval['cad_ready']} "
                f"{r.duration_s:.1f}s"
            )
        else:
            print(f"{r.status}  ({r.error})")
        results.append(r)

    print()
    print(render_summary_table(results))

    # Persist a combined eval JSON
    eval_path: Path = args.eval_path or (
        args.output_dir.parent / f"step1_ollama_eval_{args.model.replace(':', '_').replace('/', '_')}.json"
    )
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "ollama_url": args.ollama_url,
        "temperature": args.temperature,
        "record_count": len(results),
        "ok_count": sum(1 for r in results if r.status == "ok"),
        "json_error_count": sum(1 for r in results if r.status == "json_error"),
        "transport_error_count": sum(1 for r in results if r.status == "transport_error"),
        "cad_ready_count": sum(1 for r in results if r.eval and r.eval["cad_ready"]),
        "records": [
            {
                "custom_id": r.component_id,
                "status": r.status,
                "duration_s": round(r.duration_s, 3),
                "load_duration_s": round(r.load_duration_s, 3),
                "eval_count": r.eval_count,
                "prompt_eval_count": r.prompt_eval_count,
                "drawing_path": r.drawing_path,
                "specs_path": r.specs_path,
                "error": r.error,
                "schema_strict": r.eval["schema_strict"] if r.eval else None,
                "engineering": r.eval["engineering"] if r.eval else None,
                "schema_compliant": r.eval["schema_compliant"] if r.eval else None,
                "structural_issues": r.eval["structural_issues"] if r.eval else None,
                "issue_count": r.eval["issue_count"] if r.eval else None,
                "cad_ready": r.eval["cad_ready"] if r.eval else None,
            }
            for r in results
        ],
    }
    eval_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    print(f"eval written to {eval_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
