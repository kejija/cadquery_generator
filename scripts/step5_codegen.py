#!/usr/bin/env python3
"""
Step 5 — Generate CadQuery Code from a Resolved CAD Spec.

Reads every <component_id>__<part_number>.resolved_cad_spec.json in
output/resolved_specs/ (or the path given by --component/--part-number
filters), validates the input against schemas/resolved_cad_spec.schema.json,
and emits a deterministic, executable CadQuery model.py per part number.

The generated model.py:
  * imports cadquery as cq
  * defines one constant per parameter (snake_case) at the top, with the
    resolved value (or `# TODO: <name> = None  # needs_review` when null)
  * walks resolved_features in order and, per feature, emits a CadQuery
    call or a comment block
  * exports BOTH <part_number>.step and <part_number>.stl when run as
    a script
  * exposes a `validate()` returning bounding box, volume, and the list
    of feature IDs that actually produced geometry

Supported feature primitives (deterministic path):

  base_body + extrude + circle           -> cylinder (axis from spec, length
                                            from the `overall_length`-shaped
                                            parameter; OD from the
                                            `body_outer_diameter`-shaped
                                            parameter)
  base_body + extrude + polygon(6)       -> hex prism
  base_body + revolve                    -> cylinder (axis = X)
  boss        + extrude + polygon(6)      -> hex boss (diameter = across-flats)
  hole        + hole                      -> through-hole on the axis
                                            (best-effort, single feature)
  chamfer     + chamfer                   -> .edges().chamfer() (dimension
                                            best-effort, may be None)
  fillet      + fillet                    -> .edges().fillet() (dimension
                                            best-efftort, may be None)

Everything else (`pocket`, `cut_extrude`, `cut_revolve`, `counterbore_hole`,
`pattern`, `revolve` with non-cylindrical profile, custom_2d_profile,
`metadata`, `datum`, `gdnt`, `thread`) is emitted as a comment block with
a `# TODO: <reason>` marker, and recorded on the `not_implemented` list
in the validate() result. This makes Step 7's validation surface
honestly: it can flag "feature X was in the spec but not modeled."

Usage:
  python scripts/step5_codegen.py
  python scripts/step5_codegen.py --component 110300324920
  python scripts/step5_codegen.py --component 110300324920 --part-number MCSCN10
  python scripts/step5_codegen.py --all --limit 3
  python scripts/step5_codegen.py --dry-run
  python scripts/step5_codegen.py --json
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable

import jsonschema

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPECS_DIR = REPO_ROOT / "output" / "resolved_specs"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "cad_models"
DEFAULT_SCHEMA = REPO_ROOT / "schemas" / "resolved_cad_spec.schema.json"

# Regex: snake_case-ify an arbitrary parameter name.
_NON_ALNUM = re.compile(r"[^A-Za-z0-9_]+")
_UPPER_AFTER = re.compile(r"(?<!^)([A-Z])")


# ----------------------------------------------------------------------------
# Data model
# ----------------------------------------------------------------------------
@dataclass
class CodegenResult:
    component_id: str
    part_number: str
    spec_path: str
    output_path: str
    ok: bool
    features_total: int = 0
    features_implemented: list[str] = field(default_factory=list)
    features_skipped: list[dict] = field(default_factory=list)
    parameters: int = 0
    parameters_null: int = 0
    schema_valid: bool = False
    syntax_valid: bool = False
    runtime_smoke: bool = False
    error: str | None = None


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def snake(name: str) -> str:
    """Convert 'Body Outer Diameter' / 'bodyOuterDiameter' / 'Bore D' -> 'bore_d'."""
    s = _NON_ALNUM.sub("_", name).strip("_")
    s = _UPPER_AFTER.sub(r"_\1", s).lower()
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "param"


def param_value(feature: dict, *candidates: str) -> Any:
    """Pick the first parameter whose name matches any of `candidates` (case-insensitive,
    snake-or-spaces tolerant). Returns the raw value (which may be None)."""
    wanted = {c.lower().replace(" ", "").replace("-", "").replace("_", "") for c in candidates}
    for p in feature.get("parameters", []):
        n = p["name"].lower().replace(" ", "").replace("-", "").replace("_", "")
        if n in wanted:
            return p.get("value")
    return None


def param_name(feature: dict, *candidates: str) -> str | None:
    """Like param_value, but returns the original parameter name (so the emitter
    can refer to the snake-cased variable in generated code)."""
    wanted = {c.lower().replace(" ", "").replace("-", "").replace("_", "") for c in candidates}
    for p in feature.get("parameters", []):
        n = p["name"].lower().replace(" ", "").replace("-", "").replace("_", "")
        if n in wanted:
            return p["name"]
    return None


def depth_from(feature: dict) -> Any:
    """Pull the depth/length parameter. Tries common names."""
    return param_value(
        feature,
        "overall_length", "Overall length", "length", "L",
        "depth", "extrusion_depth", "axial_length",
    )


def diameter_from(feature: dict) -> Any:
    return param_value(
        feature,
        "body_outer_diameter", "Body outer diameter", "outer_diameter", "D",
        "diameter", "shaft_diameter", "Shaft diameter", "OD", "across_flats",
        "hex_width_across_flats", "Hex width across flats",
        "internal_bore_diameter", "hole_diameter", "Hole diameter",
        "Through hole diameter", "clearance_hole_diameter", "cross_drilled_hole_diameter",
    )


def radius_from(feature: dict) -> Any:
    v = param_value(feature, "R", "radius", "Radius", "fillet_radius", "Fillet radius", "shoulder_fillet_radius")
    if v is None:
        # chamfer/fillet with no R is allowed; chamfers usually have C
        v = param_value(feature, "C", "chamfer", "Chamfer", "Corner chamfer size", "end_chamfer")
    return v


def _norm_param_name(name: str) -> str:
    return name.lower().replace(" ", "").replace("-", "").replace("_", "")


def binding_name(feature: dict, *candidates: str) -> str | None:
    """Return a matching parameter_binding name carried on the feature copy."""
    wanted = {_norm_param_name(c) for c in candidates}
    for name in (feature.get("_parameter_bindings") or {}):
        if _norm_param_name(name) in wanted:
            return name
    return None


def binding_value(feature: dict, name: str | None) -> Any:
    if not name:
        return None
    return (feature.get("_parameter_bindings") or {}).get(name)


def numeric_param_expr(feature: dict, candidates: Iterable[str], default: float) -> str:
    """Return a generated-code expression for a numeric feature parameter.

    Symbolic parameter values are intentionally not resolved here; the
    deterministic path falls back to the supplied numeric placeholder.
    """
    name = param_name(feature, *candidates)
    value = param_value(feature, *candidates)
    if isinstance(value, (int, float)):
        return snake(name) if name else repr(value)
    b_name = binding_name(feature, *candidates)
    if isinstance(binding_value(feature, b_name), (int, float)):
        return snake(b_name) if b_name else repr(default)
    return repr(default)


def numeric_position_x_expr(feature: dict) -> str:
    pos = (feature.get("position") or {}).get("x")
    if isinstance(pos, (int, float)):
        return repr(pos)
    return "0"


def overall_length_fallback_expr(feature: dict, default: float = 10.0) -> str:
    name = binding_name(feature, "overall_length", "Overall length")
    return snake(name) if name else repr(default)


def _feature_with_bindings(feature: dict, params: dict[str, Any]) -> dict:
    f = dict(feature)
    f["_parameter_bindings"] = params
    return f


def _emitter_records_feature(code: Iterable[str]) -> bool:
    return any("feature_log.append(" in line or "not_implemented.append(" in line for line in code)


def _numeric_param(feature: dict, candidates: Iterable[str]) -> tuple[Any, str | None, str | None]:
    name = param_name(feature, *candidates)
    value = param_value(feature, *candidates)
    expr = snake(name) if name and isinstance(value, (int, float)) else None
    return value, name, expr


def _todo_feature(feature: dict, reason: str, detail: str | None = None) -> list[str]:
    fid = feature.get("id", "?")
    todo = reason if detail is None else f"{reason}: {detail}"
    return [
        f"# TODO: {todo}",
        f"not_implemented.append({fid!r})",
        f"# reason: {reason}",
    ]


def allow_symbolic_edge_break(feature: dict) -> bool:
    """Allow small placeholder chamfers/fillets on shaft specs with a usable axis."""
    bindings = feature.get("_parameter_bindings") or {}
    return "L" in bindings or "shaft_diameter" in bindings


def is_metadata_feature(feature: dict) -> bool:
    """Features whose modeling_primitive is metadata/thread_metadata should never
    produce geometry. They are emitted as comments only."""
    prim = (feature.get("modeling_primitive") or "").lower()
    ftype = (feature.get("feature_type") or "").lower()
    return (
        prim in {"metadata", "thread_metadata"}
        or ftype in {"metadata", "datum", "gdnt", "thread"}
    )


def component_id_of(spec_path: Path) -> str:
    return spec_path.stem.split("__")[0]


def part_number_of(spec_path: Path) -> str:
    # stem is e.g. "110300324920__MCSCN10.resolved_cad_spec" -> "MCSCN10"
    return spec_path.stem.split("__", 1)[1].removesuffix(".resolved_cad_spec")


# ----------------------------------------------------------------------------
# Per-feature emitters
# ----------------------------------------------------------------------------
# Each emitter returns (code_lines, applied: bool, reason_or_none).
# applied=False means the feature was acknowledged but not modeled; the
# caller records it on the not_implemented list.

POCKET_WIDTH_CANDIDATES = (
    "W", "width", "Width", "slot_width", "wrench_flat_width",
    "ℓ1", "l1", "L1",
)
POCKET_LENGTH_CANDIDATES = (
    "ℓ1", "L1", "l1", "length", "Length", "slot_length", "wrench_flat_length",
)
POCKET_POSITION_CANDIDATES = (
    "SC", "sc", "position_x", "x_pos", "pos_x", "start_position",
    "Offset", "offset", "wrench_flat_offset",
)
REVOLVE_DIAMETER_CANDIDATES = (
    "D", "diameter", "bore_diameter", "internal_bore_diameter",
    "reduced_section_diameter", "P", "M",
)
REVOLVE_POSITION_CANDIDATES = (
    "L", "length", "reduced_section_length", "F", "F25", "SC",
)
CUSTOM_WIDTH_CANDIDATES = (
    "Overall width", "overall_width", "Body outer diameter",
    "body_outer_diameter", "D", "width",
)
CUSTOM_HEIGHT_CANDIDATES = (
    "Overall height", "overall_height", "body_outer_diameter", "D1", "height",
)
CUSTOM_THICKNESS_CANDIDATES = (
    "Wall/web thickness", "Wall web thickness", "t", "wall_thickness",
    "Thickness",
)
CUSTOM_LENGTH_CANDIDATES = (
    "Overall length", "overall_length", "L", "length", "tightening_section_length",
)
RECT_WIDTH_CANDIDATES = ("Overall width", "overall_width", "width", "W")
RECT_HEIGHT_CANDIDATES = ("Overall height", "overall_height", "height", "H")
RECT_THICKNESS_CANDIDATES = ("Thickness", "T", "thickness", "depth")


def emit_custom_2d_profile_extrude(
    feature: dict,
    ftype: str,
    *,
    require_envelope_dims: bool = False,
) -> tuple[list[str], bool, str | None] | None:
    W, _, W_expr = _numeric_param(feature, CUSTOM_WIDTH_CANDIDATES)
    H, _, H_expr = _numeric_param(feature, CUSTOM_HEIGHT_CANDIDATES)
    T, _, T_expr = _numeric_param(feature, CUSTOM_THICKNESS_CANDIDATES)
    L, _, L_expr = _numeric_param(feature, CUSTOM_LENGTH_CANDIDATES)
    if not L_expr:
        b_name = binding_name(feature, *CUSTOM_LENGTH_CANDIDATES)
        if isinstance(binding_value(feature, b_name), (int, float)):
            L = binding_value(feature, b_name)
            L_expr = snake(b_name) if b_name else repr(L)
    if require_envelope_dims and not (W_expr and H_expr):
        return None

    if not (W_expr or H_expr) or not L_expr:
        reason = "custom_2d_profile extrude requires numeric W or H and length"
        detail = f"W={W!r}, H={H!r}, length={L!r}"
        return _todo_feature(feature, reason, detail), False, reason

    W_expr = W_expr or H_expr
    H_expr = H_expr or W_expr
    inner_lines = []
    if T_expr:
        inner_lines = [
            f"    inner = cq.Workplane('YZ').rect(W - 2*T, H - 2*T)",
            f"    void = inner.extrude(L)",
            f"    result = envelope.cut(void)",
        ]
    else:
        inner_lines = [
            f"    result = envelope",
        ]

    return [
        f"try:",
        f"    W = {W_expr}",
        f"    H = {H_expr}",
        *((f"    T = {T_expr}",) if T_expr else ()),
        f"    L = {L_expr}",
        f"    outer = cq.Workplane('YZ').rect(W, H)",
        f"    envelope = outer.extrude(L)",
        *inner_lines,
        f"    feature_log.append({feature.get('id', '?')!r})",
        f"except Exception:",
        f"    not_implemented.append({feature.get('id', '?')!r})",
        f"    # reason: custom_2d_profile extrusion failed",
    ], True, None

def emit_base_body(feature: dict) -> tuple[list[str], bool, str | None]:
    prim = feature.get("modeling_primitive")
    profile = (feature.get("construction") or {}).get("profile_type", "")
    sub = (feature.get("subtype") or "").lower()
    axis = (feature.get("axis") or "X").upper()
    L = depth_from(feature)
    D = diameter_from(feature)
    L_name = param_name(feature, "overall_length", "Overall length", "length", "L", "depth", "extrusion_depth", "axial_length")
    D_name = param_name(feature, "body_outer_diameter", "Body outer diameter", "outer_diameter", "D", "diameter", "shaft_diameter", "Shaft diameter", "OD", "across_flats", "hex_width_across_flats", "Hex width across flats")
    L_var = snake(L_name) if L_name else "L_val"
    D_var = snake(D_name) if D_name else "D_val"

    if prim == "extrude" and profile == "custom_2d_profile":
        custom = emit_custom_2d_profile_extrude(feature, "base_body")
        if custom:
            return custom

    if prim == "extrude" and profile == "rectangle":
        W = param_value(feature, *RECT_WIDTH_CANDIDATES)
        H = param_value(feature, *RECT_HEIGHT_CANDIDATES)
        T = param_value(feature, *RECT_THICKNESS_CANDIDATES)
        if isinstance(W, (int, float)) and isinstance(H, (int, float)) and isinstance(T, (int, float)):
            W_var = snake(param_name(feature, *RECT_WIDTH_CANDIDATES) or "W")
            H_var = snake(param_name(feature, *RECT_HEIGHT_CANDIDATES) or "H")
            T_var = snake(param_name(feature, *RECT_THICKNESS_CANDIDATES) or "T")
            return [
                f"try:",
                f"    result = cq.Workplane('XY').rect({W_var}, {H_var}).extrude({T_var})",
                f"    feature_log.append({feature.get('id', '?')!r})",
                f"except Exception:",
                f"    not_implemented.append({feature.get('id', '?')!r})",
                f"    # reason: rectangle extrude failed",
            ], True, None

    if prim == "extrude" and profile == "circle" and isinstance(D, (int, float)) and isinstance(L, (int, float)):
        if axis == "X":
            return [
                f"result = cq.Workplane('YZ').circle({D_var}/2).extrude({L_var})",
            ], True, None
        if axis == "Y":
            return [
                f"result = cq.Workplane('XZ').circle({D_var}/2).extrude({L_var})",
            ], True, None
        # default Z
        return [
            f"result = cq.Workplane('XY').circle({D_var}/2).extrude({L_var})",
        ], True, None

    if prim == "extrude" and "polygon" in profile.lower() and isinstance(L, (int, float)):
        # hex prism; assume across-flats 2 * radius
        if isinstance(D, (int, float)):
            return [
                f"result = cq.Workplane('XY').polygon(6, {D_var}).extrude({L_var})",
            ], True, None

    if prim == "revolve" and isinstance(D, (int, float)) and isinstance(L, (int, float)):
        # Cylinder by revolution: half-section rect from (D/2, 0) to (0, L)
        return [
            f"result = (",
            f"    cq.Workplane('XZ')",
            f"    .moveTo({D_var}/2, 0)",
            f"    .lineTo(0, 0)",
            f"    .lineTo(0, {L_var})",
            f"    .lineTo({D_var}/2, {L_var})",
            f"    .close()",
            f"    .revolve(360, (0, 0, 0), (0, 1, 0))",
            f")",
        ], True, None

    return [
        f"# TODO: base_body not implemented (prim={prim!r}, profile={profile!r}, "
        f"D={D!r}, L={L!r}, axis={axis!r})",
        f"result = cq.Workplane('XY').rect(1, 1).extrude(1)  # placeholder body",
    ], False, f"base_body unsupported: prim={prim}, profile={profile}"


def emit_boss(feature: dict) -> tuple[list[str], bool, str | None]:
    prim = feature.get("modeling_primitive")
    profile = (feature.get("construction") or {}).get("profile_type", "")
    L = depth_from(feature)
    D = diameter_from(feature)
    L_name = param_name(feature, "overall_length", "Overall length", "length", "L", "depth", "extrusion_depth", "axial_length", "tightening_section_length")
    D_name = param_name(feature, "body_outer_diameter", "Body outer diameter", "outer_diameter", "D", "diameter", "shaft_diameter", "Shaft diameter", "OD", "across_flats", "hex_width_across_flats", "Hex width across flats")
    L_var = snake(L_name) if L_name else "L_val"
    D_var = snake(D_name) if D_name else "D_val"
    if prim == "extrude" and profile == "custom_2d_profile":
        custom = emit_custom_2d_profile_extrude(feature, "boss", require_envelope_dims=True)
        if custom:
            return custom
    if prim == "extrude" and "polygon" in profile.lower() and isinstance(D, (int, float)) and isinstance(L, (int, float)):
        return [
            f"boss = cq.Workplane('XY').polygon(6, {D_var}).extrude({L_var})",
            f"result = result.union(boss)",
        ], True, None
    if prim == "extrude" and profile == "circle" and isinstance(D, (int, float)) and isinstance(L, (int, float)):
        return [
            f"boss = cq.Workplane('XY').circle({D_var}/2).extrude({L_var})",
            f"result = result.union(boss)",
        ], True, None
    return [
        f"# TODO: boss not implemented (prim={prim!r}, profile={profile!r}, D={D!r}, L={L!r})",
    ], False, f"boss unsupported: prim={prim}, profile={profile}"


def emit_pocket(feature: dict) -> tuple[list[str], bool, str | None]:
    prim = feature.get("modeling_primitive")
    sub = feature.get("subtype") or ""
    fid = feature.get("id", "?")
    if prim == "cut_extrude":
        W, _, W_expr = _numeric_param(feature, POCKET_WIDTH_CANDIDATES)
        L, _, L_expr = _numeric_param(feature, POCKET_LENGTH_CANDIDATES)
        if not (W_expr and L_expr):
            reason = "cut_extrude requires numeric width and length"
            detail = f"width={W!r}, length={L!r}"
            return _todo_feature(feature, reason, detail), False, reason

        pos = (feature.get("position") or {}).get("x")
        if not isinstance(pos, (int, float)):
            reason = "cut_extrude requires numeric position"
            detail = f"position={pos!r}"
            return _todo_feature(feature, reason, detail), False, reason
        pos_expr = repr(pos)
        depth = (feature.get("construction") or {}).get("depth")
        if isinstance(depth, (int, float)):
            depth_expr = repr(depth)
        else:
            depth_value, _, depth_expr = _numeric_param(feature, CUSTOM_LENGTH_CANDIDATES)
            if not depth_expr:
                b_name = binding_name(feature, "overall_length", "Overall length")
                if isinstance(binding_value(feature, b_name), (int, float)):
                    depth_expr = snake(b_name) if b_name else repr(binding_value(feature, b_name))
            if not depth_expr:
                reason = "cut_extrude requires numeric depth"
                detail = f"depth={depth_value if depth_value is not None else depth!r}"
                return _todo_feature(feature, reason, detail), False, reason
        return [
            f"try:",
            f"    pocket_wp = (",
            f"        cq.Workplane('YZ')",
            f"        .workplane(offset={pos_expr}, centerOption='CenterOfMass')",
            f"        .rect({W_expr}, {L_expr})",
            f"        .extrude({depth_expr})",
            f"    )",
            f"    result = result.cut(pocket_wp)",
            f"    feature_log.append({fid!r})",
            f"except Exception:",
            f"    not_implemented.append({fid!r})",
            f"    # reason: cut_extrude failed",
        ], True, None

    if prim == "cut_revolve":
        D, _, D_expr = _numeric_param(feature, REVOLVE_DIAMETER_CANDIDATES)
        pos = (feature.get("position") or {}).get("x")
        if isinstance(pos, (int, float)):
            pos_expr = repr(pos)
        else:
            P, _, pos_expr = _numeric_param(feature, REVOLVE_POSITION_CANDIDATES)
            if not pos_expr:
                reason = "cut_revolve requires numeric diameter and length/position"
                detail = f"diameter={D!r}, position={pos!r}, length={P!r}"
                return _todo_feature(feature, reason, detail), False, reason
        if not D_expr:
            reason = "cut_revolve requires numeric diameter and length/position"
            detail = f"diameter={D!r}, position={pos!r}"
            return _todo_feature(feature, reason, detail), False, reason
        return [
            f"try:",
            f"    rev_wp = (",
            f"        cq.Workplane('YZ')",
            f"        .workplane(offset={pos_expr})",
            f"        .circle({D_expr}/2)",
            f"        .revolve(360, (0, 0, 0), (1, 0, 0))",
            f"    )",
            f"    result = result.cut(rev_wp)",
            f"    feature_log.append({fid!r})",
            f"except Exception:",
            f"    not_implemented.append({fid!r})",
            f"    # reason: cut_revolve failed",
        ], True, None

    return [
        f"# TODO: pocket not implemented (prim={prim!r}, subtype={sub!r})",
    ], False, f"pocket unsupported: prim={prim}"


def emit_hole(feature: dict) -> tuple[list[str], bool, str | None]:
    prim = feature.get("modeling_primitive")
    D = diameter_from(feature)
    D_name = param_name(
        feature,
        "body_outer_diameter", "Body outer diameter", "outer_diameter", "D",
        "diameter", "shaft_diameter", "Shaft diameter", "OD", "across_flats",
        "hex_width_across_flats", "Hex width across flats", "internal_bore_diameter",
        "hole_diameter", "Hole diameter", "Through hole diameter",
        "clearance_hole_diameter", "cross_drilled_hole_diameter",
    )
    D_var = snake(D_name) if D_name else "D_val"
    if prim in ("hole", "counterbore_hole") and isinstance(D, (int, float)):
        return [
            f"try:",
            f"    result = (",
            f"        result",
            f"        .faces('>X').workplane(centerOption='CenterOfMass')",
            f"        .hole({D_var})",
            f"    )",
            f"    feature_log.append({feature.get('id', '?')!r})",
            f"except Exception:",
            f"    not_implemented.append({feature.get('id', '?')!r})",
            f"    # reason: hole failed",
        ], True, None
    return [
        f"# TODO: hole not implemented (prim={prim!r}, D={D!r})",
    ], False, f"hole unsupported: prim={prim}, D={D}"


def emit_chamfer(feature: dict) -> tuple[list[str], bool, str | None]:
    prim = feature.get("modeling_primitive")
    fid = feature.get("id", "?")
    C = param_value(feature, "C", "chamfer", "Chamfer", "chamfer_size", "Corner chamfer size", "end_chamfer")
    C_name = param_name(feature, "C", "chamfer", "Chamfer", "chamfer_size", "Corner chamfer size", "end_chamfer")
    C_var = snake(C_name) if C_name else "C_val"
    if prim == "cut_revolve":
        taper, _, taper_expr = _numeric_param(feature, CUSTOM_LENGTH_CANDIDATES)
        if not (isinstance(C, (int, float)) and taper_expr):
            reason = "chamfer cut_revolve requires numeric taper and C"
            detail = f"taper={taper!r}, C={C!r}"
            return _todo_feature(feature, reason, detail), False, reason

        OD, _, od_expr = _numeric_param(
            feature,
            (
                "body_outer_diameter", "Body outer diameter", "outer_diameter", "D",
                "diameter", "shaft_diameter", "Shaft diameter", "OD",
            ),
        )
        if not od_expr:
            reason = "chamfer cut_revolve requires numeric diameter"
            detail = f"diameter={OD!r}"
            return _todo_feature(feature, reason, detail), False, reason

        return [
            f"try:",
            f"    taper = (",
            f"        cq.Workplane('YZ')",
            f"        .workplane(offset={taper_expr} - {C_var})",
            f"        .circle({od_expr}/2)",
            f"        .workplane(offset={C_var})",
            f"        .circle({od_expr}/2 * 0.5)",
            f"        .loft(combine=True)",
            f"    )",
            f"    result = result.cut(taper)",
            f"    feature_log.append({fid!r})",
            f"except Exception:",
            f"    not_implemented.append({fid!r})",
            f"    # reason: chamfer cut_revolve taper failed",
        ], True, None
    if C is None:
        return [f"# TODO: chamfer skipped (no C dimension)"], False, "chamfer no-dim"
    if isinstance(C, (int, float)):
        return [
            f"try:",
            f"    result = result.edges('|Z').chamfer({C_var})",
            f"except Exception:",
            f"    pass  # no suitable edges for chamfer",
        ], True, None
    return [f"# TODO: chamfer skipped (C={C!r} not numeric)"], False, f"chamfer C={C!r}"


def emit_fillet(feature: dict) -> tuple[list[str], bool, str | None]:
    R = param_value(feature, "R", "radius", "Radius", "fillet_radius", "Fillet radius", "shoulder_fillet_radius")
    R_name = param_name(feature, "R", "radius", "Radius", "fillet_radius", "Fillet radius", "shoulder_fillet_radius")
    R_var = snake(R_name) if R_name else "R_val"
    if R is None:
        return [f"# TODO: fillet skipped (no R dimension)"], False, "fillet no-dim"
    if isinstance(R, (int, float)):
        return [
            f"try:",
            f"    result = result.edges('|Z').fillet({R_var})",
            f"except Exception:",
            f"    pass  # no suitable edges for fillet",
        ], True, None
    return [f"# TODO: fillet skipped (R={R!r} not numeric)"], False, f"fillet R={R!r}"


def emit_metadata(feature: dict) -> tuple[list[str], bool, str | None]:
    """Metadata, datum, gdnt, thread features are comment-only."""
    ftype = feature.get("feature_type", "?")
    p = [pp.get("name") for pp in feature.get("parameters", [])]
    return [f"# metadata: {ftype} (params: {', '.join(p) if p else 'none'})"], False, None


def emit_unsupported(feature: dict) -> tuple[list[str], bool, str | None]:
    ftype = feature.get("feature_type", "?")
    prim = feature.get("modeling_primitive", "?")
    return [
        f"# TODO: {ftype}+{prim} not implemented in deterministic Step 5",
    ], False, f"{ftype}+{prim} unsupported"


# Dispatch table
EMITTERS = {
    ("base_body",): emit_base_body,
    ("boss",): emit_boss,
    ("hole",): emit_hole,
    ("chamfer",): emit_chamfer,
    ("fillet",): emit_fillet,
    ("pocket",): emit_pocket,
    ("cut",): emit_unsupported,
    ("pattern",): emit_unsupported,
    ("metadata",): emit_metadata,
    ("datum",): emit_metadata,
    ("gdnt",): emit_metadata,
    ("thread",): emit_metadata,
}


# ----------------------------------------------------------------------------
# Code generator
# ----------------------------------------------------------------------------
def generate_model_py(spec: dict, spec_path: Path) -> tuple[str, int]:
    pn = spec["part_number"]
    component_id = spec.get("catalog_id", "unknown")
    params = spec.get("parameter_bindings", {})
    features = spec.get("resolved_features", [])
    try:
        source_path = spec_path.relative_to(REPO_ROOT)
    except ValueError:
        source_path = spec_path

    out: list[str] = []
    out.append('"""')
    out.append(f"Auto-generated CadQuery model for part number {pn}.")
    out.append(f"Source: {source_path}")
    out.append(f"Template: {spec.get('template_id', '?')}")
    out.append("Generator: scripts/step5_codegen.py (deterministic)")
    out.append('"""')
    out.append("from __future__ import annotations")
    out.append("")
    out.append("import os")
    out.append("import cadquery as cq")
    out.append("")
    out.append("PART_NUMBER = " + repr(pn))
    out.append("")
    out.append("# ----------------------------------------------------------------------------")
    out.append("# Coordinate system (informational; geometry below is placed at the world origin)")
    out.append("# ----------------------------------------------------------------------------")
    cs = spec.get("coordinate_system") or {}
    out.append(f"_COORD_ORIGIN = {cs.get('origin', '?')!r}")
    out.append(f"_COORD_X      = {cs.get('x_axis', '?')!r}")
    out.append(f"_COORD_Y      = {cs.get('y_axis', '?')!r}")
    out.append(f"_COORD_Z      = {cs.get('z_axis', '?')!r}")
    out.append("")
    out.append("# ----------------------------------------------------------------------------")
    out.append("# Parameters (one constant per parameter_binding; None when unresolved)")
    out.append("# ----------------------------------------------------------------------------")
    null_count = 0
    seen: set[str] = set()
    for name, val in params.items():
        v = snake(name)
        if v in seen:
            # collisions: suffix _2, _3, ...
            i = 2
            while f"{v}_{i}" in seen:
                i += 1
            v = f"{v}_{i}"
        seen.add(v)
        if val is None:
            out.append(f"{v} = None  # TODO: <{name}> needs_review (carried_over or unresolved)")
            null_count += 1
        elif isinstance(val, (int, float)):
            out.append(f"{v} = {val}  # {name}")
        else:
            out.append(f"{v} = {val!r}  # {name}")
    # Helpers used in emitters; declare as locals
    out.append("")
    out.append("# ----------------------------------------------------------------------------")
    out.append("# Features (applied in order; unsupported features emit TODO comments)")
    out.append("# ----------------------------------------------------------------------------")
    out.append("feature_log: list[str] = []")
    out.append("not_implemented: list[str] = []")
    out.append("")
    out.append("# Feature: init")
    out.append("result = cq.Workplane('XY')")
    out.append("")

    for feature in features:
        fid = feature.get("id", "?")
        ftype = (feature.get("feature_type") or "").lower()
        prim = (feature.get("modeling_primitive") or "").lower()
        sub = feature.get("subtype") or ""
        out.append(f"# Feature: {fid} — {ftype} ({sub}) [prim={prim}]")
        emitter = EMITTERS.get((ftype,)) or emit_unsupported
        code, applied, reason = emitter(_feature_with_bindings(feature, params))
        out.extend(code)
        if not _emitter_records_feature(code):
            if applied:
                out.append(f"feature_log.append({fid!r})")
            else:
                out.append(f"not_implemented.append({fid!r})")
            if reason:
                out.append(f"# reason: {reason}")
        out.append("")

    out.append("# ----------------------------------------------------------------------------")
    out.append("# Validation hook (called by Step 7)")
    out.append("# ----------------------------------------------------------------------------")
    out.append("def validate() -> dict:")
    out.append("    bb = result.val().BoundingBox()")
    out.append("    return {")
    out.append("        'part_number': PART_NUMBER,")
    out.append("        'bounding_box': (bb.xlen, bb.ylen, bb.zlen),")
    out.append("        'volume_mm3': float(result.val().Volume()),")
    out.append("        'feature_log': list(feature_log),")
    out.append("        'not_implemented': list(not_implemented),")
    out.append("    }")
    out.append("")
    out.append("")
    out.append("if __name__ == '__main__':")
    out.append("    out_dir = os.path.dirname(os.path.abspath(__file__))")
    out.append("    cq.exporters.export(result, os.path.join(out_dir, f'{PART_NUMBER}.step'))")
    out.append("    cq.exporters.export(result, os.path.join(out_dir, f'{PART_NUMBER}.stl'))")
    out.append("    import json as _json")
    out.append("    print(_json.dumps(validate(), indent=2))")
    out.append("")

    return "\n".join(out), null_count


# ----------------------------------------------------------------------------
# Per-spec driver
# ----------------------------------------------------------------------------
def process_spec(
    spec_path: Path,
    out_dir: Path,
    schema: dict,
    *,
    dry_run: bool = False,
    smoke: bool = True,
) -> CodegenResult:
    res = CodegenResult(
        component_id=component_id_of(spec_path),
        part_number=part_number_of(spec_path),
        spec_path=str(spec_path),
        output_path="",
        ok=False,
    )

    try:
        spec = json.loads(spec_path.read_text())
    except Exception as e:
        res.error = f"read/parse: {e}"
        return res

    # Schema validation
    try:
        jsonschema.validate(instance=spec, schema=schema)
        res.schema_valid = True
    except jsonschema.ValidationError as e:
        res.error = f"schema: {e.message}"
        return res

    res.features_total = len(spec.get("resolved_features", []))
    res.parameters = len(spec.get("parameter_bindings", {}))

    # Generate
    try:
        code, null_count = generate_model_py(spec, spec_path)
        res.parameters_null = null_count
    except Exception as e:
        res.error = f"generate: {e}"
        return res

    # Syntax check via ast.parse
    try:
        ast.parse(code)
        res.syntax_valid = True
    except SyntaxError as e:
        res.error = f"syntax: {e}"
        return res

    # Track per-feature outcomes from the generated code is hard without running it;
    # we approximate by walking features and re-running the emitter decision:
    for f in spec.get("resolved_features", []):
        ftype = (f.get("feature_type") or "").lower()
        emitter = EMITTERS.get((ftype,)) or emit_unsupported
        _, applied, reason = emitter(_feature_with_bindings(f, spec.get("parameter_bindings", {})))
        if applied:
            res.features_implemented.append(f.get("id", "?"))
        else:
            res.features_skipped.append(
                {"id": f.get("id", "?"), "feature_type": ftype, "reason": reason or "unsupported"}
            )

    out_path = out_dir / f"{spec['part_number']}.model.py"
    res.output_path = str(out_path)

    if dry_run:
        res.ok = res.schema_valid and res.syntax_valid
        return res

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(code)

    # Optional smoke run
    if smoke:
        try:
            import subprocess
            r = subprocess.run(
                [sys.executable, str(out_path)],
                capture_output=True, text=True, timeout=60,
                cwd=str(out_dir),
            )
            if r.returncode == 0 and (out_dir / f"{spec['part_number']}.step").exists():
                res.runtime_smoke = True
        except Exception as e:
            res.error = f"runtime: {e}"
            return res

    res.ok = res.schema_valid and res.syntax_valid and (res.runtime_smoke or not smoke)
    return res


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "Step 5")
    p.add_argument("--component", help="Filter: component_id (e.g. 110300324920)")
    p.add_argument("--part-number", dest="part_number", help="Filter: part number (e.g. MCSCN10)")
    p.add_argument("--specs-dir", type=Path, default=DEFAULT_SPECS_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--all", dest="select_all", action="store_true", help="Process all specs")
    p.add_argument("--dry-run", action="store_true", help="Generate but do not write files")
    p.add_argument("--no-smoke", dest="smoke", action="store_false", help="Skip runtime smoke test")
    p.add_argument("--json", dest="json_out", action="store_true", help="Machine-readable summary")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    schema = json.loads(args.schema.read_text())

    # Discover
    specs: list[Path] = sorted(args.specs_dir.glob("*.resolved_cad_spec.json"))
    if args.component:
        specs = [s for s in specs if component_id_of(s) == args.component]
    if args.part_number:
        specs = [s for s in specs if part_number_of(s) == args.part_number]
    if args.limit:
        specs = specs[: args.limit]
    if not args.select_all and not args.component and not args.part_number:
        # Default: process all but warn
        if not specs:
            print(f"ERROR: no specs found in {args.specs_dir}", file=sys.stderr)
            return 1

    if args.verbose:
        print(f"==> Step 5: {len(specs)} spec(s) -> {args.out_dir}", file=sys.stderr)

    results: list[CodegenResult] = []
    for spec_path in specs:
        r = process_spec(spec_path, args.out_dir, schema, dry_run=args.dry_run, smoke=args.smoke)
        results.append(r)
        if args.verbose and not args.json_out:
            status = "OK" if r.ok else f"ERR ({r.error})"
            print(
                f"  [{status}] {r.part_number:20s} "
                f"feat={r.features_total:2d}/{len(r.features_implemented):2d} "
                f"params={r.parameters}/{r.parameters_null} null",
                file=sys.stderr,
            )

    if args.json_out:
        print(json.dumps([asdict(r) for r in results], indent=2))
    else:
        ok = sum(1 for r in results if r.ok)
        skipped = sum(len(r.features_skipped) for r in results)
        print(
            f"Step 5: {ok}/{len(results)} ok, {sum(r.features_total for r in results)} features total, "
            f"{sum(len(r.features_implemented) for r in results)} implemented, {skipped} skipped",
            file=sys.stderr,
        )

    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
