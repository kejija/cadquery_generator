"""Component fingerprint for the similarity pipeline.

A ComponentFingerprint is a compact, embeddable representation of a catalog
component derived from its feature_template.json. The fingerprint is what gets
written to the vector DB and what clustering operates on.

Design notes
------------
- ``category_code`` is NOT stored in feature_template.json directly — it lives
  in the upstream datasheet JSON. We default to ``""`` so a missing field maps
  to ``category_root == "unknown"``, which the structural pre-filter excludes.
  This is intentional: unknown category => never cluster => always singleton,
  which is the safe default.
- ``value_keys`` is the set of parameter names that represent geometry. We use
  ``category == "geometry"`` first, fall back to ``parameter_type == "length"``
  for templates that don't tag category, and finally use the full parameter set
  so we never produce an empty fingerprint.
- ``template_signature`` is the bridge into the template-aware force-merge: two
  components with the same signature are the same geometry recipe with
  different values, and should be force-merged regardless of embedding distance.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


CATEGORY_SYNONYMS: dict[str, str] = {
    # Bearings
    "ball_bearing": "bearing",
    "angular_contact_bearing": "bearing",
    "cylindrical_roller_bearing": "bearing",
    "tapered_roller_bearing": "bearing",
    "needle_roller_bearing": "bearing",
    "thrust_bearing": "bearing",
    # Shafts
    "linear_shaft": "shaft",
    "support_shaft": "shaft",
    "precision_shaft": "shaft",
    "motor_shaft": "shaft",
}


# Fallback: when category_code is missing but part_family is set, infer the
# category from the *first* word/phrase of part_family. This is a string match
# against the leading 1-2 tokens. Examples:
#   "Linear Shaft - One End Stepped"          -> "shaft"
#   "Air Couplers - Socket, Nut Tightening"   -> "coupler"
#   "Hinge Bases - U-Shaped, Inch"            -> "hinge"
#   "Plate Brackets - 8 Series"               -> "bracket"
#   "Brackets - 8-45 Series, Reversal ..."    -> "bracket"  (note: starts with "Brackets", singular handled below)
#   "40x120 Aluminum Extrusion - 8 Series"    -> "extrusion"
PART_FAMILY_PREFIX_CATEGORIES: tuple[tuple[str, str], ...] = (
    # (lowercase prefix, normalized root)
    ("linear shaft", "shaft"),
    ("support shaft", "shaft"),
    ("precision shaft", "shaft"),
    ("motor shaft", "shaft"),
    ("drive shaft", "shaft"),
    ("ball bearing", "bearing"),
    ("angular contact bearing", "bearing"),
    ("roller bearing", "bearing"),
    ("needle bearing", "bearing"),
    ("thrust bearing", "bearing"),
    ("air coupler", "coupler"),
    ("quick connect coupler", "coupler"),
    ("coupler", "coupler"),
    ("coupling", "coupling"),
    ("hinge", "hinge"),
    # 'plate' prefixes must come before 'bracket' so "Plate Brackets" -> "plate"
    # and "Brackets" -> "bracket" still work.
    ("plate bracket", "plate"),
    ("plate", "plate"),
    ("bracket", "bracket"),
    ("aluminum extrusion", "extrusion"),
    ("aluminium extrusion", "extrusion"),
    # Catalog part numbers often lead with "40x120" or "20mm" — strip those
    # by also matching a token that contains the word.
    ("extrusion", "extrusion"),
    ("guide rail", "guide_rail"),
    ("linear guide", "guide_rail"),
    ("bushing", "bushing"),
    ("sleeve", "bushing"),
    ("pulley", "pulley"),
    ("sprocket", "sprocket"),
    ("gear", "gear"),
    ("nut", "nut"),
    ("bolt", "bolt"),
    ("screw", "screw"),
    ("washer", "washer"),
    ("pin", "pin"),
    ("key", "key"),
    ("retainer", "retainer"),
    ("spring", "spring"),
    ("o-ring", "seal"),
    ("seal", "seal"),
    ("gasket", "seal"),
    ("housing", "housing"),
    ("flange", "flange"),
    ("valve", "valve"),
    ("fitting", "fitting"),
    ("adapter", "fitting"),
    ("clamp", "clamp"),
)


def _category_from_part_family(part_family: str) -> Optional[str]:
    """Infer category_root from the leading words of a human-readable part_family.

    Returns None if no prefix matches. This is a *fallback* used only when
    ``category_code`` is empty — we try the catalog text instead.

    The match is done in two passes:
      1. Whole-string prefix match against PART_FAMILY_PREFIX_CATEGORIES
      2. If that fails, scan the lowercased string for any of the prefix
         tokens as a substring (e.g. "40x120 Aluminum Extrusion" matches
         "aluminum extrusion" mid-string).
    """
    if not part_family:
        return None
    pf = part_family.strip().lower()
    if not pf:
        return None
    for prefix, root in PART_FAMILY_PREFIX_CATEGORIES:
        if pf.startswith(prefix):
            return root
    for prefix, root in PART_FAMILY_PREFIX_CATEGORIES:
        if prefix in pf:
            return root
    return None


def _normalize_category(code: Optional[str], part_family: Optional[str] = None) -> str:
    """Map a raw category_code (or part_family fallback) to a clustering root.

    Lookup order:
      1. CATEGORY_SYNONYMS (case-insensitive) — known aliases collapse.
      2. PART_FAMILY_PREFIX_CATEGORIES fallback on ``part_family`` text.
      3. First underscore-separated token — e.g. "widget_gizmo" -> "widget".
      4. "unknown" for empty / None / unrecognized-but-singleton cases.
    """
    if code:
        key = str(code).strip().lower()
        if key:
            if key in CATEGORY_SYNONYMS:
                return CATEGORY_SYNONYMS[key]
            return key.split("_", 1)[0]
    if part_family:
        inferred = _category_from_part_family(part_family)
        if inferred:
            return inferred
    return "unknown"


@dataclass
class ComponentFingerprint:
    component_id: str
    name: str
    description: str
    category_code: str
    category_root: str
    attribute_keys: list[str]
    value_keys: list[str]
    variant_count: int
    template_signature: Optional[str]
    text_for_embedding: str

    def to_dict(self) -> dict:
        return asdict(self)


# Hand-curated value-key synonyms. Each value is a CANONICAL key; every key in
# the list maps to that canonical. This is a *seed* map; the dynamic LLM-driven
# expansion lives in scripts/table_merge/synonym_llm.py (Task 9) and runs
# per-family. We add the most common MISUMI symbol conventions here so the
# first end-to-end run produces useful clusters.
#
# Examples of what this fixes:
#   D | main_diameter | shaft_diameter          -> all "D"
#   L | main_length | overall_length            -> all "L"
#   M | reduced_section_length | left_thread_length  -> all "M" (key segment)
#   ℓ1 | overall_length                            -> all "L"  (Unicode ell)
VALUE_KEY_SYNONYMS: dict[str, list[str]] = {
    # Outer / main diameter of a cylindrical body
    "D":  ["D", "main_diameter", "shaft_diameter", "body_outer_diameter", "outer_diameter"],
    # Inner / bore diameter
    "ID": ["ID", "ID_", "id", "internal_bore_diameter", "bore_diameter", "inner_diameter", "d_in"],
    # Overall length
    "L":  ["L", "ℓ1", "ℓ", "main_length", "overall_length", "overall_length_or_footprint", "body_length", "total_length"],
    # Width
    "W":  ["W", "overall_width", "width", "body_width"],
    # Height
    "H":  ["H", "overall_height", "height", "body_height"],
    # Thickness / wall
    "T":  ["T", "thickness", "part_thickness", "base_thickness", "wall_thickness", "wall/web thickness", "wall_web_thickness"],
    # Bore / through hole
    "d":  ["d", "diameter", "through_hole_diameter", "clearance_hole_diameter", "hole_diameter", "pivot_hole_diameter", "mounting_hole_diameter", "cross_drilled_hole_diameter"],
    # Reduced-section / secondary / "M" segment length
    "M":  ["M", "reduced_section_length", "left_thread_length", "right_tapped_length", "secondary_segment_length"],
    # Reduced-section / "N" / "F" diameter
    "N":  ["N", "reduced_section_diameter", "secondary_diameter", "tapped_diameter"],
    # Fillet / corner radius
    "R":  ["R", "fillet_radius", "shoulder_fillet_radius", "outer_corner_radius", "corner_radius", "shoulder_radius"],
    # Taper
    "TR": ["TR", "taper_length", "tapered_length", "end_taper_length", "taper_angle" if False else "taper_length"],
}


def _normalize_value_keys(keys: list[str]) -> list[str]:
    """Map a raw value-key list through VALUE_KEY_SYNONYMS, returning the
    canonical set (sorted, deduplicated). Keys with no synonym entry are
    passed through unchanged (after whitespace stripping; case is preserved
    so the original symbols like "OD" vs "id" remain distinguishable in
    debug output, but matching is case-insensitive).
    """
    out: set[str] = set()
    for raw in keys:
        k = str(raw).strip()
        if not k:
            continue
        kl = k.lower()
        mapped: Optional[str] = None
        for canonical, aliases in VALUE_KEY_SYNONYMS.items():
            if kl in {a.lower() for a in aliases}:
                mapped = canonical
                break
        out.add(mapped if mapped else k)
    return sorted(out)


def _classify_parameter(param: dict) -> bool:
    """Return True if a parameter is a 'value' / geometry key we should include.

    Preference order: explicit category == "geometry" > parameter_type == "length".
    """
    if not isinstance(param, dict):
        return False
    if param.get("category") == "geometry":
        return True
    if param.get("parameter_type") == "length":
        return True
    return False


def build_fingerprint(feature_template_path: Path, component_id: str) -> ComponentFingerprint:
    """Read a feature_template.json and produce a ComponentFingerprint.

    Raises FileNotFoundError if the path doesn't exist. Raises ValueError
    with a component_id-tagged message if the JSON is malformed.
    """
    import json

    path = Path(feature_template_path)
    if not path.exists():
        raise FileNotFoundError(
            f"build_fingerprint: feature_template not found for component {component_id}: {path}"
        )
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(
            f"build_fingerprint: malformed JSON for component {component_id} at {path}: {e}"
        ) from e

    raw_params = data.get("parameters") or []
    param_names: list[str] = []
    geom_param_names: list[str] = []
    for p in raw_params:
        name = p.get("name") if isinstance(p, dict) else None
        if not name:
            continue
        param_names.append(name)
        if _classify_parameter(p):
            geom_param_names.append(name)

    modeling_mode = data.get("modeling_mode", "unknown")
    sig: Optional[str] = None
    if param_names:
        sig = "|".join([modeling_mode] + sorted(set(param_names)))

    if geom_param_names:
        value_keys = sorted(set(geom_param_names))
    elif param_names:
        # Soft fallback: if no parameter is tagged as geometry, use the full set
        # so we never produce an empty fingerprint.
        value_keys = sorted(set(param_names))
    else:
        value_keys = []

    # Apply the seed value-key synonym map so MISUMI naming variants collapse
    # (D/main_diameter/shaft_diameter -> canonical "D", etc.). This is the
    # deterministic half of the table-merge synonym work; the dynamic
    # LLM-driven half lives in scripts/table_merge/synonym_llm.py (Task 9).
    value_keys = _normalize_value_keys(value_keys)

    metadata = data.get("metadata") or {}
    attribute_keys = sorted(metadata.get("attribute_keys") or [])
    variant_count = int(metadata.get("variant_count") or 0)
    category_code = str(metadata.get("category_code") or "")

    name = (
        data.get("part_family")
        or data.get("name")
        or component_id
    )
    description = data.get("description", "") or ""

    text = " | ".join(
        filter(
            None,
            [
                name,
                description,
                f"category:{category_code}",
                f"params:{' '.join(sorted(set(param_names))[:20])}",
                f"mode:{modeling_mode}",
            ],
        )
    )

    return ComponentFingerprint(
        component_id=component_id,
        name=name,
        description=description,
        category_code=category_code,
        category_root=_normalize_category(category_code, part_family=name),
        attribute_keys=attribute_keys,
        value_keys=value_keys,
        variant_count=variant_count,
        template_signature=sig,
        text_for_embedding=text,
    )
