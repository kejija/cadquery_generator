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


def _normalize_category(code: Optional[str]) -> str:
    """Map a raw category_code to a clustering-friendly root.

    Lookup order:
      1. CATEGORY_SYNONYMS (case-insensitive) — known aliases collapse.
      2. First underscore-separated token — e.g. "widget_gizmo" -> "widget".
      3. "unknown" for empty / None / unrecognized-but-singleton cases.
    """
    if not code:
        return "unknown"
    key = str(code).strip().lower()
    if not key:
        return "unknown"
    if key in CATEGORY_SYNONYMS:
        return CATEGORY_SYNONYMS[key]
    return key.split("_", 1)[0]


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
        category_root=_normalize_category(category_code),
        attribute_keys=attribute_keys,
        value_keys=value_keys,
        variant_count=variant_count,
        template_signature=sig,
        text_for_embedding=text,
    )
