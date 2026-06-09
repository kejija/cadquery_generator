"""MISUMI category resolver.

The `data/hierachy_enriched_final.yaml` file (copied from m1) maps MISUMI
breadcrumb M-codes to canonical category names + descriptions + variant
parameter schemas. We use this as the **ground truth** for category_code
when the feature_template.json doesn't carry one.

Usage
-----
    from scripts.similarity.category_resolver import resolve_category, load_yaml_index

    index = load_yaml_index()  # ~50ms, called once per process
    result = resolve_category(
        index=index,
        component_id="110310763649",
        downloads_root=Path("downloads"),
    )
    # result.category_code   = "M0101000000"
    # result.category_name   = "Linear Shafts"
    # result.category_root   = "shaft"     (via CATEGORY_SYNONYMS)
    # result.path            = ["Linear Motion", "Linear Shafts"]
    # result.matched         = True

Strategy
--------
1. Read the component's `downloads/<cid>/json/specs.json` breadcrumbs.
2. Take the LAST M-code in the last non-home breadcrumb URL (that's the leaf).
3. Look it up in the YAML index. If found -> use it directly.
4. If the leaf code is 13 digits and not in the YAML, truncate the last 3
   digits to zero (-> 10 digits) and look that up. This handles the case
   where the URL is more specific than the YAML taxonomy.
5. If still not found, return ``matched=False`` so callers can fall back to
   the PART_FAMILY_PREFIX_CATEGORIES inference.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from scripts.similarity.fingerprint import _normalize_category


# --- Data model ---------------------------------------------------------------


@dataclass
class CategoryResolution:
    component_id: str
    category_code: Optional[str]       # e.g. "M0101000000"
    category_name: Optional[str]      # e.g. "Linear Shafts"
    category_root: str                # e.g. "shaft" (via _normalize_category)
    path: list[str] = field(default_factory=list)  # e.g. ["Linear Motion", "Linear Shafts"]
    description: str = ""             # YAML description, first 200 chars
    matched: bool = False             # True if found in the YAML

    def to_dict(self) -> dict:
        return {
            "component_id": self.component_id,
            "category_code": self.category_code,
            "category_name": self.category_name,
            "category_root": self.category_root,
            "path": list(self.path),
            "description": self.description,
            "matched": self.matched,
        }


# --- YAML loader --------------------------------------------------------------


def _walk(node: dict, parent_id: Optional[str], out: dict) -> None:
    nid = node.get("id")
    if nid:
        out[nid] = {
            "name": node.get("name", ""),
            "parent_id": parent_id,
            "description": (node.get("description") or "")[:200],
        }
    for ch in node.get("children") or []:
        _walk(ch, parent_id=nid, out=out)


def load_yaml_index(yaml_path: Optional[Path] = None) -> dict:
    """Load the YAML hierarchy into a flat {id: {name, parent_id, description}} map.

    If ``yaml_path`` is None, defaults to ``data/hierachy_enriched_final.yaml``
    at the repo root. Raises FileNotFoundError if the file is missing.
    """
    if yaml_path is None:
        yaml_path = Path(__file__).resolve().parents[2] / "data" / "hierachy_enriched_final.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"category_resolver: YAML not found at {yaml_path}")
    data = yaml.safe_load(yaml_path.read_text())
    index: dict = {}
    for c in data.get("categories", []):
        _walk(c, parent_id=None, out=index)
    return index


# --- Breadcrumb extractor ----------------------------------------------------


_M_CODE_RE = re.compile(r"M\d+")


def _leaf_code_from_specs(specs: dict) -> Optional[str]:
    """Extract the deepest (leaf) M-code from a component's specs.json breadcrumbs.

    Strategy: take the last breadcrumb URL, then look for the *first* M-code
    whose digits >= 10. This handles both forms:
      - 10-digit code already: "M0100000000/M0101000000/" -> "M0101000000"
      - 13-digit code: "M0100000000/M0101000000/M0101000000123/"
        -> "M0101000000" (the 10-digit code is the leaf category; the
        13-digit is a more specific sub-page the YAML doesn't carry).
    """
    for b in reversed(specs.get("breadcrumbs", []) or []):
        url = b.get("url", "")
        if not url:
            continue
        codes = _M_CODE_RE.findall(url)
        if not codes:
            continue
        # Prefer a 10-digit code if present (these are the YAML leaf categories).
        for c in codes:
            if len(c) == 11 and c[1:].isdigit() and c.endswith("0000000"):
                # M + 10 digits where the last 7 are zeros — that's a top-level dept.
                continue
            if len(c) == 11:  # M + 10 digits
                return c
        # Otherwise, take the last code (will be 13 digits; resolver truncates).
        return codes[-1]
    return None


# --- Resolver -----------------------------------------------------------------


def resolve_category(
    index: dict,
    component_id: str,
    downloads_root: Optional[Path] = None,
    specs: Optional[dict] = None,
) -> CategoryResolution:
    """Resolve a component's MISUMI category via YAML hierarchy.

    Parameters
    ----------
    index:
        The flat YAML hierarchy map from ``load_yaml_index()``.
    component_id:
        The catalog component id (e.g. "110310763649"). Used to find the
        specs.json file at ``downloads/<cid>/json/specs.json`` if
        ``specs`` is not passed directly.
    downloads_root:
        Path to the ``downloads/`` directory. Defaults to ``<repo>/downloads``.
    specs:
        Pre-loaded specs.json dict. If provided, skips the file read.
        Useful for batch resolution or testing.
    """
    if specs is None:
        if downloads_root is None:
            downloads_root = Path(__file__).resolve().parents[2] / "downloads"
        specs_path = Path(downloads_root) / component_id / "json" / "specs.json"
        if specs_path.exists():
            try:
                specs = json.loads(specs_path.read_text())
            except (json.JSONDecodeError, OSError):
                specs = None

    code = _leaf_code_from_specs(specs) if specs else None

    if not code:
        return CategoryResolution(
            component_id=component_id,
            category_code=None,
            category_name=None,
            category_root="unknown",
            matched=False,
        )

    # Direct match.
    resolved_code: Optional[str] = None
    node = index.get(code)
    if node is None and len(code) == 13:
        # Truncate 13->10 by zeroing the last 3 digits.
        truncated = code[:10] + "000"
        node = index.get(truncated)
        if node is not None:
            resolved_code = truncated
    elif node is not None:
        resolved_code = code

    if node is None:
        return CategoryResolution(
            component_id=component_id,
            category_code=code,
            category_name=None,
            category_root="unknown",
            matched=False,
        )

    # Walk up the YAML hierarchy to build the full breadcrumb path.
    path: list[str] = []
    cur = resolved_code
    while cur:
        n = index.get(cur)
        if not n:
            break
        path.append(n["name"])
        cur = n["parent_id"]
    path.reverse()

    return CategoryResolution(
        component_id=component_id,
        category_code=resolved_code,
        category_name=node["name"],
        category_root=_normalize_category("", part_family=node["name"]),
        path=path,
        description=node["description"],
        matched=True,
    )


def resolve_many(
    index: dict,
    component_ids: list[str],
    downloads_root: Optional[Path] = None,
) -> dict[str, CategoryResolution]:
    """Batch resolve. Returns ``{component_id: CategoryResolution}``."""
    out: dict[str, CategoryResolution] = {}
    for cid in component_ids:
        out[cid] = resolve_category(index, cid, downloads_root=downloads_root)
    return out
