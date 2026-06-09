"""Step 0.5 — between Step 0 and Step 1: cluster components into part families.

Reads:   output/feature_templates/<cid>.feature_template.json (one per component)
Writes:  output/families/<family_id>/family.json
         output/families/index.json          (flat list, sorted by member count)
         output/families/REPORT.md           (when --report-md)
         output/workflow_state.sqlite3       (adds family_id column on components)

Pipeline (per --templates-dir scan):
  1. For each feature_template.json, build a ComponentFingerprint.
  2. Upsert into a local Chroma store (.chroma/component_fingerprints).
  3. Run structural_candidates (same category_root + Jaccard(value_keys) >= 0.5).
  4. Run cluster_fingerprints (agglomerative on MiniLM cosine, silhouette pick,
     then template-signature force-merge).
  5. Write per-family files + index.
  6. Update state DB family_id column.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.similarity.category_resolver import load_yaml_index, resolve_category
from scripts.similarity.cluster import PartFamily, cluster_fingerprints
from scripts.similarity.filter import same_root_components, structural_candidates
from scripts.similarity.fingerprint import build_fingerprint
from scripts.similarity.store import FingerprintStore


DEFAULT_TEMPLATES_DIR = Path("output/feature_templates")
DEFAULT_OUT_DIR = Path("output/families")
DEFAULT_CHROMA_DIR = Path(".chroma/component_fingerprints")
DEFAULT_STATE_DB = Path("output/workflow_state.sqlite3")
DEFAULT_DOWNLOADS_DIR = Path("downloads")
DEFAULT_YAML_PATH = Path("data/hierachy_enriched_final.yaml")
DEFAULT_JACCARD = 0.5
DEFAULT_TEMPLATE_JACCARD = 0.8


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def ensure_family_id_column(state_db: Path) -> None:
    """Add the family_id column to the components table if it doesn't exist.

    Idempotent. Uses SQLite's PRAGMA to inspect columns first; this avoids
    the 'duplicate column' error that ALTER TABLE would raise otherwise.
    """
    if not state_db.exists():
        return
    con = sqlite3.connect(state_db)
    try:
        cols = {
            row[1] for row in con.execute("PRAGMA table_info(components)").fetchall()
        }
        if "family_id" not in cols:
            con.execute("ALTER TABLE components ADD COLUMN family_id TEXT")
            con.commit()
    finally:
        con.close()


def update_state_db(state_db: Path, families: list[PartFamily]) -> int:
    """Set family_id for every member of every family in the state DB.

    Returns the number of rows updated. Components without a family_id in the
    state DB (i.e. not yet seen by Step 0) are left alone — we don't insert
    new rows from this CLI; the state DB is owned by Step 0.
    """
    if not state_db.exists():
        return 0
    con = sqlite3.connect(state_db)
    try:
        n = 0
        for fam in families:
            for cid in fam.member_component_ids:
                cur = con.execute(
                    "UPDATE components SET family_id = ? WHERE component_id = ?",
                    (fam.family_id, cid),
                )
                n += cur.rowcount
        con.commit()
        return n
    finally:
        con.close()


def write_family_file(out_dir: Path, fam: PartFamily) -> Path:
    fam_dir = out_dir / fam.family_id
    fam_dir.mkdir(parents=True, exist_ok=True)
    path = fam_dir / "family.json"
    path.write_text(json.dumps(fam.to_dict(), indent=2, sort_keys=True))
    return path


def write_index(out_dir: Path, families: list[PartFamily]) -> Path:
    """Write the flat list index, sorted by member count descending.

    The index is the small file other tools should read to discover families.
    """
    index_path = out_dir / "index.json"
    payload = {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "family_count": len(families),
        "singleton_count": sum(1 for f in families if len(f.member_component_ids) == 1),
        "max_family_size": max((len(f.member_component_ids) for f in families), default=0),
        "families": [
            {
                "family_id": f.family_id,
                "category_root": f.category_root,
                "members": f.member_component_ids,
                "n_members": len(f.member_component_ids),
                "shared_value_keys": f.shared_value_keys,
                "shared_attribute_keys": f.shared_attribute_keys,
                "canonical_template_id": f.canonical_template_id,
                "avg_template_signature_jaccard": f.avg_template_signature_jaccard,
            }
            for f in sorted(families, key=lambda f: -len(f.member_component_ids))
        ],
    }
    index_path.write_text(json.dumps(payload, indent=2))
    return index_path


def write_report_md(out_dir: Path, families: list[PartFamily], root_groups: dict[str, list[str]]) -> Path:
    """Write a human-readable Markdown summary.

    Includes:
      - Header with totals
      - One section per family (size > 1 first, then singletons)
      - Per-family: member list, shared_value_keys, canonical template
    """
    lines: list[str] = []
    lines.append("# Component Similarity Report")
    lines.append("")
    lines.append(f"_Generated at {utc_now()}_")
    lines.append("")

    n_families = len(families)
    n_singletons = sum(1 for f in families if len(f.member_component_ids) == 1)
    n_multi = n_families - n_singletons
    max_size = max((len(f.member_component_ids) for f in families), default=0)
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Total components clustered:** {sum(len(f.member_component_ids) for f in families)}")
    lines.append(f"- **Total families:** {n_families}")
    lines.append(f"- **Multi-member families:** {n_multi}")
    lines.append(f"- **Singletons:** {n_singletons}")
    lines.append(f"- **Largest family size:** {max_size}")
    lines.append("")

    if root_groups:
        lines.append("## Components by category_root")
        lines.append("")
        lines.append("| category_root | n_components |")
        lines.append("|---|---|")
        for root in sorted(root_groups.keys()):
            lines.append(f"| `{root}` | {len(root_groups[root])} |")
        lines.append("")

    sorted_fams = sorted(families, key=lambda f: (-len(f.member_component_ids), f.family_id))
    lines.append("## Families")
    lines.append("")
    for f in sorted_fams:
        size = len(f.member_component_ids)
        is_singleton = size == 1
        marker = "🟦 singleton" if is_singleton else f"⭐ {size} members"
        lines.append(f"### `{f.family_id}` — {marker}")
        lines.append("")
        lines.append(f"- **category_root:** `{f.category_root}`")
        lines.append(f"- **members:** {', '.join(f'`{m}`' for m in f.member_component_ids)}")
        if f.shared_value_keys:
            lines.append(f"- **shared_value_keys:** {', '.join(f'`{k}`' for k in f.shared_value_keys)}")
        if f.shared_attribute_keys:
            lines.append(f"- **shared_attribute_keys:** {', '.join(f'`{k}`' for k in f.shared_attribute_keys)}")
        if f.canonical_template_id:
            lines.append(f"- **canonical_template_id:** `{f.canonical_template_id}`")
        if size > 1:
            lines.append(f"- **avg_template_signature_jaccard:** {f.avg_template_signature_jaccard:.3f}")
        lines.append("")

    path = out_dir / "REPORT.md"
    path.write_text("\n".join(lines))
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="step0_5_component_similarity",
        description="Cluster catalog components into part families using MiniLM embeddings + structural Jaccard.",
    )
    p.add_argument("--templates-dir", type=Path, default=DEFAULT_TEMPLATES_DIR,
                   help="Directory of *.feature_template.json files.")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                   help="Output directory for family files + index + report.")
    p.add_argument("--chroma-dir", type=Path, default=DEFAULT_CHROMA_DIR,
                   help="Local Chroma persist directory.")
    p.add_argument("--state-db", type=Path, default=DEFAULT_STATE_DB,
                   help="workflow_state.sqlite3 path (will ALTER TABLE to add family_id).")
    p.add_argument("--downloads-dir", type=Path, default=DEFAULT_DOWNLOADS_DIR,
                   help="Downloads directory for category resolution via specs.json breadcrumbs.")
    p.add_argument("--yaml-path", type=Path, default=DEFAULT_YAML_PATH,
                   help="MISUMI category hierarchy YAML. If missing, falls back to part_family text inference.")
    p.add_argument("--jaccard", type=float, default=DEFAULT_JACCARD,
                   help="Structural pre-filter Jaccard threshold on value_keys (default 0.5).")
    p.add_argument("--template-jaccard", type=float, default=DEFAULT_TEMPLATE_JACCARD,
                   help="Force-merge trigger on template_signature Jaccard (default 0.8).")
    p.add_argument("--report-md", action="store_true",
                   help="Also write a human-readable REPORT.md.")
    p.add_argument("--reset", action="store_true",
                   help="Reset the Chroma collection before ingesting (for re-runs).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    templates_dir: Path = args.templates_dir
    out_dir: Path = args.out_dir
    chroma_dir: Path = args.chroma_dir
    state_db: Path = args.state_db
    downloads_dir: Path = args.downloads_dir
    yaml_path: Path = args.yaml_path

    out_dir.mkdir(parents=True, exist_ok=True)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    # 1. Collect input files.
    template_files = sorted(templates_dir.glob("*.feature_template.json"))
    if not template_files:
        print(f"[step0.5] No feature templates found in {templates_dir}; nothing to do.")
        return 0

    print(f"[step0.5] Found {len(template_files)} feature template(s) in {templates_dir}")

    # 1b. Load YAML category index (if available) and pre-resolve all categories.
    yaml_index = None
    if yaml_path and yaml_path.exists():
        try:
            yaml_index = load_yaml_index(yaml_path=yaml_path)
            print(f"[step0.5] Loaded category hierarchy: {len(yaml_index)} nodes from {yaml_path}")
        except Exception as e:
            print(f"[step0.5] WARN: could not load YAML at {yaml_path}: {e}")
            yaml_index = None
    else:
        print(f"[step0.5] No category YAML at {yaml_path}; falling back to part_family text inference.")

    category_resolutions: dict[str, object] = {}
    if yaml_index is not None:
        for tf in template_files:
            cid = tf.stem.split(".")[0]
            res = resolve_category(yaml_index, cid, downloads_root=downloads_dir)
            category_resolutions[cid] = res
        matched = sum(1 for r in category_resolutions.values() if r.matched)
        print(f"[step0.5] Resolved {matched}/{len(category_resolutions)} component categories from YAML")

    # 2. Build fingerprints.
    fps = []
    for tf in template_files:
        cid = tf.stem.split(".")[0]  # '110300324920.feature_template' -> '110300324920'
        cat_res = category_resolutions.get(cid)
        try:
            fp = build_fingerprint(
                tf,
                component_id=cid,
                category_resolution=cat_res,
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"[step0.5] WARN: skipping {tf.name}: {e}")
            continue
        fps.append(fp)
    print(f"[step0.5] Built {len(fps)} fingerprint(s)")

    if not fps:
        print("[step0.5] No usable fingerprints; exiting.")
        return 0

    # 3. Persist to Chroma.
    store = FingerprintStore(persist_dir=chroma_dir, collection_name="component_fingerprints")
    if args.reset:
        store.reset()
    store.upsert_many(fps)
    print(f"[step0.5] Chroma store now holds {store.count()} vector(s)")

    # 4. Structural pre-filter.
    candidates = structural_candidates(fps, jaccard_threshold=args.jaccard)
    root_groups = same_root_components(fps)
    print(f"[step0.5] Structural candidates (Jaccard>={args.jaccard}): {len(candidates)} pair(s)")

    # 5. Cluster.
    families = cluster_fingerprints(
        fps,
        candidate_pairs=candidates,
        template_jaccard_threshold=args.template_jaccard,
    )
    for f in families:
        f.created_at = utc_now()
    n_multi = sum(1 for f in families if len(f.member_component_ids) > 1)
    n_singletons = len(families) - n_multi
    print(f"[step0.5] Produced {len(families)} families ({n_multi} multi-member, {n_singletons} singletons)")

    # 6. Write per-family files.
    for fam in families:
        write_family_file(out_dir, fam)
    write_index(out_dir, families)
    if args.report_md:
        write_report_md(out_dir, families, root_groups)

    # 7. Update state DB.
    ensure_family_id_column(state_db)
    n_updated = update_state_db(state_db, families)
    print(f"[step0.5] Updated family_id for {n_updated} component row(s) in {state_db}")

    # 8. Summary.
    print("")
    print("[step0.5] Top families by member count:")
    for f in sorted(families, key=lambda f: -len(f.member_component_ids))[:10]:
        size = len(f.member_component_ids)
        canon = f" canon={f.canonical_template_id}" if f.canonical_template_id else ""
        print(f"  {f.family_id}  cat={f.category_root:14s}  n={size}  shared={f.shared_value_keys}{canon}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
