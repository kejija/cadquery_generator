"""One-off diagnostic: resolve every component's breadcrumb M-code through the
YAML hierarchy index, with a 13->10 digit truncation fallback.

Run: .venv-sim/bin/python scripts/diag_yaml_resolver.py
"""
import json
import sys
from pathlib import Path

REPO = Path("/home/keji/fe/cadquery_generator")
sys.path.insert(0, str(REPO))

from scripts.similarity.category_resolver import (  # noqa: E402
    load_yaml_index,
    resolve_category,
)


def main():
    index = load_yaml_index()
    print(f"yaml index: {len(index)} nodes", file=sys.stderr)

    downloads = REPO / "downloads"
    cids = sorted(p.name for p in downloads.iterdir() if p.is_dir() and p.name.isdigit())
    matched = unmatched = 0
    for cid in cids:
        specs_path = downloads / cid / "json" / "specs.json"
        if not specs_path.exists():
            continue
        d = json.loads(specs_path.read_text())
        res = resolve_category(index, cid, downloads_root=downloads)
        if res.matched:
            matched += 1
            marker = "  "
        else:
            unmatched += 1
            marker = "!!"
        print(f"{marker} {cid:14s}  {res.category_code or '-':13s}  {res.category_name or '(unmatched)':40s}  root={res.category_root}")
    print(f"\nmatched={matched}  unmatched={unmatched}  total={matched+unmatched}", file=sys.stderr)


if __name__ == "__main__":
    main()
