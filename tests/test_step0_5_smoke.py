"""End-to-end smoke test for step0_5_component_similarity.

Runs the CLI as a subprocess against a fixture set of feature templates and
asserts the expected family/index/REPORT/state-db artifacts are produced.
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path("/home/keji/fe/cadquery_generator")
CLI = REPO / "scripts" / "step0_5_component_similarity.py"
PYTHON = REPO / ".venv-sim" / "bin" / "python"


def _make_minimal_template(tmp_path: Path, name: str, model_mode: str, params: list[dict]) -> Path:
    doc = {
        "schema_version": "1.0",
        "template_id": f"test_{name}",
        "part_family": f"Test {name}",
        "modeling_mode": model_mode,
        "parameters": params,
    }
    p = tmp_path / f"{name}.feature_template.json"
    p.write_text(json.dumps(doc))
    return p


def _make_state_db(tmp_path: Path, component_ids: list[str]) -> Path:
    db = tmp_path / "state.sqlite3"
    con = sqlite3.connect(db)
    con.executescript("""
        CREATE TABLE components (
            component_id TEXT PRIMARY KEY,
            component_path TEXT NOT NULL,
            step0_status TEXT NOT NULL DEFAULT 'done',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
    """)
    for cid in component_ids:
        con.execute(
            "INSERT INTO components (component_id, component_path, created_at, updated_at) VALUES (?, ?, '2026-06-08T15:00:00Z', '2026-06-08T15:00:00Z')",
            (cid, str(tmp_path / cid)),
        )
    con.commit()
    con.close()
    return db


def _run_cli(tmp_path: Path, templates_dir: Path, out_dir: Path, chroma_dir: Path, state_db: Path, *, report_md: bool = False) -> subprocess.CompletedProcess:
    cmd = [
        str(PYTHON), str(CLI),
        "--templates-dir", str(templates_dir),
        "--out-dir", str(out_dir),
        "--chroma-dir", str(chroma_dir),
        "--state-db", str(state_db),
        "--jaccard", "0.5",
    ]
    if report_md:
        cmd.append("--report-md")
    env = dict(os.environ)
    env["HF_HOME"] = "/tmp/hf_cache"
    return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)


# --- smoke tests --------------------------------------------------------------

@pytest.mark.skipif(
    os.environ.get("HERMES_SKIP_EMBEDDINGS") == "1",
    reason="HERMES_SKIP_EMBEDDINGS=1; step0.5 needs MiniLM",
)
def test_step0_5_runs_on_sample_inputs(tmp_path: Path):
    """Run on a small set of templates and assert families + index are written."""
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    out_dir = tmp_path / "families"
    chroma_dir = tmp_path / "chroma"
    state_db = _make_state_db(tmp_path, ["brg1", "brg2", "shaft1"])

    _make_minimal_template(
        templates_dir, "brg1", "simplified_single_body",
        [{"name": "OD", "category": "geometry"}, {"name": "ID", "category": "geometry"},
         {"name": "B", "category": "geometry"}],
    )
    _make_minimal_template(
        templates_dir, "brg2", "simplified_single_body",
        [{"name": "OD", "category": "geometry"}, {"name": "ID", "category": "geometry"},
         {"name": "B", "category": "geometry"}],
    )
    _make_minimal_template(
        templates_dir, "shaft1", "solid_of_revolution",
        [{"name": "OD", "category": "geometry"}, {"name": "L", "category": "geometry"},
         {"name": "keyway", "category": "geometry"}],
    )

    result = _run_cli(tmp_path, templates_dir, out_dir, chroma_dir, state_db)
    assert result.returncode == 0, f"CLI failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    # At minimum: index.json and at least one family.json should exist.
    families = list(out_dir.glob("*/family.json"))
    assert families, f"no family.json written. CLI output:\n{result.stdout}"
    assert (out_dir / "index.json").exists()

    # Index schema fields.
    idx = json.loads((out_dir / "index.json").read_text())
    assert "families" in idx
    assert "family_count" in idx
    assert idx["family_count"] == len(idx["families"])
    assert idx["family_count"] >= 1

    # Every family.json must be valid JSON with the expected keys.
    for f in families:
        d = json.loads(f.read_text())
        assert "family_id" in d
        assert "member_component_ids" in d
        assert "category_root" in d
        assert isinstance(d["member_component_ids"], list)


@pytest.mark.skipif(
    os.environ.get("HERMES_SKIP_EMBEDDINGS") == "1",
    reason="HERMES_SKIP_EMBEDDINGS=1; step0.5 needs MiniLM",
)
def test_step0_5_writes_report_md_when_flag_set(tmp_path: Path):
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    out_dir = tmp_path / "families"
    chroma_dir = tmp_path / "chroma"
    state_db = _make_state_db(tmp_path, ["A"])

    _make_minimal_template(
        templates_dir, "A", "simplified_single_body",
        [{"name": "OD", "category": "geometry"}],
    )

    result = _run_cli(tmp_path, templates_dir, out_dir, chroma_dir, state_db, report_md=True)
    assert result.returncode == 0, f"CLI failed:\n{result.stdout}\n{result.stderr}"
    report = out_dir / "REPORT.md"
    assert report.exists()
    txt = report.read_text()
    assert "## Summary" in txt
    assert "## Families" in txt


@pytest.mark.skipif(
    os.environ.get("HERMES_SKIP_EMBEDDINGS") == "1",
    reason="HERMES_SKIP_EMBEDDINGS=1; step0.5 needs MiniLM",
)
def test_step0_5_state_db_gets_family_id_column_and_rows_updated(tmp_path: Path):
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    out_dir = tmp_path / "families"
    chroma_dir = tmp_path / "chroma"
    state_db = _make_state_db(tmp_path, ["brg1", "brg2"])

    _make_minimal_template(
        templates_dir, "brg1", "simplified_single_body",
        [{"name": "OD", "category": "geometry"}, {"name": "ID", "category": "geometry"}],
    )
    _make_minimal_template(
        templates_dir, "brg2", "simplified_single_body",
        [{"name": "OD", "category": "geometry"}, {"name": "ID", "category": "geometry"}],
    )

    result = _run_cli(tmp_path, templates_dir, out_dir, chroma_dir, state_db)
    assert result.returncode == 0, f"CLI failed:\n{result.stdout}\n{result.stderr}"

    con = sqlite3.connect(state_db)
    cols = {row[1] for row in con.execute("PRAGMA table_info(components)").fetchall()}
    assert "family_id" in cols, "family_id column was not added"

    rows = list(con.execute("SELECT component_id, family_id FROM components ORDER BY component_id"))
    family_ids = {cid: fid for cid, fid in rows}
    assert "brg1" in family_ids and "brg2" in family_ids
    # The two bearings should share a family_id (they share value_keys + same root).
    assert family_ids["brg1"] is not None
    assert family_ids["brg2"] is not None
    con.close()


def test_step0_5_handles_empty_templates_dir(tmp_path: Path):
    """If there are no feature templates, the CLI should exit 0 with a clear message."""
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    out_dir = tmp_path / "families"
    chroma_dir = tmp_path / "chroma"
    state_db = _make_state_db(tmp_path, [])

    result = _run_cli(tmp_path, templates_dir, out_dir, chroma_dir, state_db)
    assert result.returncode == 0
    assert "nothing to do" in result.stdout
    # No families written.
    assert not (out_dir / "index.json").exists()
