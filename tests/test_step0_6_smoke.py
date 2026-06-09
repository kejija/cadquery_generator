"""End-to-end smoke tests for step0_6_table_merge.

Runs the CLI as a subprocess against fixture feature templates and family
metadata, asserts the expected merged_tables.json + merge_report.md
artifacts are produced, and verifies dry-run never touches the LLM.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path("/home/keji/fe/cadquery_generator")
CLI = REPO / "scripts" / "step0_6_table_merge.py"
PYTHON = REPO / ".venv-sim" / "bin" / "python"


# --- helpers -----------------------------------------------------------------


def _make_template(
    templates_dir: Path,
    component_id: str,
    *,
    n_params: int = 3,
    n_components: int = 2,
    n_features: int = 3,
    param_overrides: dict[int, dict] | None = None,
) -> Path:
    """Write a minimal but realistic feature_template.json fixture.

    param_overrides maps row-index -> field overrides, so tests can inject
    a conflicting value for one parameter without rewriting the whole file.
    """
    parameters = []
    for i in range(n_params):
        row = {
            "name": f"param_{i}",
            "symbol": ["D", "L", "M", "R", "T", "W", "H"][i % 7],
            "description": f"Test param {i}",
            "parameter_type": "diameter" if i == 0 else "length",
            "category": "geometry",
            "default_value": 10 + i,
            "units": "mm",
        }
        if param_overrides and i in param_overrides:
            row.update(param_overrides[i])
        parameters.append(row)

    components = [
        {
            "id": "comp_main_body",
            "name": "main_body",
            "component_type": "solid_body",
            "parent_component": None,
        }
    ]
    for i in range(n_components - 1):
        components.append({
            "id": f"comp_aux_{i}",
            "name": f"aux_{i}",
            "component_type": "metadata",
            "parent_component": None,
        })

    feature_graph = []
    for i in range(n_features):
        feature_graph.append({
            "id": f"feat_{i:03d}_base",
            "feature_type": "base_body",
            "operation": "base",
            "modeling_primitive": "extrude",
            "depends_on": [],
        })

    doc = {
        "schema_version": "1.0",
        "template_id": f"test_{component_id}",
        "part_family": f"Test family {component_id}",
        "modeling_mode": "simplified_single_body",
        "parameters": parameters,
        "components": components,
        "feature_graph": feature_graph,
        "catalog_metadata": {
            "material_options": ["SS304", "SUS440C"],
            "surface_treatment_options": ["none", "electroless_nickel"],
            "cleaning_options": ["standard"],
            "packaging_options": ["bulk"],
        },
    }
    p = templates_dir / f"{component_id}.feature_template.json"
    p.write_text(json.dumps(doc))
    return p


def _make_family(
    families_dir: Path,
    family_id: str,
    member_ids: list[str],
    category_root: str = "shaft",
) -> Path:
    family_dir = families_dir / family_id
    family_dir.mkdir(parents=True, exist_ok=True)
    family = {
        "family_id": family_id,
        "category_root": category_root,
        "member_component_ids": member_ids,
        "canonical_template_id": member_ids[0] if member_ids else None,
        "shared_value_keys": [],
        "shared_attribute_keys": [],
    }
    out = family_dir / "family.json"
    out.write_text(json.dumps(family))
    return out


def _write_index(families_dir: Path, families: list[dict]) -> Path:
    idx = {
        "schema_version": "1.0",
        "family_count": len(families),
        "families": families,
    }
    out = families_dir / "index.json"
    out.write_text(json.dumps(idx))
    return out


def _run_cli(
    families_dir: Path,
    templates_dir: Path,
    out_dir: Path,
    *,
    family_id: str | None = None,
    dry_run: bool = True,
    extra_pythonpath: Path | None = None,
) -> subprocess.CompletedProcess:
    cmd = [
        str(PYTHON), str(CLI),
        "--families-dir", str(families_dir),
        "--templates-dir", str(templates_dir),
        "--out-dir", str(out_dir),
    ]
    if dry_run:
        cmd.append("--dry-run")
    if family_id:
        cmd.extend(["--family-id", family_id])
    env = dict(os.environ)
    paths = [str(REPO)]
    if extra_pythonpath is not None:
        paths.insert(0, str(extra_pythonpath))
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)


# --- tests -------------------------------------------------------------------


def test_step0_6_no_families_dir(tmp_path: Path):
    """If --families-dir has no index.json, exit 0 and print a no-op message."""
    empty = tmp_path / "empty_families"
    empty.mkdir()
    out_dir = tmp_path / "out"
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()

    result = _run_cli(empty, templates_dir, out_dir)
    assert result.returncode == 0, (
        f"CLI failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    combined = (result.stdout + result.stderr).lower()
    assert "no" in combined and ("famil" in combined or "index" in combined), (
        f"expected a no-op / no-families message; got stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    # No output artifacts written.
    assert not (out_dir / "fam_X" / "merged_tables.json").exists()


def test_step0_6_dry_run_merges_shaft_family(tmp_path: Path):
    """Dry-run merges a 2-member shaft family and writes the expected artifacts."""
    families_dir = tmp_path / "families"
    families_dir.mkdir()
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    out_dir = tmp_path / "out"

    _make_template(templates_dir, "C1", n_params=4, n_components=2, n_features=4)
    _make_template(templates_dir, "C2", n_params=4, n_components=2, n_features=4)
    _make_family(families_dir, "fam_shaft_X", ["C1", "C2"], category_root="shaft")
    _write_index(families_dir, [
        {"family_id": "fam_shaft_X", "category_root": "shaft",
         "members": ["C1", "C2"], "n_members": 2},
    ])

    result = _run_cli(families_dir, templates_dir, out_dir, family_id="fam_shaft_X")
    assert result.returncode == 0, (
        f"CLI failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    merged_path = out_dir / "fam_shaft_X" / "merged_tables.json"
    report_path = out_dir / "fam_shaft_X" / "merge_report.md"
    assert merged_path.exists(), f"missing {merged_path}; CLI out={result.stdout}"
    assert report_path.exists(), f"missing {report_path}; CLI out={result.stdout}"

    merged = json.loads(merged_path.read_text())
    assert merged["family_id"] == "fam_shaft_X"
    assert merged["n_members"] == 2
    assert "parameters" in merged["sections"]
    assert "components" in merged["sections"]
    assert "feature_graph" in merged["sections"]
    # Headers are canonicalized — synonyms collapsed.
    params_sec = merged["sections"]["parameters"]
    for h in params_sec["headers"]:
        assert h in {"name", "symbol", "description", "parameter_type", "category",
                     "default_value", "units", "tolerance", "notes"} | {f"param_{i}" for i in range(4)}, (
            f"unexpected canonical header: {h!r}"
        )

    report = report_path.read_text()
    assert "fam_shaft_X" in report
    assert "parameters" in report


def test_step0_6_skips_missing_template(tmp_path: Path):
    """If one member's feature_template.json is missing, skip with a warning."""
    families_dir = tmp_path / "families"
    families_dir.mkdir()
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    out_dir = tmp_path / "out"

    _make_template(templates_dir, "M1")
    _make_template(templates_dir, "M3")  # M2 intentionally missing
    _make_family(families_dir, "fam_skip", ["M1", "M2", "M3"], category_root="shaft")
    _write_index(families_dir, [
        {"family_id": "fam_skip", "category_root": "shaft",
         "members": ["M1", "M2", "M3"], "n_members": 3},
    ])

    result = _run_cli(families_dir, templates_dir, out_dir, family_id="fam_skip")
    assert result.returncode == 0, (
        f"CLI failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    # A warning about the missing template should appear on stderr.
    assert "M2" in result.stderr or "missing" in result.stderr.lower(), (
        f"expected a missing-template warning; stderr={result.stderr!r}"
    )

    merged_path = out_dir / "fam_skip" / "merged_tables.json"
    assert merged_path.exists()
    merged = json.loads(merged_path.read_text())
    # n_members in the output reflects only the 2 present members.
    assert merged["n_members"] == 2, (
        f"expected n_members=2 (M1+M3); got {merged['n_members']}"
    )
    # The source_component_ids in each section must not include M2.
    for sec_name, sec_doc in merged["sections"].items():
        src_ids = sec_doc.get("source_component_ids") or []
        assert "M2" not in src_ids, (
            f"section {sec_name!r} should not list M2 as a source; got {src_ids}"
        )
        assert set(src_ids) <= {"M1", "M3"}, (
            f"section {sec_name!r} unexpected source ids: {src_ids}"
        )


def test_step0_6_single_family_filter(tmp_path: Path):
    """With --family-id, only that family's merged_tables.json is written."""
    families_dir = tmp_path / "families"
    families_dir.mkdir()
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    out_dir = tmp_path / "out"

    _make_template(templates_dir, "X1")
    _make_template(templates_dir, "X2")
    _make_template(templates_dir, "Y1")
    _make_template(templates_dir, "Y2")
    _make_family(families_dir, "fam_X", ["X1", "X2"], category_root="shaft")
    _make_family(families_dir, "fam_Y", ["Y1", "Y2"], category_root="bracket")
    _write_index(families_dir, [
        {"family_id": "fam_X", "category_root": "shaft", "members": ["X1", "X2"], "n_members": 2},
        {"family_id": "fam_Y", "category_root": "bracket", "members": ["Y1", "Y2"], "n_members": 2},
    ])

    result = _run_cli(families_dir, templates_dir, out_dir, family_id="fam_X")
    assert result.returncode == 0, (
        f"CLI failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    assert (out_dir / "fam_X" / "merged_tables.json").exists()
    assert not (out_dir / "fam_Y" / "merged_tables.json").exists(), (
        "fam_Y was processed despite --family-id fam_X"
    )


def test_step0_6_writes_merge_report_with_conflict_count(tmp_path: Path):
    """Two members with conflicting parameter values → report has Conflicts: N > 0."""
    families_dir = tmp_path / "families"
    families_dir.mkdir()
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    out_dir = tmp_path / "out"

    # Both members share the same parameter symbol D and collide on default_value.
    _make_template(templates_dir, "A", n_params=3, param_overrides={0: {"symbol": "D", "default_value": 20}})
    _make_template(templates_dir, "B", n_params=3, param_overrides={0: {"symbol": "D", "default_value": 25}})
    _make_family(families_dir, "fam_conflict", ["A", "B"], category_root="shaft")
    _write_index(families_dir, [
        {"family_id": "fam_conflict", "category_root": "shaft",
         "members": ["A", "B"], "n_members": 2},
    ])

    result = _run_cli(families_dir, templates_dir, out_dir, family_id="fam_conflict")
    assert result.returncode == 0, (
        f"CLI failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    report = (out_dir / "fam_conflict" / "merge_report.md").read_text()
    assert "Conflicts:" in report, (
        f"expected a 'Conflicts:' line in the report; got:\n{report}"
    )

    merged = json.loads((out_dir / "fam_conflict" / "merged_tables.json").read_text())
    n_conflicts = sum(s.get("n_conflicts", 0) for s in merged["sections"].values())
    assert n_conflicts > 0, (
        f"expected >=1 conflict across sections; got sections={list(merged['sections'].keys())} "
        f"counts={ {k: s.get('n_conflicts', 0) for k, s in merged['sections'].items()} }"
    )
    # The headline conflict count (first 'Conflicts:' line in the doc) should
    # match the merged-table count and be > 0. Parse it back out robustly.
    headline_match = re.search(r"Conflicts:\s*(\d+)", report)
    assert headline_match, f"no 'Conflicts: N' line in report:\n{report}"
    headline_n = int(headline_match.group(1))
    assert headline_n == n_conflicts, (
        f"report headline says {headline_n} conflicts, but {n_conflicts} found in JSON"
    )
    assert headline_n > 0


def test_step0_6_dry_run_does_not_call_openai(tmp_path: Path):
    """In --dry-run, the CLI must never import the `openai` package.

    Strategy: prepend a tmpdir to PYTHONPATH that contains a stub
    ``openai/__init__.py`` which raises ImportError on attribute access.
    If the CLI's synonym/conflict resolvers ever try to import or use the
    real openai module, the subprocess will exit non-zero and the test fails.
    The CLI is expected to pass ``openai_client=None`` through the dry-run
    path, so it should never touch the package at all.
    """
    fake = tmp_path / "fakepkg"
    fake.mkdir()
    (fake / "openai").mkdir()
    (fake / "openai" / "__init__.py").write_text(
        'class _Forbidden:\n'
        '    def __getattr__(self, name):\n'
        '        raise ImportError("openai package is forbidden during --dry-run tests")\n'
        '\n'
        'def __getattr__(name):\n'
        '    raise ImportError("openai package is forbidden during --dry-run tests")\n'
    )

    families_dir = tmp_path / "families"
    families_dir.mkdir()
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    out_dir = tmp_path / "out"

    _make_template(templates_dir, "C1")
    _make_template(templates_dir, "C2")
    _make_family(families_dir, "fam_dry", ["C1", "C2"], category_root="shaft")
    _write_index(families_dir, [
        {"family_id": "fam_dry", "category_root": "shaft",
         "members": ["C1", "C2"], "n_members": 2},
    ])

    result = _run_cli(
        families_dir, templates_dir, out_dir,
        family_id="fam_dry", dry_run=True, extra_pythonpath=fake,
    )
    assert result.returncode == 0, (
        "CLI in --dry-run imported the openai package:\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
