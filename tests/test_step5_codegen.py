"""Tests for scripts/step5_codegen.py — the deterministic CadQuery code generator.

These are smoke tests, not full coverage. The goal is to lock in:
  * 12/12 resolved specs parse, generate, and pass schema validation
  * Every generated model.py is syntactically valid Python
  * Every generated model.py runs to completion in a fresh subprocess and
    produces both .step and .stl outputs
  * validate() returns the expected shape: bounding box, volume, feature_log,
    not_implemented
  * The "not_implemented" list surfaces unsupported features honestly
    (so Step 7 can flag what's missing, not hide it)

Run with:
    cd ~/fe/cadquery_generator
    source .venv/bin/activate
    pytest tests/test_step5_codegen.py -v
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import step5_codegen  # noqa: E402

SPECS_DIR = REPO_ROOT / "output" / "resolved_specs"
OUT_DIR = REPO_ROOT / "output" / "cad_models"
SCHEMA = REPO_ROOT / "schemas" / "resolved_cad_spec.schema.json"

ALL_SPECS = sorted(SPECS_DIR.glob("*.resolved_cad_spec.json"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_one(spec_path: Path, smoke: bool = False) -> step5_codegen.CodegenResult:
    schema = json.loads(SCHEMA.read_text())
    return step5_codegen.process_spec(
        spec_path, OUT_DIR, schema, dry_run=False, smoke=smoke
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def all_results() -> list[step5_codegen.CodegenResult]:
    """Run Step 5 across the whole corpus once, share across tests."""
    return [_generate_one(p, smoke=True) for p in ALL_SPECS]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_every_spec_is_discovered():
    """We have 12 resolved specs in the corpus; make sure the discovery
    helper sees all of them and parses part numbers correctly."""
    assert len(ALL_SPECS) == 12, f"expected 12 specs, found {len(ALL_SPECS)}"
    for p in ALL_SPECS:
        pn = step5_codegen.part_number_of(p)
        cid = step5_codegen.component_id_of(p)
        assert "__" not in pn, f"part number {pn!r} still has '__'"
        assert pn, f"empty part number for {p.name}"
        assert cid.isdigit(), f"component id {cid!r} not numeric"


def test_every_spec_generates_and_passes_smoke(all_results):
    """12/12 specs should produce valid .py files that run to completion."""
    failed = [r for r in all_results if not r.ok]
    assert not failed, f"failures: {[(r.part_number, r.error) for r in failed]}"
    assert all(r.runtime_smoke for r in all_results), "all should have run the smoke test"


def test_every_model_file_is_valid_python(all_results):
    """The generated model.py files must parse with ast.parse."""
    for r in all_results:
        code = (OUT_DIR / f"{r.part_number}.model.py").read_text()
        try:
            ast.parse(code)
        except SyntaxError as e:
            pytest.fail(f"{r.part_number}.model.py: SyntaxError {e}")


def test_every_model_file_exports_step_and_stl(all_results):
    """Each model.py should produce both <pn>.step and <pn>.stl when run.
    The .step should be a real STEP file (ISO 10303-21 header); the .stl
    can be small if the part itself is a placeholder 1x1x1mm cube (which
    is what the deterministic emitter produces when no feature has
    resolvable circle/polygon/revolve geometry)."""
    for r in all_results:
        step = OUT_DIR / f"{r.part_number}.step"
        stl = OUT_DIR / f"{r.part_number}.stl"
        assert step.exists() and step.stat().st_size > 200, f"{r.part_number}.step missing"
        assert stl.exists() and stl.stat().st_size > 100, f"{r.part_number}.stl missing"
        # STEP should start with the ISO 10303-21 header
        with open(step, "rb") as f:
            head = f.read(20)
        assert b"ISO-10303-21" in head, f"{r.part_number}.step is not a valid STEP file"


def test_validate_returns_expected_shape(all_results):
    """validate() should return the documented shape."""
    # Import one model and check the dict shape
    import importlib.util
    spec = importlib.util.spec_from_file_location("m", OUT_DIR / "MCSCN10.model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    v = mod.validate()
    assert set(v.keys()) == {"part_number", "bounding_box", "volume_mm3", "feature_log", "not_implemented"}
    bb = v["bounding_box"]
    assert len(bb) == 3
    assert all(isinstance(x, (int, float)) for x in bb)
    assert v["volume_mm3"] > 0
    assert isinstance(v["feature_log"], list)
    assert isinstance(v["not_implemented"], list)


def test_mcscn10_dimensions_match_datasheet():
    """MCSCN10 is a cylinder of body_outer_diameter=26.5, overall_length=64.5.
    Bounding box should be (64.5, 26.5, 26.5) and volume ~ pi*r^2*L."""
    import importlib.util
    import math
    spec = importlib.util.spec_from_file_location("m", OUT_DIR / "MCSCN10.model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    v = mod.validate()
    bb = v["bounding_box"]
    assert abs(bb[0] - 64.5) < 0.1, f"length off: {bb[0]}"
    assert abs(bb[1] - 26.5) < 0.1, f"width off: {bb[1]}"
    assert abs(bb[2] - 26.5) < 0.1, f"height off: {bb[2]}"
    expected_volume = math.pi * (26.5 / 2) ** 2 * 64.5
    assert abs(v["volume_mm3"] - expected_volume) / expected_volume < 0.01


def test_not_implemented_honest_for_unsupported_features():
    """Features that aren't implemented (e.g. custom_2d_profile pockets in
    the linear shaft templates) should appear in not_implemented, not be
    silently dropped. This is what lets Step 7 flag missing geometry."""
    # SH-PSSGTN20 has a base_body with circle extrude but the D/L values are
    # unresolved expression references ('D', 'F40') so the emitter skips it.
    import importlib.util
    spec = importlib.util.spec_from_file_location("m", OUT_DIR / "SH-PSSGTN20.model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    v = mod.validate()
    assert "f0_base_body" in v["not_implemented"], (
        "f0_base_body should be flagged as not implemented (string-valued "
        "D/L parameters) so Step 7 sees it"
    )
    # But the linear-shaft placeholders (everything else) should also be
    # listed, since their construction_notes describe real geometry we
    # can't deterministically emit.
    assert len(v["not_implemented"]) >= 5, (
        f"expected many not_implemented entries for the linear shaft, "
        f"got {v['not_implemented']}"
    )


def test_pocket_cut_extrude_emits_geometry():
    """Pocket cut_extrude should emit a guarded CadQuery cut, not a TODO."""
    spec = {
        "part_number": "POCKET_TEST",
        "catalog_id": "0",
        "template_id": "unit_test",
        "parameter_bindings": {"W": 6, "L1": 12, "overall_length": 50},
        "resolved_features": [
            {
                "id": "f_slot",
                "feature_type": "pocket",
                "subtype": "side_slot",
                "modeling_primitive": "cut_extrude",
                "parameters": [
                    {"name": "W", "value": 6},
                    {"name": "L1", "value": 12},
                ],
                "construction": {
                    "profile_type": "rectangle",
                    "depth": "SC + SX + L1",
                },
                "position": {"x": "SC", "y": 0, "z": 0},
            }
        ],
    }
    code, _ = step5_codegen.generate_model_py(
        spec, REPO_ROOT / "tests" / "fixtures" / "pocket.resolved_cad_spec.json"
    )
    assert "# Feature: f_slot" in code
    assert "pocket_wp = (" in code
    assert ".workplane(offset=0, centerOption='CenterOfMass')" in code
    assert ".rect(w, l1)" in code
    assert ".extrude(overall_length)" in code
    assert "feature_log.append('f_slot')" in code
    assert "not_implemented.append('f_slot')" in code
