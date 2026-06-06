from __future__ import annotations

import json
import subprocess
from pathlib import Path


def test_qc_regression_fixtures(tmp_path: Path) -> None:
    output = tmp_path / "qc"
    proc = subprocess.run(
        [
            ".venv/bin/python",
            "scripts/qc_cadquery_outputs.py",
            "--outputs",
            "tests/fixtures/qc",
            "--downloads",
            "downloads",
            "--output",
            str(output),
            "--skip-vision",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    rows = json.loads((output / "qc_summary.json").read_text(encoding="utf-8"))
    by_id = {row["component_id"]: row for row in rows}
    assert by_id["empty_script"]["overall_status"] == "fail"
    assert by_id["show_object_only"]["overall_status"] == "fail"
    assert by_id["wrong_envelope"]["overall_status"] == "review"
    assert by_id["assembly"]["cad_metrics"]["assembly_children"] == 2
