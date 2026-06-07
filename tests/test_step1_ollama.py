"""Tests for the Ollama-backend Step 1 wrapper.

These tests do not require an Ollama server. They exercise:
  - evaluate_template: schema-strict vs engineering counts, structural_issues,
    cad_ready logic, and tolerance to model-native key names.
  - extract_template: code-fence stripping and thinking-fallback.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.step1_ollama import evaluate_template, extract_template


# ---------------------------------------------------------------------------
# evaluate_template
# ---------------------------------------------------------------------------


def _strict_template() -> dict:
    """A perfectly schema-compliant template (matches step1's strict schema)."""
    return {
        "schema_version": "1.0",
        "template_id": "T",
        "part_family": "Test family",
        "units": "mm",
        "modeling_mode": "simplified_single_body",
        "coordinate_system": {
            "origin": "origin",
            "x_axis": "+X",
            "y_axis": "+Y",
            "z_axis": "+Z",
        },
        "components": [
            {
                "id": "body_1",
                "name": "Main body",
                "component_type": "solid_body",
                "description": "x",
                "parent_component": None,
                "created_by_feature": "f1",
            }
        ],
        "parameters": [
            {
                "name": "D",
                "symbol": "$D",
                "description": "diameter",
                "parameter_type": "diameter",
                "category": "geometry",
                "source": "datasheet",
                "default_value": 10.0,
                "units": "mm",
                "tolerance": None,
                "notes": "",
            }
        ],
        "feature_graph": [
            {
                "id": "f1",
                "feature_type": "base_body",
                "operation": "base",
                "modeling_primitive": "extrude",
                "subtype": "rect",
                "target_bodies": [],
                "output_bodies": ["body_1"],
                "depends_on": [],
                "parameters": [
                    {"name": "D", "value": "$D", "role": "driving_dimension"},
                ],
                "construction": {},
                "position": {},
                "axis": "Z",
                "side": "top",
                "pattern": None,
                "confidence": 0.9,
                "needs_review": False,
                "review_reason": "",
                "notes": "",
            }
        ],
        "catalog_metadata": {
            "material_options": [],
            "surface_treatment_options": [],
            "cleaning_options": [],
            "packaging_options": [],
            "accessories": [],
            "notes": "",
        },
        "assumptions": [],
        "missing_information": [],
        "validation_requirements": [
            {
                "check": "diameter matches D",
                "target_feature": "f1",
                "target_body": "body_1",
                "expected": "10.0 mm",
                "tolerance": "0.1",
                "validation_method": "diameter_measurement",
            }
        ],
    }


def _native_template() -> dict:
    """A qwen3.6-style template: features, validation_checks, etc."""
    return {
        "modeling_mode": "simplified_single_body",
        "parameters": [
            {"symbol": "$L", "description": "Length", "unit": "mm"},
            {"symbol": "$D", "description": "Diameter", "unit": "mm"},
        ],
        "features": [
            {
                "feature_id": "base",
                "feature_type": "base_body",
                "operation": "add",
                "modeling_primitive": "extrude",
                "symbolic_driving_parameters": ["$D", "$L"],
                "target_body": "main_body",
                "dependencies": [],
                "confidence_score": 0.9,
                "needs_review": False,
            },
            {
                "feature_id": "hole",
                "feature_type": "hole",
                "operation": "subtract",
                "modeling_primitive": "hole",
                "symbolic_driving_parameters": ["$D"],
                "target_body": "main_body",
                "dependencies": ["base"],
                "confidence_score": 0.8,
                "needs_review": True,
            },
        ],
        "metadata": {
            "assumptions": ["single body"],
        },
        "missing_information": ["internal bore size"],
        "validation_checks": [
            {"check_type": "bbox", "description": "x", "target_feature": "base"},
        ],
    }


class EvaluateTemplateTests(unittest.TestCase):
    def test_strict_template_is_cad_ready(self) -> None:
        ev = evaluate_template(_strict_template())
        self.assertTrue(ev["schema_compliant"])
        self.assertEqual(ev["issue_count"], 0)
        self.assertTrue(ev["cad_ready"])
        # strict and engineering views should match for a strict template
        self.assertEqual(ev["schema_strict"]["features"], ev["engineering"]["features"])

    def test_native_template_fails_schema_compliance(self) -> None:
        ev = evaluate_template(_native_template())
        self.assertFalse(ev["schema_compliant"])
        # but the engineering view should still pick up the 2 features
        self.assertEqual(ev["engineering"]["features"], 2)
        self.assertEqual(ev["engineering"]["parameters"], 2)
        self.assertEqual(ev["engineering"]["needs_review_features"], 1)
        # validation_checks -> engineering.validation_requirements
        self.assertEqual(ev["engineering"]["validation_requirements"], 1)
        # bad_body_refs: target_body="main_body" but no components[] array
        self.assertEqual(len(ev["structural_issues"]["bad_body_refs"]), 2)
        # missing_information -> not cad_ready
        self.assertFalse(ev["cad_ready"])

    def test_null_driving_dimension_flagged(self) -> None:
        t = _strict_template()
        # set the driving dimension value to None
        t["feature_graph"][0]["parameters"][0]["value"] = None
        ev = evaluate_template(t)
        self.assertEqual(len(ev["structural_issues"]["null_driving_dimensions"]), 1)
        self.assertFalse(ev["cad_ready"])

    def test_bad_dependency_ref_flagged(self) -> None:
        t = _strict_template()
        t["feature_graph"][0]["depends_on"] = ["ghost_feature"]
        ev = evaluate_template(t)
        self.assertEqual(len(ev["structural_issues"]["bad_dependency_refs"]), 1)

    def test_bad_parameter_ref_flagged(self) -> None:
        t = _strict_template()
        t["feature_graph"][0]["parameters"] = [
            {"name": "Z", "value": "$Z", "role": "driving_dimension"}
        ]
        ev = evaluate_template(t)
        self.assertEqual(len(ev["structural_issues"]["bad_parameter_refs"]), 1)

    def test_empty_native_template(self) -> None:
        # The qwen3.6 1-line stub seen in practice
        t = {"type": "parametric_feature_template"}
        ev = evaluate_template(t)
        self.assertEqual(ev["engineering"]["features"], 0)
        self.assertEqual(ev["issue_count"], 0)
        # CAD-ready requires at least 1 feature and 1 validation check
        self.assertFalse(ev["cad_ready"])


# ---------------------------------------------------------------------------
# extract_template
# ---------------------------------------------------------------------------


class ExtractTemplateTests(unittest.TestCase):
    def test_direct_json(self) -> None:
        body = {"response": '{"a": 1, "b": [1,2,3]}'}
        out = extract_template(body)
        self.assertEqual(out, {"a": 1, "b": [1, 2, 3]})

    def test_strips_code_fences(self) -> None:
        body = {"response": "```json\n{\"a\": 2}\n```"}
        out = extract_template(body)
        self.assertEqual(out, {"a": 2})

    def test_falls_back_to_thinking(self) -> None:
        body = {"response": "", "thinking": '{"a": 3}'}
        out = extract_template(body)
        self.assertEqual(out, {"a": 3})

    def test_empty_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            extract_template({"response": "", "thinking": ""})


if __name__ == "__main__":
    unittest.main()
