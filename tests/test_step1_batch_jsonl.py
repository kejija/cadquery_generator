import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.step1_drawing_datasheetjson import (
    RESPONSES_BATCH_ENDPOINT,
    build_batch_jsonl_record,
    find_specs_json,
    validate_batch_jsonl,
)


class Step1BatchJsonlTests(unittest.TestCase):
    def test_batch_record_targets_responses_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            component = Path(tmp) / "component_a"
            drawing = component / "main_engineering_drawing" / "drw_01.gif"
            specs = component / "json" / "specs.json"
            drawing.parent.mkdir(parents=True)
            specs.parent.mkdir(parents=True)
            Image.new("RGB", (80, 60), "white").save(drawing)
            specs.write_text(json.dumps({"title": "Test Part"}), encoding="utf-8")

            record = build_batch_jsonl_record(component)

            self.assertEqual(record["method"], "POST")
            self.assertEqual(record["url"], RESPONSES_BATCH_ENDPOINT)

    def test_validate_batch_jsonl_detects_endpoint_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / "batch.jsonl"
            jsonl.write_text(
                json.dumps(
                    {
                        "custom_id": "component_a",
                        "method": "POST",
                        "url": RESPONSES_BATCH_ENDPOINT,
                        "body": {"model": "test", "input": "hello"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "does not match requested batch endpoint"):
                validate_batch_jsonl(jsonl, endpoint="/v1/chat/completions")

    def test_validate_batch_jsonl_returns_detected_endpoint_and_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / "batch.jsonl"
            records = [
                {"custom_id": "a", "method": "POST", "url": RESPONSES_BATCH_ENDPOINT, "body": {}},
                {"custom_id": "b", "method": "POST", "url": RESPONSES_BATCH_ENDPOINT, "body": {}},
            ]
            jsonl.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

            endpoint, count = validate_batch_jsonl(jsonl)

            self.assertEqual(endpoint, RESPONSES_BATCH_ENDPOINT)
            self.assertEqual(count, 2)

    def test_find_specs_json_falls_back_when_specs_after_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            component = Path(tmp) / "component_a"
            json_dir = component / "json"
            json_dir.mkdir(parents=True)
            specs_after = json_dir / "specs_after.json"
            specs = json_dir / "specs.json"
            specs_after.write_text(
                json.dumps({"title": "empty cleaned file", "tables": [], "specImages": [], "filters": [], "configFields": []}),
                encoding="utf-8",
            )
            specs.write_text(
                json.dumps({"title": "source file", "tables": [{"rows": [{"D": 10}]}], "specImages": [], "filters": [], "configFields": []}),
                encoding="utf-8",
            )

            self.assertEqual(find_specs_json(component), specs)


if __name__ == "__main__":
    unittest.main()
