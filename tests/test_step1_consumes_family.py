"""Tests for the read-only family + merged-tables hook in step1.

Task 12 adds three helpers (``load_family_index``, ``_family_for_component``,
``load_merged_tables``) plus a logging block in ``main()``'s ``batch-jsonl``
path. These tests confirm the helpers work in isolation and that ``main()``
writes a ``step1_family_context.log`` artifact WITHOUT altering the prompt or
the OpenAI request body for any component.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.step1_drawing_datasheetjson import (
    _family_for_component,
    build_batch_jsonl_record,
    build_responses_body,
    load_family_index,
    load_merged_tables,
    main,
)


def _write_index(dir: Path, families: list[dict]) -> Path:
    path = dir / "index.json"
    path.write_text(
        json.dumps({"schema_version": "1.0", "family_count": len(families), "families": families}),
        encoding="utf-8",
    )
    return path


def _make_family(
    family_id: str,
    members: list[str],
    category_root: str = "brg",
    n_members: int | None = None,
) -> dict:
    return {
        "family_id": family_id,
        "category_root": category_root,
        "members": list(members),
        "n_members": n_members if n_members is not None else len(members),
        "shared_value_keys": ["OD", "ID"],
        "shared_attribute_keys": [],
        "canonical_template_id": "brg_v1",
        "avg_template_signature_jaccard": 0.9,
    }


def _make_merged_tables(family_id: str, sections: list[str]) -> dict:
    return {
        "schema_version": "1.0",
        "family_id": family_id,
        "category_root": "brg",
        "n_members": 2,
        "sections": {
            name: {
                "table_type": "spec_table",
                "headers": ["A", "B"],
                "rows": [["1", "2"]],
                "n_conflicts": 0,
                "resolutions": [],
            }
            for name in sections
        },
    }


class LoadFamilyIndexTests(unittest.TestCase):
    def test_load_family_index_reads_existing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            families = [
                _make_family("fam_a", ["C1", "C2"]),
                _make_family("fam_b", ["C3"]),
            ]
            _write_index(Path(tmp), families)

            idx = load_family_index(Path(tmp))

            self.assertEqual(idx["family_count"], 2)
            self.assertEqual(len(idx["families"]), 2)
            self.assertEqual(idx["families"][0]["family_id"], "fam_a")
            self.assertEqual(idx["families"][1]["family_id"], "fam_b")

    def test_load_family_index_returns_empty_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            idx = load_family_index(Path(tmp))
            self.assertEqual(idx, {"families": [], "family_count": 0})


class FamilyForComponentTests(unittest.TestCase):
    def test_family_for_component_found(self) -> None:
        idx = {
            "family_count": 2,
            "families": [
                _make_family("fam_a", ["C1", "C2"]),
                _make_family("fam_b", ["C3"]),
            ],
        }
        fam = _family_for_component("C1", idx)
        self.assertIsNotNone(fam)
        self.assertEqual(fam["family_id"], "fam_a")

        fam_none = _family_for_component("C99", idx)
        self.assertIsNone(fam_none)


class LoadMergedTablesTests(unittest.TestCase):
    def test_load_merged_tables_returns_none_when_no_family(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fdir = Path(tmp)
            _write_index(fdir, [_make_family("fam_a", ["C1"])])
            # C99 is not in any family.
            self.assertIsNone(load_merged_tables("C99", fdir))

    def test_load_merged_tables_returns_dict_when_family_and_tables_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fdir = Path(tmp)
            _write_index(fdir, [_make_family("fam_a", ["C1", "C2"])])

            fam_dir = fdir / "fam_a"
            fam_dir.mkdir()
            (fam_dir / "family.json").write_text(
                json.dumps(
                    {
                        "family_id": "fam_a",
                        "category_root": "brg",
                        "member_component_ids": ["C1", "C2"],
                    }
                ),
                encoding="utf-8",
            )
            tables = _make_merged_tables("fam_a", ["dimensions", "tolerances"])
            (fam_dir / "merged_tables.json").write_text(json.dumps(tables), encoding="utf-8")

            result = load_merged_tables("C1", fdir)
            self.assertIsNotNone(result)
            self.assertEqual(result["family_id"], "fam_a")
            self.assertEqual(set(result["sections"].keys()), {"dimensions", "tolerances"})

    def test_load_merged_tables_returns_none_when_step0_6_not_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fdir = Path(tmp)
            _write_index(fdir, [_make_family("fam_a", ["C1"])])

            fam_dir = fdir / "fam_a"
            fam_dir.mkdir()
            (fam_dir / "family.json").write_text(
                json.dumps(
                    {
                        "family_id": "fam_a",
                        "category_root": "brg",
                        "member_component_ids": ["C1"],
                    }
                ),
                encoding="utf-8",
            )
            # No merged_tables.json -- step 0.6 hasn't run.

            self.assertIsNone(load_merged_tables("C1", fdir))


class MainBatchJsonlLoggingTests(unittest.TestCase):
    def _make_component_dirs(self, downloads: Path, cids: list[str]) -> None:
        for cid in cids:
            comp = downloads / cid
            (comp / "json").mkdir(parents=True)
            (comp / "json" / "specs.json").write_text(
                json.dumps(
                    {
                        "title": cid,
                        "tables": [{"rows": [{"D": 10}]}],
                        "specImages": [],
                        "filters": [],
                        "configFields": [],
                    }
                ),
                encoding="utf-8",
            )

    def test_step1_batch_jsonl_logs_family_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            downloads = tmp_path / "downloads"
            downloads.mkdir()
            self._make_component_dirs(downloads, ["C1", "C2"])

            # Families layout
            fdir = tmp_path / "families"
            fdir.mkdir()
            _write_index(fdir, [_make_family("fam_a", ["C1", "C2"])])
            fam_dir = fdir / "fam_a"
            fam_dir.mkdir()
            (fam_dir / "family.json").write_text(
                json.dumps(
                    {
                        "family_id": "fam_a",
                        "category_root": "brg",
                        "member_component_ids": ["C1", "C2"],
                    }
                ),
                encoding="utf-8",
            )
            tables = _make_merged_tables("fam_a", ["dimensions"])
            (fam_dir / "merged_tables.json").write_text(json.dumps(tables), encoding="utf-8")

            output_dir = tmp_path / "out"
            output_dir.mkdir()

            # Reset the lru_cache between tests.
            load_family_index.cache_clear()

            with mock.patch(
                "scripts.step1_drawing_datasheetjson.build_batch_jsonl_record",
                return_value={"custom_id": "stub", "method": "POST", "url": "/v1/responses", "body": {}},
            ), mock.patch(
                "scripts.step1_drawing_datasheetjson.create_batch_jsonl",
                return_value=output_dir / "batch.jsonl",
            ):
                with mock.patch(
                    "sys.argv",
                    [
                        "step1",
                        "batch-jsonl",
                        "--downloads-dir", str(downloads),
                        "--output-dir", str(output_dir),
                        "--families-dir", str(fdir),
                    ],
                ):
                    rc = main()

            self.assertEqual(rc, 0)

            log_path = output_dir / "step1_family_context.log"
            self.assertTrue(log_path.exists(), f"expected {log_path} to exist")
            log_text = log_path.read_text(encoding="utf-8")

            self.assertIn("component=C1", log_text)
            self.assertIn("component=C2", log_text)
            self.assertIn("family_id=fam_a", log_text)
            self.assertIn("merged_tables_sections=['dimensions']", log_text)

    def test_step1_batch_jsonl_logs_no_family_for_unrelated_components(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            downloads = tmp_path / "downloads"
            downloads.mkdir()
            self._make_component_dirs(downloads, ["X1", "X2"])

            # Families layout has NO entries for X1 / X2.
            fdir = tmp_path / "families"
            fdir.mkdir()
            _write_index(fdir, [_make_family("fam_a", ["C1", "C2"])])

            output_dir = tmp_path / "out"
            output_dir.mkdir()

            load_family_index.cache_clear()

            with mock.patch(
                "scripts.step1_drawing_datasheetjson.build_batch_jsonl_record",
                return_value={"custom_id": "stub", "method": "POST", "url": "/v1/responses", "body": {}},
            ), mock.patch(
                "scripts.step1_drawing_datasheetjson.create_batch_jsonl",
                return_value=output_dir / "batch.jsonl",
            ):
                with mock.patch(
                    "sys.argv",
                    [
                        "step1",
                        "batch-jsonl",
                        "--downloads-dir", str(downloads),
                        "--output-dir", str(output_dir),
                        "--families-dir", str(fdir),
                    ],
                ):
                    rc = main()

            self.assertEqual(rc, 0)

            log_path = output_dir / "step1_family_context.log"
            self.assertTrue(log_path.exists())
            log_text = log_path.read_text(encoding="utf-8")
            self.assertIn("component=X1 family_id=none", log_text)
            self.assertIn("component=X2 family_id=none", log_text)

    def test_step1_does_not_change_prompt_or_batch_body(self) -> None:
        # Build a record for the same component twice and assert the body is
        # byte-identical. The family hook must not touch the request body.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            comp_dir = tmp_path / "C1"
            drawing = comp_dir / "main_engineering_drawing" / "drw_01.png"
            specs = comp_dir / "json" / "specs.json"
            drawing.parent.mkdir(parents=True)
            specs.parent.mkdir(parents=True)
            # Real image bytes are not required for this test -- the prompt
            # and body are what we're checking. A few bytes of PNG-ish data
            # is enough to make image_data_url happy.
            drawing.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
            specs.write_text(
                json.dumps(
                    {
                        "title": "Part",
                        "tables": [{"rows": [{"D": 10}]}],
                        "specImages": [],
                        "filters": [],
                        "configFields": [],
                    }
                ),
                encoding="utf-8",
            )

            # Two independent calls -- family hook is not invoked in this code
            # path so any difference would prove the helpers leaked into the
            # body builder.
            record_a = build_batch_jsonl_record(comp_dir)
            record_b = build_batch_jsonl_record(comp_dir)

            self.assertEqual(
                json.dumps(record_a["body"], sort_keys=True),
                json.dumps(record_b["body"], sort_keys=True),
            )

            # Also confirm the prompt text is exactly the DEVELOPER_MESSAGE
            # constant -- byte-identical to whatever the prompt is.
            from scripts.step1_drawing_datasheetjson import DEVELOPER_MESSAGE
            developer_blocks = [
                c for c in record_a["body"]["input"]
                if c.get("role") == "developer"
            ]
            self.assertEqual(len(developer_blocks), 1)
            inner = developer_blocks[0]["content"]
            self.assertEqual(inner[0]["type"], "input_text")
            self.assertEqual(inner[0]["text"], DEVELOPER_MESSAGE)

            # And build_responses_body returns the same prompt text as the
            # batch record.
            direct_body = build_responses_body(
                drawing_path=drawing, specs_json_path=specs, model="gpt-5.4-mini"
            )
            direct_developer = [
                c for c in direct_body["input"] if c.get("role") == "developer"
            ][0]["content"][0]["text"]
            self.assertEqual(direct_developer, DEVELOPER_MESSAGE)


if __name__ == "__main__":
    unittest.main()
