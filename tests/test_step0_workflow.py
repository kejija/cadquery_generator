import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.step0_main_engineering_drawing import (
    apply_model_selection_guard,
    drawing_files,
    parse_ollama_generate_response,
    run_step0_batch,
)
from scripts.workflow_state import connect_state_db, count_by_status


def create_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (200, 120), "white").save(path)


def create_component(root: Path, component_id: str, image_names: list[str]) -> Path:
    component = root / component_id
    for name in image_names:
        if name == "empty.gif":
            path = component / "drawings" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")
        else:
            create_image(component / "drawings" / name)
    (component / "json").mkdir(parents=True, exist_ok=True)
    return component


class Step0WorkflowTests(unittest.TestCase):
    def test_candidate_discovery_ignores_zero_byte_and_noise_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            component = create_component(
                Path(tmp),
                "component_a",
                ["drw_01.gif", "empty.gif", "product_photo.jpg", "cleaning_process_101.gif"],
            )

            self.assertEqual([path.name for path in drawing_files(component)], ["drw_01.gif"])

    def test_qwen_thinking_field_is_parsed_as_ollama_json(self) -> None:
        parsed, source = parse_ollama_generate_response(
            {
                "response": "",
                "thinking": '{"filename":"drw_02_100.gif","confidence":0.95,"reason":"main drawing"}',
            }
        )

        self.assertEqual(source, "thinking")
        self.assertEqual(parsed["filename"], "drw_02_100.gif")

    def test_model_selection_guard_rejects_materially_weaker_small_pick(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            component = create_component(root, "component_a", [])
            top = component / "drawings" / "drw_02.gif"
            weak = component / "drawings" / "drw_31.gif"
            top.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (1000, 250), "white").save(top)
            Image.new("RGB", (180, 110), "white").save(weak)
            result = {"filename": weak.name, "confidence": 0.95, "_method": "ollama"}

            selected = apply_model_selection_guard(weak, [top, weak], result)

            self.assertEqual(selected, top)
            self.assertEqual(result["filename"], top.name)
            self.assertEqual(result["_method"], "ollama_guarded_heuristic")
            self.assertEqual(result["_ollama_filename"], weak.name)

    def test_skip_completed_returns_skipped_without_modifying_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            component = create_component(root, "component_a", ["drw_01.gif"])
            main_dir = component / "main_engineering_drawing"
            main_dir.mkdir()
            manifest_path = main_dir / "selection.json"
            manifest = {
                "component_id": "component_a",
                "source_path": str(component / "drawings" / "drw_01.gif"),
                "copied_path": str(main_dir / "drw_01.gif"),
                "ollama_model": "test",
                "selection": {"filename": "drw_01.gif", "confidence": 0.8, "_method": "test"},
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            before = manifest_path.read_text(encoding="utf-8")

            completed, errors = run_step0_batch(
                [component],
                "test",
                state_db=root / "state.sqlite3",
                force=False,
            )

            self.assertEqual(errors, [])
            self.assertEqual(completed[0].status, "skipped")
            self.assertEqual(manifest_path.read_text(encoding="utf-8"), before)

    def test_force_recomputes_and_overwrites_existing_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            component = create_component(root, "component_a", ["drw_01.gif"])
            main_dir = component / "main_engineering_drawing"
            main_dir.mkdir()
            manifest_path = main_dir / "selection.json"
            manifest_path.write_text('{"old": true}', encoding="utf-8")

            completed, errors = run_step0_batch(
                [component],
                "test",
                state_db=root / "state.sqlite3",
                force=True,
            )

            self.assertEqual(errors, [])
            self.assertEqual(completed[0].status, "done")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["component_id"], "component_a")
            self.assertEqual(manifest["selection"]["_method"], "single")

    def test_missing_drawings_marks_error_and_continues(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = create_component(root, "component_good", ["drw_01.gif"])
            bad = root / "component_bad"
            bad.mkdir()

            completed, errors = run_step0_batch(
                [bad, good],
                "test",
                state_db=root / "state.sqlite3",
                force=True,
            )

            self.assertEqual(len(completed), 1)
            self.assertEqual(len(errors), 1)
            self.assertEqual(errors[0].component_id, "component_bad")
            self.assertIn("No usable drawing files", errors[0].message)

    def test_sqlite_state_updates_for_done_skipped_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            done = create_component(root, "component_done", ["drw_01.gif"])
            skipped = create_component(root, "component_skipped", ["drw_01.gif"])
            main_dir = skipped / "main_engineering_drawing"
            main_dir.mkdir()
            (main_dir / "selection.json").write_text(
                json.dumps(
                    {
                        "component_id": "component_skipped",
                        "source_path": str(skipped / "drawings" / "drw_01.gif"),
                        "copied_path": str(main_dir / "drw_01.gif"),
                        "ollama_model": "test",
                        "selection": {"filename": "drw_01.gif", "confidence": 0.7, "_method": "test"},
                    }
                ),
                encoding="utf-8",
            )
            error = root / "component_error"
            error.mkdir()

            run_step0_batch(
                [done, skipped, error],
                "test",
                state_db=root / "state.sqlite3",
                force=False,
            )

            with connect_state_db(root / "state.sqlite3") as conn:
                counts = count_by_status(conn)
            self.assertEqual(counts["done"], 1)
            self.assertEqual(counts["skipped"], 1)
            self.assertEqual(counts["error"], 1)


if __name__ == "__main__":
    unittest.main()
