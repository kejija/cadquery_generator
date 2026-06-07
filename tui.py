#!/usr/bin/env python3
"""Textual dashboard for the data -> CAD batch workflow."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from scripts.step0_main_engineering_drawing import DEFAULT_OLLAMA_MODEL, Step0Event, run_step0_batch
from scripts.step1_drawing_datasheetjson import (
    DEFAULT_MODEL,
    create_batch_jsonl,
    download_batch_outputs,
    get_batch,
    submit_batch_jsonl,
)
from scripts.workflow_state import (
    DEFAULT_STATE_DB,
    component_paths_for_filter,
    completed_component_dirs,
    connect_state_db,
    count_by_status,
    count_filtered,
    latest_batch_job,
    list_batch_jobs,
    scan_downloads,
    upsert_batch_job,
)


STATUS_FILTERS = ["all", "pending", "done", "error", "skipped"]
STEP1_MODEL_CHOICES = ("gpt-5.4-mini", "gpt-5.5")
BATCH_TERMINAL_STATUSES = {"completed", "failed", "cancelled", "expired"}


@dataclass
class WorkflowStep:
    key: str
    label: str
    status: str
    enabled: bool
    selected: bool


@dataclass
class WorkflowState:
    downloads_dir: Path
    output_dir: Path
    state_db: Path = DEFAULT_STATE_DB
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    step1_model: str = DEFAULT_MODEL
    limit: int | None = None
    page: int = 0
    page_size: int = 100
    status_filter: str = "all"
    search: str = ""
    step1_jsonl_path: Path | None = None
    step_cursor: int = 0
    steps: list[WorkflowStep] = field(
        default_factory=lambda: [
            WorkflowStep("step0", "0 Main drawing", "ready", True, True),
            WorkflowStep("step1_jsonl", "1 Create JSONL", "ready", True, True),
            WorkflowStep("step1_submit", "1b Batch submit/output", "ready", True, False),
            WorkflowStep("step2_parse", "2 Parse templates", "placeholder", False, False),
            WorkflowStep("step3_cad", "3 Generate CAD", "placeholder", False, False),
            WorkflowStep("step4_qc", "4 Render/QC", "placeholder", False, False),
            WorkflowStep("step5_export", "5 Export artifacts", "placeholder", False, False),
        ]
    )
    message: str = "Ready"
    errors: list[str] = field(default_factory=list)


def index_downloads(state: WorkflowState) -> None:
    with connect_state_db(state.state_db) as conn:
        scan_downloads(conn, state.downloads_dir, state.limit)


def step0_scope(state: WorkflowState, *, force: bool = False) -> list[Path]:
    with connect_state_db(state.state_db) as conn:
        if force:
            status = state.status_filter
        else:
            status = state.status_filter if state.status_filter in {"pending", "error"} else "all"
        paths = component_paths_for_filter(conn, status=status, search=state.search, limit=state.limit)
    if force:
        return paths
    return [path for path in paths if not (path / "main_engineering_drawing" / "selection.json").exists()]


def run_step0_for_state(
    state: WorkflowState,
    *,
    force: bool = False,
    on_event=None,
) -> tuple[list[Step0Event], list[Step0Event]]:
    index_downloads(state)
    components = step0_scope(state, force=force)
    if not components:
        state.message = "No Step 0 components to process"
        return [], []
    completed, errors = run_step0_batch(
        components,
        state.ollama_model,
        state_db=state.state_db,
        force=force,
        workers=1,
        on_event=on_event,
    )
    state.message = f"Step 0 complete: completed={len(completed)} errors={len(errors)}"
    state.errors = [event.message for event in errors]
    return completed, errors


def run_step1_jsonl_for_state(state: WorkflowState) -> Path:
    with connect_state_db(state.state_db) as conn:
        components = completed_component_dirs(conn, status="done", limit=state.limit)
    output_path = create_batch_jsonl(
        component_dirs=components,
        output_dir=state.output_dir,
        model=state.step1_model,
    )
    state.step1_jsonl_path = output_path
    state.message = f"Created {output_path}"
    return output_path


def latest_step1_jsonl(state: WorkflowState) -> Path:
    if state.step1_jsonl_path and state.step1_jsonl_path.exists():
        return state.step1_jsonl_path
    candidates = sorted(
        state.output_dir.glob("step1_batch*.jsonl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No Step 1 batch JSONL found. Press j to create one first.")
    state.step1_jsonl_path = candidates[0]
    return candidates[0]


def submit_latest_batch_for_state(state: WorkflowState) -> dict:
    jsonl_path = latest_step1_jsonl(state)
    batch = submit_batch_jsonl(jsonl_path)
    with connect_state_db(state.state_db) as conn:
        upsert_batch_job(conn, batch, jsonl_path=jsonl_path)
    state.message = f"Submitted batch {batch['id']} status={batch.get('status')}"
    return batch


def latest_batch_id(state: WorkflowState) -> str:
    with connect_state_db(state.state_db) as conn:
        row = latest_batch_job(conn)
    if not row:
        raise FileNotFoundError("No batch job recorded. Submit a batch first.")
    return str(row["batch_id"])


def refresh_latest_batch_for_state(state: WorkflowState) -> dict:
    batch_id = latest_batch_id(state)
    batch = get_batch(batch_id)
    with connect_state_db(state.state_db) as conn:
        upsert_batch_job(conn, batch)
    counts = batch.get("request_counts") or {}
    state.message = (
        f"Batch {batch_id}: {batch.get('status')} "
        f"{counts.get('completed', 0)}/{counts.get('total', 0)} failed={counts.get('failed', 0)}"
    )
    return batch


def download_latest_batch_for_state(state: WorkflowState) -> dict:
    batch_id = latest_batch_id(state)
    result = download_batch_outputs(batch_id, state.output_dir)
    with connect_state_db(state.state_db) as conn:
        upsert_batch_job(
            conn,
            result["batch"],
            output_path=result.get("output_path"),
            error_path=result.get("error_path"),
        )
    output_path = result.get("output_path") or "no output file yet"
    error_path = result.get("error_path") or "no error file"
    state.message = f"Downloaded batch {batch_id}: output={output_path} error={error_path}"
    return result


def run_step1_batch_for_state(state: WorkflowState) -> dict:
    try:
        batch = refresh_latest_batch_for_state(state)
        if batch.get("status") == "completed":
            return download_latest_batch_for_state(state)
        return batch
    except FileNotFoundError:
        return submit_latest_batch_for_state(state)


def selected_step_keys(state: WorkflowState) -> list[str]:
    return [step.key for step in state.steps if step.enabled and step.selected]


def selected_step_labels(state: WorkflowState) -> str:
    labels = [step.label for step in state.steps if step.enabled and step.selected]
    return ", ".join(labels) if labels else "none"


def next_step1_model(current_model: str) -> str:
    if current_model not in STEP1_MODEL_CHOICES:
        return STEP1_MODEL_CHOICES[0]
    index = STEP1_MODEL_CHOICES.index(current_model)
    return STEP1_MODEL_CHOICES[(index + 1) % len(STEP1_MODEL_CHOICES)]


def is_terminal_batch_status(status: str | None) -> bool:
    return str(status or "").lower() in BATCH_TERMINAL_STATUSES


def batch_output_path_exists(row) -> bool:
    output_path = row["output_path"]
    return bool(output_path and Path(output_path).exists())


def batch_needs_output_download(row) -> bool:
    return str(row["status"] or "").lower() == "completed" and not batch_output_path_exists(row)


def known_batches_need_sync(state: WorkflowState) -> bool:
    with connect_state_db(state.state_db) as conn:
        rows = list_batch_jobs(conn, limit=100)
    return any(not is_terminal_batch_status(row["status"]) or batch_needs_output_download(row) for row in rows)


def sync_openai_batches_for_state(state: WorkflowState, *, limit: int = 100) -> dict[str, int]:
    with connect_state_db(state.state_db) as conn:
        rows = list_batch_jobs(conn, limit=limit)

    refreshed = 0
    downloaded = 0
    active = 0
    skipped = 0

    for row in rows:
        batch_id = str(row["batch_id"])
        status = str(row["status"] or "").lower()
        batch: dict | None = None

        if not is_terminal_batch_status(status):
            batch = get_batch(batch_id)
            refreshed += 1
            status = str(batch.get("status") or "").lower()
            with connect_state_db(state.state_db) as conn:
                upsert_batch_job(conn, batch)
            if not is_terminal_batch_status(status):
                active += 1

        if status == "completed":
            output_path = row["output_path"]
            output_exists = bool(output_path and Path(output_path).exists())
            if output_exists:
                skipped += 1
                continue
            if batch is None:
                batch = get_batch(batch_id)
                refreshed += 1
                with connect_state_db(state.state_db) as conn:
                    upsert_batch_job(conn, batch)
            if batch.get("output_file_id") or batch.get("error_file_id"):
                result = download_batch_outputs(batch_id, state.output_dir)
                downloaded += 1
                with connect_state_db(state.state_db) as conn:
                    upsert_batch_job(
                        conn,
                        result["batch"],
                        output_path=result.get("output_path"),
                        error_path=result.get("error_path"),
                    )
            else:
                skipped += 1

    state.message = f"Batch sync: refreshed={refreshed} active={active} downloaded={downloaded}"
    return {"refreshed": refreshed, "active": active, "downloaded": downloaded, "skipped": skipped}


def run_textual_app(state: WorkflowState) -> None:
    try:
        from textual.app import App, ComposeResult
        from textual.binding import Binding
        from textual.containers import Horizontal, Vertical
        from textual.widgets import DataTable, Footer, Header, Input, Label, ProgressBar, RichLog, Static
        from rich.text import Text
    except ImportError as exc:
        raise SystemExit(
            "Textual is required for interactive mode. Install it with `python -m pip install textual` "
            "or run the CLI path with `--no-tui`."
        ) from exc

    class CadWorkflowDashboard(App[None]):
        CSS = """
        Screen {
            layout: vertical;
        }

        #overview {
            height: 5;
        }

        .summary-panel {
            width: 1fr;
            height: 100%;
            padding: 0 1;
            border: solid $panel;
        }

        #workflow-summary {
            width: 1fr;
        }

        #batch-summary {
            width: 1fr;
        }

        #main {
            height: 1fr;
        }

        #steps-table {
            width: 3fr;
            height: 100%;
            border: solid $panel;
        }

        #run-table {
            width: 5fr;
            height: 100%;
            border: solid $panel;
        }

        #activity {
            height: 5;
        }

        #log {
            width: 2fr;
            height: 100%;
            border: solid $panel;
        }

        #status {
            width: 1fr;
            border: solid $panel;
            padding: 1 2;
        }

        #progress {
            height: 2;
            padding: 0 2;
        }

        #input-row {
            height: auto;
            padding: 0 2 1 2;
        }

        #search-input {
            width: 1fr;
        }
        """

        BINDINGS = [
            Binding("q", "quit", "Quit"),
            Binding("enter", "run_selected", "Run", priority=True),
            Binding("space", "toggle_step", "Toggle", priority=True),
            Binding("up", "step_up", "Up", show=False, priority=True),
            Binding("down", "step_down", "Down", show=False, priority=True),
            Binding("s", "step0", "Step0", show=False),
            Binding("f", "force_step0", "Force Step0", show=False),
            Binding("j", "step1_jsonl", "JSONL", show=False),
            Binding("b", "both", "Both", show=False),
            Binding("u", "submit_batch", "Submit"),
            Binding("c", "batch_monitor", "Monitor"),
            Binding("d", "download_batch", "Output"),
            Binding("r", "rescan", "Rescan", show=False),
            Binding("n", "next_page", "Next", show=False),
            Binding("p", "prev_page", "Prev", show=False),
            Binding("t", "cycle_filter", "Filter", show=False),
            Binding("/", "edit_search", "Search", show=False),
            Binding("o", "edit_ollama_model", "Ollama", show=False),
            Binding("m", "toggle_step1_model", "Step1 model", show=False),
        ]

        def __init__(self, workflow_state: WorkflowState) -> None:
            super().__init__()
            self.workflow_state = workflow_state
            self.busy = False
            self.editing_field: str | None = None
            self.progress_total = 0
            self.progress_done = 0
            self.syncing_step_cursor = False
            self.batch_monitor_task: asyncio.Task[None] | None = None
            self.batch_monitor_enabled = False

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            with Horizontal(id="overview"):
                yield Static(id="workflow-summary", classes="summary-panel", markup=False)
                yield Static(id="batch-summary", classes="summary-panel", markup=False)
            with Horizontal(id="main"):
                yield DataTable(id="steps-table")
                yield DataTable(id="run-table")
            with Horizontal(id="activity"):
                yield RichLog(id="log", markup=False, wrap=True)
                yield Static(id="status", markup=False)
            yield ProgressBar(total=1, id="progress")
            with Horizontal(id="input-row"):
                yield Label("", id="input-label", markup=False)
                yield Input(id="search-input")
            yield Footer()

        def on_mount(self) -> None:
            self.title = "Data -> CAD Workflow"
            self.query_one("#input-row").display = False
            steps = self.query_one("#steps-table", DataTable)
            steps.cursor_type = "row"
            steps.zebra_stripes = True
            steps.add_columns("", "Run", "Step", "State")
            table = self.query_one("#run-table", DataTable)
            table.cursor_type = "row"
            table.zebra_stripes = True
            table.add_columns("OpenAI run", "Status", "Req", "Output")
            self.rescan_index()
            self.refresh_view()
            if known_batches_need_sync(self.workflow_state) and os.environ.get("OPENAI_API_KEY"):
                self.start_batch_monitor()

        def on_unmount(self) -> None:
            self.stop_batch_monitor("Batch monitor stopped")

        def rescan_index(self) -> None:
            index_downloads(self.workflow_state)

        def filtered_total(self) -> int:
            with connect_state_db(self.workflow_state.state_db) as conn:
                return count_filtered(
                    conn,
                    status=self.workflow_state.status_filter,
                    search=self.workflow_state.search,
                )

        def refresh_view(self) -> None:
            with connect_state_db(self.workflow_state.state_db) as conn:
                counts = count_by_status(conn)
                total = count_filtered(
                    conn,
                    status=self.workflow_state.status_filter,
                    search=self.workflow_state.search,
                )
                max_page = max(0, (total - 1) // self.workflow_state.page_size) if total else 0
                self.workflow_state.page = min(self.workflow_state.page, max_page)
                batch_row = latest_batch_job(conn)
                batch_rows = list_batch_jobs(conn, limit=min(self.workflow_state.page_size, 100))

            batch_summary = "Batch\nnone"
            if batch_row:
                output_path = Path(batch_row["output_path"]).name if batch_row["output_path"] else "-"
                batch_summary = (
                    f"Batch\n{batch_row['status']}  {batch_row['request_completed']}/{batch_row['request_total']} "
                    f"failed={batch_row['request_failed']}\n"
                    f"{batch_row['batch_id']}\n"
                    f"output: {output_path}"
                )

            workflow_summary = (
                f"Step 0  total:{counts['total']} done:{counts['done']} err:{counts['error']}\n"
                f"pending:{counts['pending']} running:{counts['running']} skipped:{counts['skipped']}\n"
                f"Selected: {selected_step_labels(self.workflow_state)}\n"
                f"View  {self.workflow_state.status_filter}  search:{self.workflow_state.search or '*'}  "
                f"page:{self.workflow_state.page + 1}/{max_page + 1} rows:{total}"
            )
            self.query_one("#workflow-summary", Static).update(workflow_summary)
            self.query_one("#batch-summary", Static).update(batch_summary)

            steps_table = self.query_one("#steps-table", DataTable)
            self.syncing_step_cursor = True
            steps_table.clear()
            for index, step in enumerate(self.workflow_state.steps):
                marker = ">" if index == self.workflow_state.step_cursor else " "
                run_state = "RUN" if step.selected else "SKIP"
                cells = [marker, run_state, step.label, step.status]
                if not step.enabled:
                    run_state = "HOLD"
                    cells = [
                        Text(" ", style="dim"),
                        Text(run_state, style="dim"),
                        Text(step.label, style="dim"),
                        Text(step.status, style="dim"),
                    ]
                steps_table.add_row(
                    *cells,
                    key=step.key,
                )
            steps_table.move_cursor(row=self.workflow_state.step_cursor, column=0, animate=False, scroll=True)
            self.syncing_step_cursor = False

            table = self.query_one("#run-table", DataTable)
            table.clear()
            for row in batch_rows:
                output_state = self.batch_output_state(row)
                table.add_row(
                    self.compact_text(row["batch_id"], 18),
                    row["status"],
                    self.batch_progress_text(row),
                    output_state,
                    key=row["batch_id"],
                )

            status = self.workflow_state.message
            monitor_state = "on" if self.batch_monitor_enabled else "off"
            status += (
                "\nUp/Down choose step  Space toggles  Enter runs selected"
                f"\nBatch monitor: {monitor_state}  c toggles"
                f"\nOllama: {self.workflow_state.ollama_model}"
                f"\nStep1: {self.workflow_state.step1_model}  m toggles 5.4-mini/5.5"
            )
            if self.busy:
                status += "\nWorking..."
            if self.workflow_state.errors:
                status += f"\nLast error: {self.workflow_state.errors[-1]}"
            self.query_one("#status", Static).update(status)

        @staticmethod
        def compact_text(value: str, max_len: int) -> str:
            if len(value) <= max_len:
                return value
            return value[: max_len - 3] + "..."

        @staticmethod
        def compact_time(value: str | None) -> str:
            if not value:
                return "-"
            return str(value).replace("T", " ").replace("Z", "")

        @staticmethod
        def batch_progress_text(row) -> str:
            total = int(row["request_total"] or 0)
            completed = int(row["request_completed"] or 0)
            failed = int(row["request_failed"] or 0)
            if not total:
                return "0/0"
            suffix = f" f{failed}" if failed else ""
            return f"{completed}/{total}{suffix}"

        @staticmethod
        def batch_output_state(row) -> str:
            if row["output_path"] and Path(row["output_path"]).exists():
                return "saved"
            if row["output_path"]:
                return "missing"
            if row["output_file_id"]:
                return "ready"
            if row["error_path"] and Path(row["error_path"]).exists():
                return "err saved"
            if row["error_path"]:
                return "err missing"
            if row["error_file_id"]:
                return "err ready"
            return "-"

        def _handle_event(self, event: Step0Event) -> None:
            if event.status == "running":
                return
            if event.status in {"done", "skipped", "error"}:
                self.progress_done += 1
                self.query_one("#progress", ProgressBar).update(progress=self.progress_done)
            detail = event.selected_drawing or event.error or event.message
            self.query_one("#log", RichLog).write(f"{event.component_id}: {event.status} {detail}", scroll_end=True)
            self.refresh_view()

        def log_event(self, event: Step0Event) -> None:
            self.call_from_thread(self._handle_event, event)

        async def run_work(self, label: str, func, *args, **kwargs) -> None:
            if self.busy:
                return
            self.busy = True
            self.workflow_state.message = label
            self.refresh_view()
            try:
                await asyncio.to_thread(func, *args, **kwargs)
            except Exception as exc:
                message = f"{label} failed: {exc}"
                self.workflow_state.message = message
                self.workflow_state.errors.append(str(exc))
                self.query_one("#log", RichLog).write(message, scroll_end=True)
            finally:
                self.busy = False
                self.refresh_view()

        async def action_step0(self) -> None:
            components = step0_scope(self.workflow_state, force=False)
            self.progress_total = max(1, len(components))
            self.progress_done = 0
            self.query_one("#progress", ProgressBar).update(total=self.progress_total, progress=0)
            await self.run_work(
                "Running Step 0",
                run_step0_for_state,
                self.workflow_state,
                force=False,
                on_event=self.log_event,
            )

        async def action_force_step0(self) -> None:
            components = step0_scope(self.workflow_state, force=True)
            self.progress_total = max(1, len(components))
            self.progress_done = 0
            self.query_one("#progress", ProgressBar).update(total=self.progress_total, progress=0)
            await self.run_work(
                "Force running Step 0",
                run_step0_for_state,
                self.workflow_state,
                force=True,
                on_event=self.log_event,
            )

        async def action_step1_jsonl(self) -> None:
            await self.run_work("Creating Step 1 JSONL", run_step1_jsonl_for_state, self.workflow_state)

        async def action_submit_batch(self) -> None:
            await self.run_work("Submitting latest Step 1 batch JSONL", submit_latest_batch_for_state, self.workflow_state)
            if known_batches_need_sync(self.workflow_state):
                self.start_batch_monitor()

        async def action_batch_status(self) -> None:
            await self.run_work("Refreshing latest batch status", refresh_latest_batch_for_state, self.workflow_state)

        async def action_batch_monitor(self) -> None:
            if self.batch_monitor_enabled:
                self.stop_batch_monitor("Batch monitor stopped")
                self.refresh_view()
                return
            await self.run_work("Syncing OpenAI batch status", sync_openai_batches_for_state, self.workflow_state)
            if known_batches_need_sync(self.workflow_state):
                self.start_batch_monitor()
            else:
                self.workflow_state.message = "Batch monitor idle: all known runs are synced"
                self.refresh_view()

        async def action_download_batch(self) -> None:
            await self.run_work("Downloading latest batch output", download_latest_batch_for_state, self.workflow_state)

        async def action_run_selected(self) -> None:
            selected = selected_step_keys(self.workflow_state)
            if not selected:
                self.workflow_state.message = "No runnable steps selected"
                self.refresh_view()
                return
            for key in selected:
                if key == "step0":
                    await self.action_step0()
                elif key == "step1_jsonl":
                    await self.action_step1_jsonl()
                elif key == "step1_submit":
                    await self.run_work("Running Step 1 batch action", run_step1_batch_for_state, self.workflow_state)
                    if known_batches_need_sync(self.workflow_state):
                        self.start_batch_monitor()
            self.workflow_state.message = f"Finished selected steps: {selected_step_labels(self.workflow_state)}"
            self.refresh_view()

        def action_step_up(self) -> None:
            self.set_step_cursor(self.next_enabled_step_index(-1))
            self.refresh_view()

        def action_step_down(self) -> None:
            self.set_step_cursor(self.next_enabled_step_index(1))
            self.refresh_view()

        def next_enabled_step_index(self, direction: int) -> int:
            index = self.workflow_state.step_cursor
            while 0 <= index + direction < len(self.workflow_state.steps):
                index += direction
                if self.workflow_state.steps[index].enabled:
                    return index
            return self.workflow_state.step_cursor

        def set_step_cursor(self, row: int) -> None:
            self.workflow_state.step_cursor = row
            self.syncing_step_cursor = True
            self.query_one("#steps-table", DataTable).move_cursor(row=row, column=0, animate=False, scroll=True)
            self.syncing_step_cursor = False

        def action_toggle_step(self) -> None:
            step = self.workflow_state.steps[self.workflow_state.step_cursor]
            if not step.enabled:
                self.workflow_state.message = f"{step.label} is a placeholder"
            else:
                step.selected = not step.selected
                state = "selected" if step.selected else "skipped"
                self.workflow_state.message = f"{step.label}: {state}"
            self.refresh_view()

        async def action_both(self) -> None:
            await self.action_step0()
            if not self.workflow_state.errors:
                await self.action_step1_jsonl()

        def action_rescan(self) -> None:
            self.rescan_index()
            self.workflow_state.message = "Rescanned downloads"
            self.refresh_view()

        def action_next_page(self) -> None:
            total = self.filtered_total()
            max_page = max(0, (total - 1) // self.workflow_state.page_size) if total else 0
            self.workflow_state.page = min(self.workflow_state.page + 1, max_page)
            self.refresh_view()

        def action_prev_page(self) -> None:
            self.workflow_state.page = max(0, self.workflow_state.page - 1)
            self.refresh_view()

        def action_cycle_filter(self) -> None:
            index = STATUS_FILTERS.index(self.workflow_state.status_filter)
            self.workflow_state.status_filter = STATUS_FILTERS[(index + 1) % len(STATUS_FILTERS)]
            self.workflow_state.page = 0
            self.workflow_state.message = f"Filter set to {self.workflow_state.status_filter}"
            self.refresh_view()

        def action_edit_search(self) -> None:
            self.show_input("search", "Search")

        def action_edit_ollama_model(self) -> None:
            self.show_input("ollama", "Ollama model")

        def action_toggle_step1_model(self) -> None:
            self.workflow_state.step1_model = next_step1_model(self.workflow_state.step1_model)
            self.workflow_state.message = (
                f"Step 1 model set to {self.workflow_state.step1_model} "
                "with reasoning effort medium"
            )
            self.refresh_view()

        def start_batch_monitor(self) -> None:
            if self.batch_monitor_task and not self.batch_monitor_task.done():
                self.batch_monitor_enabled = True
                return
            self.batch_monitor_enabled = True
            self.workflow_state.message = "Batch monitor started"
            self.batch_monitor_task = asyncio.create_task(self.batch_monitor_loop())
            self.refresh_view()

        def stop_batch_monitor(self, message: str | None = None) -> None:
            self.batch_monitor_enabled = False
            if self.batch_monitor_task and not self.batch_monitor_task.done():
                self.batch_monitor_task.cancel()
            if message:
                self.workflow_state.message = message

        async def batch_monitor_loop(self) -> None:
            try:
                while self.batch_monitor_enabled:
                    try:
                        await asyncio.to_thread(sync_openai_batches_for_state, self.workflow_state)
                        self.query_one("#log", RichLog).write(self.workflow_state.message, scroll_end=True)
                    except Exception as exc:
                        message = f"Batch monitor error: {exc}"
                        self.workflow_state.message = message
                        self.workflow_state.errors.append(str(exc))
                        self.query_one("#log", RichLog).write(message, scroll_end=True)
                    self.refresh_view()
                    if not known_batches_need_sync(self.workflow_state):
                        self.batch_monitor_enabled = False
                        self.workflow_state.message = "Batch monitor idle: all known runs are synced"
                        self.refresh_view()
                        break
                    await asyncio.sleep(5)
            except asyncio.CancelledError:
                pass
            finally:
                self.batch_monitor_task = None

        def show_input(self, field: str, label_text: str) -> None:
            if self.busy:
                return
            self.editing_field = field
            row = self.query_one("#input-row")
            label = self.query_one("#input-label", Label)
            input_widget = self.query_one("#search-input", Input)
            label.update(label_text)
            if field == "search":
                input_widget.value = self.workflow_state.search
            elif field == "ollama":
                input_widget.value = self.workflow_state.ollama_model
            else:
                input_widget.value = self.workflow_state.step1_model
            row.display = True
            input_widget.focus()

        def on_input_submitted(self, event: Input.Submitted) -> None:
            value = event.value.strip()
            if self.editing_field == "search":
                self.workflow_state.search = value
                self.workflow_state.page = 0
                self.workflow_state.message = f"Search set to {value or '*'}"
            elif self.editing_field == "ollama" and value:
                self.workflow_state.ollama_model = value
                self.workflow_state.message = f"Ollama model set to {value}"
            elif self.editing_field == "step1" and value:
                self.workflow_state.step1_model = value
                self.workflow_state.message = f"Step1 model set to {value}"
            self.editing_field = None
            self.query_one("#input-row").display = False
            self.refresh_view()

        def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
            if event.data_table.id != "steps-table" or self.syncing_step_cursor:
                return
            key = str(event.row_key.value)
            for index, step in enumerate(self.workflow_state.steps):
                if step.key != key:
                    continue
                if not step.enabled:
                    self.set_step_cursor(self.workflow_state.step_cursor)
                    return
                if index != self.workflow_state.step_cursor:
                    self.workflow_state.step_cursor = index
                    self.refresh_step_cursor_marker()
                    return

        def refresh_step_cursor_marker(self) -> None:
            steps_table = self.query_one("#steps-table", DataTable)
            self.syncing_step_cursor = True
            for index, step in enumerate(self.workflow_state.steps):
                if not step.enabled:
                    continue
                marker = ">" if index == self.workflow_state.step_cursor else " "
                marker_column = next(iter(steps_table.columns))
                steps_table.update_cell(step.key, marker_column, marker)
            self.syncing_step_cursor = False

    CadWorkflowDashboard(state).run()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Textual workflow dashboard for Step 0 and Step 1 batch JSONL.")
    parser.add_argument("--downloads-dir", default="downloads", type=Path)
    parser.add_argument("--output-dir", default="output", type=Path)
    parser.add_argument("--state-db", default=DEFAULT_STATE_DB, type=Path)
    parser.add_argument("--ollama-model", default=os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL))
    parser.add_argument("--step1-model", default=os.environ.get("OPENAI_STEP1_MODEL", DEFAULT_MODEL))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--no-tui", action="store_true", help="Run from CLI instead of Textual.")
    parser.add_argument("--step0", action="store_true", help="Select and copy main engineering drawings.")
    parser.add_argument("--force", action="store_true", help="Force Step 0 rerun.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--jsonl", action="store_true", help="Create Step 1 batch JSONL from completed Step 0 rows.")
    parser.add_argument("--both", action="store_true", help="Run step0, then create Step 1 batch JSONL.")
    parser.add_argument("--submit-batch", action="store_true", help="Submit the latest Step 1 JSONL as an OpenAI Batch.")
    parser.add_argument("--batch-status", action="store_true", help="Refresh the latest or specified OpenAI Batch status.")
    parser.add_argument("--download-batch", action="store_true", help="Download output/error files for the latest or specified batch.")
    parser.add_argument("--batch-id", help="Batch id for --batch-status or --download-batch.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    state = WorkflowState(
        downloads_dir=args.downloads_dir,
        output_dir=args.output_dir,
        state_db=args.state_db,
        ollama_model=args.ollama_model,
        step1_model=args.step1_model,
        limit=args.limit,
        page_size=args.page_size,
    )

    if args.no_tui:
        index_downloads(state)
        if args.both or args.step0:
            components = step0_scope(state, force=args.force)

            def print_event(event: Step0Event) -> None:
                if event.status != "running":
                    detail = event.selected_drawing or event.error or event.message
                    print(f"{event.component_id}: {event.status} {detail}", flush=True)

            completed, errors = run_step0_batch(
                components,
                state.ollama_model,
                state_db=state.state_db,
                force=args.force,
                workers=args.workers,
                on_event=print_event,
            )
            print(f"Step0 complete: completed={len(completed)} errors={len(errors)}")
            for error in errors:
                print(f"ERROR: {error.component_id}: {error.message}", file=sys.stderr)
            state.errors = [event.message for event in errors]
        if args.both or args.jsonl:
            output_path = run_step1_jsonl_for_state(state)
            print(output_path)
        if args.submit_batch:
            batch = submit_latest_batch_for_state(state)
            print(json.dumps(batch, indent=2))
        if args.batch_id:
            with connect_state_db(state.state_db) as conn:
                if args.batch_status:
                    upsert_batch_job(conn, {"id": args.batch_id, "status": "unknown"})
                elif args.download_batch:
                    upsert_batch_job(conn, {"id": args.batch_id, "status": "unknown"})
        if args.batch_status:
            batch = refresh_latest_batch_for_state(state)
            print(json.dumps(batch, indent=2))
        if args.download_batch:
            result = download_latest_batch_for_state(state)
            print(json.dumps(result, indent=2))
        if not (args.both or args.step0 or args.jsonl or args.submit_batch or args.batch_status or args.download_batch):
            print("Use --step0, --jsonl, --submit-batch, --batch-status, --download-batch, or --both with --no-tui.", file=sys.stderr)
            return 2
        return 1 if state.errors else 0

    run_textual_app(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
