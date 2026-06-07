from __future__ import annotations

import sqlite3
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable


DEFAULT_STATE_DB = Path("output/workflow_state.sqlite3")
STATUSES = ("pending", "running", "done", "skipped", "error")


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def connect_state_db(path: str | Path = DEFAULT_STATE_DB) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    init_state_db(conn)
    return conn


def init_state_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS components (
            component_id TEXT PRIMARY KEY,
            component_path TEXT NOT NULL,
            drawing_count INTEGER NOT NULL DEFAULT 0,
            step0_status TEXT NOT NULL DEFAULT 'pending',
            selected_drawing TEXT,
            confidence REAL,
            method TEXT,
            error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_components_status ON components(step0_status)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_components_updated ON components(updated_at)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS batch_jobs (
            batch_id TEXT PRIMARY KEY,
            jsonl_path TEXT,
            endpoint TEXT,
            status TEXT NOT NULL,
            input_file_id TEXT,
            output_file_id TEXT,
            error_file_id TEXT,
            request_total INTEGER NOT NULL DEFAULT 0,
            request_completed INTEGER NOT NULL DEFAULT 0,
            request_failed INTEGER NOT NULL DEFAULT 0,
            output_path TEXT,
            error_path TEXT,
            error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            completed_at TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_batch_jobs_updated ON batch_jobs(updated_at)")
    conn.commit()


def image_count(component_dir: Path) -> int:
    drawings_dir = component_dir / "drawings"
    if not drawings_dir.exists():
        return 0
    suffixes = {".gif", ".png", ".jpg", ".jpeg", ".webp"}
    return sum(
        1
        for path in drawings_dir.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes and path.stat().st_size > 0
    )


def component_dirs(downloads_dir: str | Path) -> list[Path]:
    downloads_dir = Path(downloads_dir)
    if not downloads_dir.exists():
        return []
    return sorted(path for path in downloads_dir.iterdir() if path.is_dir())


def selection_manifest_exists(component_dir: str | Path) -> bool:
    return (Path(component_dir) / "main_engineering_drawing" / "selection.json").exists()


def selection_manifest_summary(component_dir: str | Path) -> tuple[str | None, float | None, str | None]:
    manifest_path = Path(component_dir) / "main_engineering_drawing" / "selection.json"
    if not manifest_path.exists():
        return None, None, None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None, None, None
    selection = manifest.get("selection", {})
    copied_path = manifest.get("copied_path")
    selected = Path(copied_path).name if copied_path else selection.get("filename")
    return selected, selection.get("confidence"), selection.get("_method")


def scan_downloads(
    conn: sqlite3.Connection,
    downloads_dir: str | Path,
    limit: int | None = None,
    component_paths: Iterable[str | Path] | None = None,
) -> list[Path]:
    components = [Path(path) for path in component_paths] if component_paths else component_dirs(downloads_dir)
    if limit is not None:
        components = components[:limit]

    now = utc_now()
    for component in components:
        existing = conn.execute(
            "SELECT step0_status FROM components WHERE component_id = ?",
            (component.name,),
        ).fetchone()
        status = existing["step0_status"] if existing else "pending"
        if selection_manifest_exists(component) and status in {"pending", "error", "running"}:
            status = "done"
        selected_drawing, confidence, method = selection_manifest_summary(component)
        conn.execute(
            """
            INSERT INTO components (
                component_id, component_path, drawing_count, step0_status,
                selected_drawing, confidence, method, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(component_id) DO UPDATE SET
                component_path = excluded.component_path,
                drawing_count = excluded.drawing_count,
                step0_status = ?,
                selected_drawing = COALESCE(excluded.selected_drawing, components.selected_drawing),
                confidence = COALESCE(excluded.confidence, components.confidence),
                method = COALESCE(excluded.method, components.method),
                updated_at = excluded.updated_at
            """,
            (
                component.name,
                str(component),
                image_count(component),
                status,
                selected_drawing,
                confidence,
                method,
                now,
                now,
                status,
            ),
        )
    conn.commit()
    return components


def update_component_status(
    conn: sqlite3.Connection,
    component_id: str,
    status: str,
    *,
    selected_drawing: str | None = None,
    confidence: float | None = None,
    method: str | None = None,
    error: str | None = None,
) -> None:
    if status not in STATUSES:
        raise ValueError(f"Unknown component status: {status}")
    conn.execute(
        """
        UPDATE components
        SET step0_status = ?,
            selected_drawing = ?,
            confidence = ?,
            method = ?,
            error = ?,
            updated_at = ?
        WHERE component_id = ?
        """,
        (status, selected_drawing, confidence, method, error, utc_now(), component_id),
    )
    conn.commit()


def count_by_status(conn: sqlite3.Connection) -> dict[str, int]:
    counts = {status: 0 for status in STATUSES}
    rows = conn.execute(
        "SELECT step0_status, COUNT(*) AS count FROM components GROUP BY step0_status"
    ).fetchall()
    for row in rows:
        counts[row["step0_status"]] = row["count"]
    counts["total"] = sum(counts.values())
    return counts


def list_components(
    conn: sqlite3.Connection,
    *,
    status: str = "all",
    search: str = "",
    limit: int = 100,
    offset: int = 0,
) -> list[sqlite3.Row]:
    clauses = []
    params: list[object] = []
    if status != "all":
        clauses.append("step0_status = ?")
        params.append(status)
    if search:
        clauses.append("component_id LIKE ?")
        params.append(f"%{search}%")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.extend([limit, offset])
    return conn.execute(
        f"""
        SELECT component_id, component_path, drawing_count, step0_status,
               selected_drawing, confidence, method, error, updated_at
        FROM components
        {where}
        ORDER BY component_id
        LIMIT ? OFFSET ?
        """,
        params,
    ).fetchall()


def count_filtered(conn: sqlite3.Connection, *, status: str = "all", search: str = "") -> int:
    clauses = []
    params: list[object] = []
    if status != "all":
        clauses.append("step0_status = ?")
        params.append(status)
    if search:
        clauses.append("component_id LIKE ?")
        params.append(f"%{search}%")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    row = conn.execute(f"SELECT COUNT(*) AS count FROM components {where}", params).fetchone()
    return int(row["count"])


def component_paths_for_status(
    conn: sqlite3.Connection,
    *,
    status: str = "all",
    limit: int | None = None,
) -> list[Path]:
    clauses = []
    params: list[object] = []
    if status == "done":
        clauses.append("step0_status IN ('done', 'skipped')")
    elif status != "all":
        clauses.append("step0_status = ?")
        params.append(status)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"SELECT component_path FROM components {where} ORDER BY component_id"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    return [Path(row["component_path"]) for row in conn.execute(sql, params).fetchall()]


def component_paths_for_filter(
    conn: sqlite3.Connection,
    *,
    status: str = "all",
    search: str = "",
    limit: int | None = None,
) -> list[Path]:
    clauses = []
    params: list[object] = []
    if status == "done":
        clauses.append("step0_status IN ('done', 'skipped')")
    elif status != "all":
        clauses.append("step0_status = ?")
        params.append(status)
    if search:
        clauses.append("component_id LIKE ?")
        params.append(f"%{search}%")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"SELECT component_path FROM components {where} ORDER BY component_id"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    return [Path(row["component_path"]) for row in conn.execute(sql, params).fetchall()]


def completed_component_dirs(
    conn: sqlite3.Connection,
    *,
    status: str = "done",
    limit: int | None = None,
) -> list[Path]:
    return component_paths_for_status(conn, status=status, limit=limit)


def batch_request_counts(batch: dict) -> tuple[int, int, int]:
    counts = batch.get("request_counts") or {}
    return (
        int(counts.get("total") or 0),
        int(counts.get("completed") or 0),
        int(counts.get("failed") or 0),
    )


def upsert_batch_job(
    conn: sqlite3.Connection,
    batch: dict,
    *,
    jsonl_path: str | Path | None = None,
    output_path: str | Path | None = None,
    error_path: str | Path | None = None,
    error: str | None = None,
) -> None:
    batch_id = str(batch.get("id") or "")
    if not batch_id:
        raise ValueError("Batch object does not include an id")
    request_total, request_completed, request_failed = batch_request_counts(batch)
    now = utc_now()
    existing = conn.execute(
        "SELECT jsonl_path, output_path, error_path, created_at FROM batch_jobs WHERE batch_id = ?",
        (batch_id,),
    ).fetchone()
    created_at = existing["created_at"] if existing else now
    persisted_jsonl = str(jsonl_path) if jsonl_path is not None else (existing["jsonl_path"] if existing else None)
    persisted_output = str(output_path) if output_path is not None else (existing["output_path"] if existing else None)
    persisted_error_path = str(error_path) if error_path is not None else (existing["error_path"] if existing else None)
    completed_at = batch.get("completed_at") or batch.get("failed_at") or batch.get("cancelled_at") or batch.get("expired_at")
    completed_at_text = str(completed_at) if completed_at is not None else None
    conn.execute(
        """
        INSERT INTO batch_jobs (
            batch_id, jsonl_path, endpoint, status, input_file_id, output_file_id, error_file_id,
            request_total, request_completed, request_failed, output_path, error_path, error,
            created_at, updated_at, completed_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(batch_id) DO UPDATE SET
            jsonl_path = excluded.jsonl_path,
            endpoint = excluded.endpoint,
            status = excluded.status,
            input_file_id = excluded.input_file_id,
            output_file_id = excluded.output_file_id,
            error_file_id = excluded.error_file_id,
            request_total = excluded.request_total,
            request_completed = excluded.request_completed,
            request_failed = excluded.request_failed,
            output_path = excluded.output_path,
            error_path = excluded.error_path,
            error = excluded.error,
            updated_at = excluded.updated_at,
            completed_at = excluded.completed_at
        """,
        (
            batch_id,
            persisted_jsonl,
            batch.get("endpoint"),
            str(batch.get("status") or "unknown"),
            batch.get("input_file_id"),
            batch.get("output_file_id"),
            batch.get("error_file_id"),
            request_total,
            request_completed,
            request_failed,
            persisted_output,
            persisted_error_path,
            error,
            created_at,
            now,
            completed_at_text,
        ),
    )
    conn.commit()


def latest_batch_job(conn: sqlite3.Connection) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT *
        FROM batch_jobs
        ORDER BY updated_at DESC, created_at DESC
        LIMIT 1
        """
    ).fetchone()


def get_batch_job(conn: sqlite3.Connection, batch_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM batch_jobs WHERE batch_id = ?", (batch_id,)).fetchone()


def list_batch_jobs(conn: sqlite3.Connection, limit: int = 10) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT *
        FROM batch_jobs
        ORDER BY updated_at DESC, created_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
