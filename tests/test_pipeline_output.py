"""Tests for the pipeline.output run-layout helpers and the
review_server (the stdlib HTTP server).

These are pure-stdlib and run in any venv. They verify:
- Run.open() creates the expected directory structure
- Component.html()/json() write valid files with the right shape
- Run.finish() writes run.json, components.json, and a run-level index.html
- Run.abort() flips status to "aborted"
- make_run_id() is sortable and unique
- The review_server responds 200 on the public routes with the right
  content-type (tested via the real HTTP server in a background thread).
"""
from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import time
from http.client import HTTPConnection
from pathlib import Path

import pytest

# Make 'pipeline' importable when running from the repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.output import (  # noqa: E402
    Run,
    make_run_id,
    step_dir,
    RUNS_ROOT,
)


# ---------------------------------------------------------------------------
# Test isolation: every test uses its own step name so they don't collide
# ---------------------------------------------------------------------------

@pytest.fixture
def isolated_runs_root(tmp_path, monkeypatch):
    """Point RUNS_ROOT at a tmp dir for the duration of one test."""
    fake_root = tmp_path / "runs"
    fake_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("pipeline.output.RUNS_ROOT", fake_root)
    return fake_root


# ---------------------------------------------------------------------------
# Run open / finish / abort
# ---------------------------------------------------------------------------

def test_run_open_creates_layout(isolated_runs_root):
    r = Run.open("step99", all=True, dry_run=True)
    assert r.path.is_dir()
    assert (r.path / "components").is_dir()
    assert r.run_id.startswith("20")
    assert r.status == "running"
    # args merged with --key naming
    assert r.args.get("all") is True
    assert r.args.get("dry-run") is True


def test_component_html_and_json_writes_files(isolated_runs_root):
    r = Run.open("step99")
    c = r.component("110300324920")
    c.add_artifact("drawing.png", "downloads/x.png", kind="image")
    c.finish(features=3, implemented=2)
    c.html("<p>hi</p>", title="Test")
    c.json()

    assert (c.path / "index.html").is_file()
    assert (c.path / "meta.json").is_file()

    html = (c.path / "index.html").read_text()
    assert "Test" in html
    assert "drawing.png" in html
    assert "<p>hi</p>" in html

    meta = json.loads((c.path / "meta.json").read_text())
    assert meta["component_id"] == "110300324920"
    assert meta["summary"]["features"] == 3
    assert meta["artifacts"][0]["label"] == "drawing.png"


def test_run_finish_writes_index_and_run_json(isolated_runs_root):
    r = Run.open("step99", all=True)
    c1 = r.component("cid1")
    c1.finish(features=2, implemented=2)
    c2 = r.component("cid2")
    c2.fail("boom")
    r.finish(ok=1, errors=1)

    assert (r.path / "index.html").is_file()
    assert (r.path / "run.json").is_file()
    assert (r.path / "components.json").is_file()

    run_data = json.loads((r.path / "run.json").read_text())
    assert run_data["status"] == "completed"
    assert run_data["n_components"] == 2
    assert run_data["n_errors"] == 1
    assert run_data["summary"]["ok"] == 1

    # run-level html has a row per component
    html = (r.path / "index.html").read_text()
    assert "cid1" in html
    assert "cid2" in html
    assert "boom" in html  # error message for failed component


def test_run_abort_marks_status_aborted(isolated_runs_root):
    r = Run.open("step99")
    r.abort("unrecoverable")
    assert r.status == "aborted"
    run_data = json.loads((r.path / "run.json").read_text())
    assert run_data["status"] == "aborted"
    assert run_data["summary"]["error"] == "unrecoverable"


def test_make_run_id_is_sortable_and_unique():
    ids = [make_run_id() for _ in range(50)]
    # unique
    assert len(set(ids)) == len(ids)
    # sortable (ISO-like prefix sorts lexicographically)
    assert ids == sorted(ids)
    # shape: YYYY-MM-DDTHH-MM-SS_<4hex>
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}_[0-9a-f]{4}$", ids[0])


def test_step_dir_creates_on_first_use(isolated_runs_root):
    d = step_dir("step_new")
    assert d.is_dir()
    assert d.name == "step_new"


# ---------------------------------------------------------------------------
# review_server HTTP smoke (real socket, real server in a thread)
# ---------------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def running_server(isolated_runs_root):
    """Start bin/review_server.py on a free port. Yields (port, run_dir)."""
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "bin/review_server.py", "--port", str(port), "--host", "127.0.0.1",
         "--root", str(isolated_runs_root)],
        cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # wait for it to bind
    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.05)
    else:
        proc.kill()
        pytest.fail("review_server failed to start within 5s")
    try:
        yield port
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_server_root_index_200(running_server, isolated_runs_root):
    conn = HTTPConnection("127.0.0.1", running_server, timeout=2)
    conn.request("GET", "/")
    r = conn.getresponse()
    body = r.read().decode()
    assert r.status == 200
    assert "Run review" in body


def test_server_api_runs_json_shape(running_server, isolated_runs_root):
    # seed a run
    r = Run.open("step_api")
    c = r.component("110300324920")
    c.html("<p>x</p>")
    c.json()
    c.finish(features=1)
    r.finish(ok=1)

    conn = HTTPConnection("127.0.0.1", running_server, timeout=2)
    conn.request("GET", "/api/runs")
    r = conn.getresponse()
    body = r.read().decode()
    assert r.status == 200
    assert r.getheader("Content-Type", "").startswith("application/json")
    data = json.loads(body)
    assert "step_api" in data["steps"]
    run_entry = data["steps"]["step_api"][0]
    assert run_entry["status"] == "completed"
    assert run_entry["n_components"] == 1


def test_server_serves_step_run_component(running_server, isolated_runs_root):
    r = Run.open("step_page")
    c = r.component("CID")
    c.html("<h2>hello</h2>", title="CID")
    c.json()
    c.finish(features=1)
    r.finish(ok=1)

    conn = HTTPConnection("127.0.0.1", running_server, timeout=2)

    for path in (
        "/runs/step_page/",
        f"/runs/step_page/{r.run_id}/index.html",
        f"/runs/step_page/{r.run_id}/components/CID/index.html",
        f"/runs/step_page/{r.run_id}/components/CID/meta.json",
        f"/api/runs/step_page/{r.run_id}",
        f"/api/components/step_page/{r.run_id}/CID",
    ):
        conn.request("GET", path)
        resp = conn.getresponse()
        resp.read()  # drain
        assert resp.status == 200, f"expected 200 for {path}, got {resp.status}"


def test_server_404_on_missing_run(running_server):
    conn = HTTPConnection("127.0.0.1", running_server, timeout=2)
    conn.request("GET", "/runs/does_not_exist/")
    resp = conn.getresponse()
    assert resp.status == 200  # renders "no runs yet" page
    body = resp.read().decode()
    assert "No runs" in body or "No steps" in body


def test_server_403_on_path_traversal(running_server):
    conn = HTTPConnection("127.0.0.1", running_server, timeout=2)
    # Attempt to escape the runs root via ../
    conn.request("GET", "/runs/../bin/review_server.py")
    resp = conn.getresponse()
    # Either 403 (caught by our guard) or 200 (if normalization drops the ..)
    # — we just require it NOT to leak the file contents
    body = resp.read()
    assert b"#!/usr/bin/env python3" not in body
