"""Tests for scripts.table_merge.conflict_llm.

All tests use mocked OpenAI clients; no network calls.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from scripts.table_merge.conflict_llm import (
    ConflictResolution,
    _build_request_body,
    _coerce_resolution,
    _parse_resolution,
    resolve_conflicts,
    resolve_conflicts_sync,
    submit_conflict_batch,
    poll_conflict_batch,
    VALID_RESOLUTIONS,
)
from scripts.table_merge.merge import Conflict


def _conflict(idx: int = 0, **kwargs) -> Conflict:
    base = dict(
        table_type="variant_lookup",
        row_key="22",
        column="B",
        value_a=7,
        value_b=7.5,
        source_a="A#t0",
        source_b="B#t1",
        note="",
    )
    base.update(kwargs)
    return Conflict(**base)


def _mock_chat_response(payload: dict) -> MagicMock:
    msg = MagicMock()
    msg.content = json.dumps(payload)
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _mock_client_with_response(payload: dict) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_chat_response(payload)
    return client


# --- _build_request_body ------------------------------------------------------


def test_build_request_body_contains_conflict_details():
    c = _conflict()
    body = _build_request_body(c, 0, "gpt-5.4-mini")
    assert body["method"] == "POST"
    assert body["url"] == "/v1/chat/completions"
    assert body["custom_id"].startswith("conflict-0-")
    messages = body["body"]["messages"]
    user_text = messages[1]["content"]
    assert "row_key:    '22'" in user_text
    assert "value_a:    7" in user_text
    assert "value_b:    7.5" in user_text


# --- _parse_resolution / _coerce_resolution ----------------------------------


def test_parse_resolution_direct_json():
    out = _parse_resolution('{"resolution": "use_value_a", "confidence": 0.9, "reasoning": "x"}')
    assert out == {"resolution": "use_value_a", "confidence": 0.9, "reasoning": "x"}


def test_parse_resolution_strips_markdown_fences():
    out = _parse_resolution('```json\n{"resolution": "use_value_a"}\n```')
    assert out == {"resolution": "use_value_a"}


def test_parse_resolution_handles_preamble():
    out = _parse_resolution('Here is the answer:\n{"resolution": "use_value_b"}')
    assert out == {"resolution": "use_value_b"}


def test_parse_resolution_returns_none_for_garbage():
    assert _parse_resolution("this is not json") is None
    assert _parse_resolution("") is None
    assert _parse_resolution(None) is None  # type: ignore[arg-type]


def test_coerce_resolution_clamps_confidence():
    c = _conflict()
    out = _coerce_resolution({"resolution": "use_value_a", "confidence": 1.5}, c)
    assert out.confidence == 1.0
    out = _coerce_resolution({"resolution": "use_value_a", "confidence": -0.2}, c)
    assert out.confidence == 0.0


def test_coerce_resolution_unknown_resolution_falls_back_to_human_review():
    c = _conflict()
    out = _coerce_resolution({"resolution": "garbage", "confidence": 0.5}, c)
    assert out.resolution == "needs_human_review"


def test_coerce_resolution_chooses_correct_value_for_each_resolution():
    c = _conflict(value_a=7, value_b=7.5)
    out_a = _coerce_resolution({"resolution": "use_value_a"}, c)
    assert out_a.chosen_value == 7
    out_b = _coerce_resolution({"resolution": "use_value_b"}, c)
    assert out_b.chosen_value == 7.5
    out_d = _coerce_resolution({"resolution": "use_both_distinct"}, c)
    # use_both_distinct: chosen_value is whatever the LLM returned (none here)
    assert out_d.chosen_value is None


# --- resolve_conflicts_sync ---------------------------------------------------


def test_resolve_conflicts_sync_empty_input():
    out = resolve_conflicts_sync([])
    assert out == []


def test_resolve_conflicts_sync_calls_llm_once_per_conflict():
    client = _mock_client_with_response({
        "resolution": "use_value_a",
        "confidence": 0.9,
        "reasoning": "A is more recent",
    })
    conflicts = [_conflict(0), _conflict(1, value_a=1, value_b=2), _conflict(2)]
    out = resolve_conflicts_sync(conflicts, openai_client=client)
    assert len(out) == 3
    assert client.chat.completions.create.call_count == 3
    # Indices preserved.
    assert [r.conflict_index for r in out] == [0, 1, 2]
    # Resolutions match what the mock returned.
    for r in out:
        assert r.resolution == "use_value_a"
        # use_value_a coercion sets chosen_value to the conflict's value_a.
        expected_values = {0: 7, 1: 1, 2: 7}
        assert r.chosen_value == expected_values[r.conflict_index]


def test_resolve_conflicts_sync_handles_llm_failure():
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("rate limit")
    out = resolve_conflicts_sync([_conflict(0)], openai_client=client)
    assert len(out) == 1
    assert out[0].resolution == "needs_human_review"
    assert out[0].confidence == 0.0


def test_resolve_conflicts_sync_handles_unparseable_response():
    client = MagicMock()
    msg = MagicMock()
    msg.content = "garbage no json"
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    client.chat.completions.create.return_value = resp
    out = resolve_conflicts_sync([_conflict(0)], openai_client=client)
    assert out[0].resolution == "needs_human_review"


# --- resolve_conflicts (entry point, batch path) -----------------------------


def test_resolve_conflicts_small_n_uses_sync_path():
    """Below the batch threshold, the sync path is used (mocked)."""
    client = _mock_client_with_response({"resolution": "use_value_a", "confidence": 0.7})
    out = resolve_conflicts(
        conflicts=[_conflict(0), _conflict(1)],
        openai_client=client,
        use_batch_threshold=5,
    )
    assert len(out) == 2
    assert client.chat.completions.create.call_count == 2


def test_resolve_conflicts_empty_returns_empty():
    out = resolve_conflicts(conflicts=[], openai_client=MagicMock())
    assert out == []


# --- submit_conflict_batch / poll_conflict_batch ----------------------------


def test_submit_conflict_batch_writes_jsonl_and_submits(tmp_path):
    jsonl_path = tmp_path / "batch.jsonl"
    client = MagicMock()
    batch = MagicMock()
    batch.id = "batch_abc123"
    client.batches.create.return_value = batch

    conflicts = [_conflict(0), _conflict(1)]
    batch_id = submit_conflict_batch(conflicts, str(jsonl_path), openai_client=client)
    assert batch_id == "batch_abc123"
    assert jsonl_path.exists()

    # Verify the JSONL has one line per conflict.
    lines = jsonl_path.read_text().strip().splitlines()
    assert len(lines) == 2
    obj0 = json.loads(lines[0])
    assert obj0["custom_id"].startswith("conflict-0-")
    obj1 = json.loads(lines[1])
    assert obj1["custom_id"].startswith("conflict-1-")


def test_poll_conflict_batch_returns_empty_for_incomplete(monkeypatch):
    """If the batch is not yet completed, poll returns empty within max_wait_s."""
    client = MagicMock()
    batch = MagicMock()
    batch.status = "in_progress"
    client.batches.retrieve.return_value = batch
    monkeypatch.setattr("time.sleep", lambda _s: None)  # speed up the test
    out = poll_conflict_batch("batch_xyz", openai_client=client, max_wait_s=0.5, poll_interval_s=0.1)
    assert out == []


def test_poll_conflict_batch_parses_completed_output(monkeypatch):
    """When the batch completes, poll returns parsed ConflictResolution objects."""
    client = MagicMock()
    batch = MagicMock()
    batch.status = "completed"
    batch.output_file_id = "file_abc"
    client.batches.retrieve.return_value = batch

    # Build a fake batch output JSONL.
    output_lines = [
        json.dumps({
            "custom_id": "conflict-0-abc",
            "response": {
                "body": {
                    "choices": [{
                        "message": {
                            "content": json.dumps({
                                "resolution": "use_value_a",
                                "confidence": 0.8,
                                "reasoning": "A is correct",
                            }),
                        },
                    }],
                },
            },
        }),
        json.dumps({
            "custom_id": "conflict-1-def",
            "response": {
                "body": {
                    "choices": [{
                        "message": {
                            "content": json.dumps({
                                "resolution": "needs_human_review",
                                "confidence": 0.3,
                                "reasoning": "Cannot decide",
                            }),
                        },
                    }],
                },
            },
        }),
    ]
    file_content = "\n".join(output_lines).encode()
    file_resp = MagicMock()
    file_resp.read.return_value = file_content
    client.files.content.return_value = file_resp

    monkeypatch.setattr("time.sleep", lambda _s: None)
    out = poll_conflict_batch("batch_done", openai_client=client, max_wait_s=0.1)
    assert len(out) == 2
    # Sorted by conflict_index.
    assert out[0].conflict_index == 0
    assert out[0].resolution == "use_value_a"
    assert out[1].conflict_index == 1
    assert out[1].resolution == "needs_human_review"


def test_valid_resolutions_includes_all_expected():
    assert VALID_RESOLUTIONS == {
        "use_value_a", "use_value_b", "use_both_distinct", "needs_human_review",
    }
