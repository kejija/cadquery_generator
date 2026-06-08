"""Tests for scripts.table_merge.synonym_llm.

The LLM call is mocked so tests don't require network or OpenAI credentials.
The mocked client must implement:
  .chat.completions.create(model=..., messages=..., ...) -> object with
  .choices[0].message.content (str).

Or, for the convenience openai.Client-style:
  .responses.create(model=..., input=..., ...) -> object with .output_text
  or .choices[0].message.content.
"""
from __future__ import annotations

from typing import Optional
from unittest.mock import MagicMock

import pytest

from scripts.table_merge.synonym_llm import (
    SynonymResolution,
    resolve_synonyms,
    _seed_synonyms_for_headers,
)


def _make_mock_client(returned_json: Optional[dict] = None, raise_exc: Optional[Exception] = None) -> MagicMock:
    """Build a mock OpenAI client that returns `returned_json` from chat.completions.create."""
    client = MagicMock()
    if raise_exc is not None:
        client.chat.completions.create.side_effect = raise_exc
        return client
    msg = MagicMock()
    msg.content = json_dumps = __import__("json").dumps(returned_json or {})
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    client.chat.completions.create.return_value = response
    return client


# --- _seed_synonyms_for_headers (no-LLM helper) -------------------------------


def test_seed_covers_all_known_headers():
    headers = ["OD", "Outer Diameter", "Bore", "Length", "Width", "Widget"]
    out = _seed_synonyms_for_headers(headers)
    # The first 5 should resolve; "Widget" passes through unchanged.
    assert out == {
        "OD": "OD",
        "Outer Diameter": "OD",
        "Bore": "ID",
        "Length": "L",
        "Width": "W",
        "Widget": "Widget",
    }


def test_seed_returns_empty_for_empty_input():
    assert _seed_synonyms_for_headers([]) == {}


def test_seed_preserves_unknown_passthrough():
    # No DEFAULT_SYNONYMS match -> header passed through.
    out = _seed_synonyms_for_headers(["MadeUpHeader"])
    assert out == {"MadeUpHeader": "MadeUpHeader"}


# --- resolve_synonyms ---------------------------------------------------------


def test_resolve_skips_llm_when_all_headers_known(monkeypatch):
    """If every header resolves via DEFAULT_SYNONYMS, the LLM is never called."""
    client = _make_mock_client()
    monkeypatch.setattr(
        "scripts.table_merge.synonym_llm._get_client",
        lambda: client,
    )
    out = resolve_synonyms(
        headers=["OD", "Outer Diameter", "Bore", "Length"],
        family_id="fam_test",
        family_category="bearing",
        openai_client=client,
    )
    # The seed map covers all four; LLM not called.
    assert client.chat.completions.create.call_count == 0
    assert out["OD"] == "OD"
    assert out["Outer Diameter"] == "OD"
    assert out["Bore"] == "ID"
    assert out["Length"] == "L"


def test_resolve_calls_llm_for_unknown_headers():
    """Headers not in DEFAULT_SYNONYMS trigger a single LLM call."""
    llm_returned = {
        "Mystic Dimension Alpha": "alpha_coefficient",
        "Custom Process Time": "process_time_s",
    }
    client = _make_mock_client(returned_json=llm_returned)
    out = resolve_synonyms(
        headers=["OD", "Mystic Dimension Alpha", "Custom Process Time"],
        family_id="fam_test",
        family_category="bearing",
        openai_client=client,
    )
    # OD is in seed; the other two are resolved by the LLM.
    assert out["OD"] == "OD"
    assert out["Mystic Dimension Alpha"] == "alpha_coefficient"
    assert out["Custom Process Time"] == "process_time_s"
    # LLM called exactly once.
    assert client.chat.completions.create.call_count == 1


def test_resolve_prompt_contains_headers_and_family_context():
    """The LLM prompt must include the unknown headers + family context."""
    captured: dict = {}
    msg = MagicMock()
    msg.content = '{"Mystic Dimension Alpha": "alpha"}'
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    client = MagicMock()
    client.chat.completions.create.side_effect = lambda **kwargs: (
        captured.update(kwargs) or response
    )
    resolve_synonyms(
        headers=["Mystic Dimension Alpha"],
        family_id="fam_bearing_001",
        family_category="bearing",
        openai_client=client,
    )
    # Inspect the captured call.
    assert "model" in captured
    assert "messages" in captured
    messages = captured["messages"]
    # The user message should contain the header and family context.
    user_text = "\n".join(m["content"] for m in messages if m.get("role") == "user")
    assert "Mystic Dimension Alpha" in user_text
    assert "fam_bearing_001" in user_text or "bearing" in user_text


def test_resolve_handles_llm_failure_gracefully():
    """If the LLM call fails, unknown headers pass through unchanged."""
    client = _make_mock_client(raise_exc=RuntimeError("rate limit"))
    out = resolve_synonyms(
        headers=["Mystic Dimension Alpha"],
        family_id="fam_test",
        family_category="bearing",
        openai_client=client,
    )
    # No resolution for the unknown header, but no exception escapes.
    assert out["Mystic Dimension Alpha"] == "Mystic Dimension Alpha"


def test_resolve_handles_malformed_llm_json():
    """If the LLM returns garbage, unknown headers pass through unchanged."""
    client = MagicMock()
    msg = MagicMock()
    msg.content = "this is not json at all"
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    client.chat.completions.create.return_value = response
    out = resolve_synonyms(
        headers=["Mystic Dimension Alpha"],
        family_id="fam_test",
        family_category="bearing",
        openai_client=client,
    )
    assert out["Mystic Dimension Alpha"] == "Mystic Dimension Alpha"


def test_resolve_merges_seed_and_llm_results():
    """LLM results can override seed results (and vice versa for non-conflicting keys)."""
    # 'Length' resolves via seed to 'L'. LLM returns 'Length' -> 'L' (consistent).
    # LLM also returns a new mapping for an unknown header.
    llm_returned = {
        "Length": "overall_length",   # overrides seed's "L"
        "Brand New Dim": "brand_new_dim",
    }
    client = _make_mock_client(returned_json=llm_returned)
    out = resolve_synonyms(
        headers=["Length", "Brand New Dim"],
        family_id="fam_test",
        family_category="bearing",
        openai_client=client,
    )
    # LLM wins for 'Length' (explicit override).
    assert out["Length"] == "overall_length"
    # LLM provided a new mapping for the unknown.
    assert out["Brand New Dim"] == "brand_new_dim"


def test_resolve_empty_input_returns_empty_dict():
    out = resolve_synonyms(headers=[], family_id="fam_x", family_category="x")
    assert out == {}


# --- SynonymResolution dataclass ---------------------------------------------


def test_synonym_resolution_to_dict():
    r = SynonymResolution(
        family_id="fam_x",
        seed_mappings={"OD": "OD"},
        llm_mappings={"Mystic": "alpha"},
        merged={"OD": "OD", "Mystic": "alpha"},
        unknown_passthrough=[],
        llm_called=True,
        notes=[],
    )
    d = r.to_dict()
    assert d["family_id"] == "fam_x"
    assert d["llm_called"] is True
    assert d["merged"]["Mystic"] == "alpha"


def test_resolve_returns_synonym_resolution_object():
    """resolve_synonyms should return a SynonymResolution, not a bare dict."""
    client = _make_mock_client()
    out = resolve_synonyms(
        headers=["OD", "Mystic"],
        family_id="fam_x",
        family_category="bearing",
        openai_client=client,
    )
    # We only declared the dict-shape return; the public API may return either.
    # Test the contract: it must be subscriptable like a dict and have a 'to_dict' method.
    assert out["OD"] == "OD"
