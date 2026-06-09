"""LLM-based conflict resolver (Phase C of the Step 0.6 pipeline).

When the deterministic merger (Task 8) produces a Conflict — two sources
disagree on the value of (row_key, column) — we send the conflict to an
LLM and ask which value to use, with reasoning. Resolutions are:

  - "use_value_a"          : keep value_a (drop value_b)
  - "use_value_b"          : keep value_b (drop value_a)
  - "use_both_distinct"    : the values are intentionally different
                              (e.g. different models have different specs)
  - "needs_human_review"   : cannot decide with confidence

We use the OpenAI Batch API (the same pattern Step 1 uses) to send all
conflicts in a single batch job. Each conflict is its own request so the
LLM can reason about each one independently. Resolutions are returned in
the same order as the input conflicts.

Design notes
------------
- One request per conflict, not chunked. Conflict counts are typically
  10-50 per family; this is well under the 50k-per-batch limit and the
  Batch API is 50% cheaper than synchronous calls.
- The LLM is asked for JSON; we use a strict JSON parser that handles
  markdown fences.
- The Batch API may take minutes to hours; this is async by design.
- For testing, an injectable ``openai_client`` lets us skip the network
  entirely (see tests/table_merge/test_conflict_llm.py).
"""
from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from scripts.table_merge.merge import Conflict


logger = logging.getLogger(__name__)

DEFAULT_LLM_MODEL = "gpt-5.4-mini"
VALID_RESOLUTIONS = {"use_value_a", "use_value_b", "use_both_distinct", "needs_human_review"}


@dataclass
class ConflictResolution:
    """The LLM's answer for a single Conflict."""
    conflict_index: int
    resolution: str          # one of VALID_RESOLUTIONS
    confidence: float        # 0.0 - 1.0
    reasoning: str = ""
    chosen_value: object = None  # the value to use, if any

    def to_dict(self) -> dict:
        return {
            "conflict_index": self.conflict_index,
            "resolution": self.resolution,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "chosen_value": self.chosen_value,
        }


# --- Prompt helpers -----------------------------------------------------------


def _build_request_body(
    conflict: Conflict,
    conflict_index: int,
    model: str,
) -> dict:
    """Build a single OpenAI batch request body for one conflict."""
    system = (
        "You are a senior mechanical engineering data reconciler. "
        "Two catalog sources disagree on a value for a row/column. "
        "Return ONLY a JSON object with: "
        'resolution (one of "use_value_a", "use_value_b", "use_both_distinct", "needs_human_review"), '
        "confidence (0.0-1.0), reasoning (short), chosen_value (the value to use, or null). "
        "Do not add commentary. Do not wrap the JSON in markdown."
    )
    user = (
        f"Conflict {conflict_index}:\n"
        f"  table_type: {conflict.table_type}\n"
        f"  row_key:    {conflict.row_key!r}\n"
        f"  column:     {conflict.column!r}\n"
        f"  value_a:    {conflict.value_a!r}   (source: {conflict.source_a})\n"
        f"  value_b:    {conflict.value_b!r}   (source: {conflict.source_b})\n"
        f"  note:       {conflict.note!r}\n\n"
        "Which value should be kept? If they're both legitimate (e.g. different\n"
        "configurations of the same model), choose use_both_distinct."
    )
    return {
        "custom_id": f"conflict-{conflict_index}-{uuid.uuid4().hex[:8]}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
    }


def _parse_resolution(content: str) -> Optional[dict]:
    """Extract {resolution, confidence, reasoning, chosen_value} from a response."""
    if not content:
        return None
    text = content.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    # Direct parse.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "resolution" in obj:
            return obj
    except (json.JSONDecodeError, ValueError):
        pass
    # Brace-match fallback.
    depth = 0
    start = text.find("{")
    if start < 0:
        return None
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start : i + 1])
                    if isinstance(obj, dict) and "resolution" in obj:
                        return obj
                except (json.JSONDecodeError, ValueError):
                    return None
    return None


def _coerce_resolution(raw: dict, conflict: Conflict) -> ConflictResolution:
    """Normalize a parsed resolution into a ConflictResolution object."""
    resolution = str(raw.get("resolution", "needs_human_review")).strip()
    if resolution not in VALID_RESOLUTIONS:
        resolution = "needs_human_review"
    try:
        confidence = float(raw.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))
    reasoning = str(raw.get("reasoning", "")).strip()
    chosen = raw.get("chosen_value", None)
    if resolution == "use_value_a":
        chosen = conflict.value_a
    elif resolution == "use_value_b":
        chosen = conflict.value_b
    return ConflictResolution(
        conflict_index=0,  # set by caller
        resolution=resolution,
        confidence=confidence,
        reasoning=reasoning,
        chosen_value=chosen,
    )


# --- Synchronous fallback (no Batch) -----------------------------------------


def resolve_conflicts_sync(
    conflicts: list[Conflict],
    openai_client: Optional[object] = None,
    model: str = DEFAULT_LLM_MODEL,
) -> list[ConflictResolution]:
    """Resolve conflicts one-at-a-time using chat.completions.create.

    Used for small conflict counts (<5) where Batch overhead is unwarranted,
    and as a fallback if the Batch path fails. Slower per-call but simpler.
    """
    if not conflicts:
        return []
    client = openai_client
    if client is None:
        import openai
        client = openai.OpenAI()

    out: list[ConflictResolution] = []
    for i, c in enumerate(conflicts):
        body = _build_request_body(c, i, model)
        try:
            response = client.chat.completions.create(**body["body"])
            content = response.choices[0].message.content
            parsed = _parse_resolution(content or "")
        except Exception as e:  # noqa: BLE001
            logger.warning("resolve_conflicts_sync: call %d failed: %s", i, e)
            parsed = None
        if parsed:
            res = _coerce_resolution(parsed, c)
        else:
            res = ConflictResolution(
                conflict_index=i,
                resolution="needs_human_review",
                confidence=0.0,
                reasoning="LLM call failed or returned unparseable response",
                chosen_value=None,
            )
        res.conflict_index = i
        out.append(res)
    return out


# --- Batch path ---------------------------------------------------------------


def submit_conflict_batch(
    conflicts: list[Conflict],
    jsonl_path: str,
    openai_client: Optional[object] = None,
    model: str = DEFAULT_LLM_MODEL,
) -> str:
    """Write the conflict-resolution requests to a JSONL file and submit a
    Batch job. Returns the batch_id.

    The caller is expected to poll ``poll_conflict_batch`` later.
    """
    if not conflicts:
        raise ValueError("submit_conflict_batch: empty conflict list")
    client = openai_client
    if client is None:
        import openai
        client = openai.OpenAI()

    with open(jsonl_path, "w") as f:
        for i, c in enumerate(conflicts):
            body = _build_request_body(c, i, model)
            f.write(json.dumps(body) + "\n")

    with open(jsonl_path, "rb") as f:
        batch = client.batches.create(
            input_file=f,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
    return batch.id


def poll_conflict_batch(
    batch_id: str,
    openai_client: Optional[object] = None,
    poll_interval_s: float = 5.0,
    max_wait_s: float = 60.0,
) -> list[ConflictResolution]:
    """Poll a Batch job until completion (or max_wait_s) and return resolutions.

    This is a *bounded* poller intended for tests and small jobs. In
    production, a longer-lived worker would call this in a loop with a
    much larger max_wait_s.
    """
    client = openai_client
    if client is None:
        import openai
        client = openai.OpenAI()

    deadline = time.time() + max_wait_s
    while time.time() < deadline:
        batch = client.batches.retrieve(batch_id)
        status = getattr(batch, "status", None)
        if status in ("completed", "failed", "expired", "cancelled"):
            break
        time.sleep(poll_interval_s)
    else:
        logger.warning("poll_conflict_batch: timed out after %ss for batch %s", max_wait_s, batch_id)
        return []

    if status != "completed":
        logger.warning("poll_conflict_batch: batch %s ended with status %s", batch_id, status)
        return []

    # Download the output file and parse it.
    output_file_id = batch.output_file_id
    if not output_file_id:
        logger.warning("poll_conflict_batch: batch %s has no output_file_id", batch_id)
        return []
    file_resp = client.files.content(output_file_id)
    text = file_resp.read().decode("utf-8") if hasattr(file_resp, "read") else str(file_resp)
    return _parse_batch_output(text)


def _parse_batch_output(text: str) -> list[ConflictResolution]:
    """Parse the JSONL batch output into ConflictResolution objects.

    The output is sorted by ``custom_id``; we extract the conflict_index
    back from the id. If parsing fails for a line, we emit a
    ConflictResolution with ``needs_human_review``.
    """
    out: list[ConflictResolution] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        # Custom id is "conflict-N-XXXX" — extract N.
        cid = obj.get("custom_id", "")
        m = re.match(r"^conflict-(\d+)-", cid)
        idx = int(m.group(1)) if m else len(out)
        # Response body is in obj['response']['body'] for a successful line,
        # or obj['error'] for a failure.
        body = obj.get("response", {}).get("body") or {}
        try:
            content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        except (KeyError, IndexError):
            content = ""
        parsed = _parse_resolution(content or "")
        # We need a Conflict to coerce; reconstruct a minimal one from parsed.
        # If parsing failed, emit needs_human_review.
        if parsed:
            fake_conflict = Conflict(
                table_type=parsed.get("table_type", ""),
                row_key=parsed.get("row_key", ""),
                column=parsed.get("column", ""),
                value_a=parsed.get("value_a"),
                value_b=parsed.get("value_b"),
                source_a=parsed.get("source_a", ""),
                source_b=parsed.get("source_b", ""),
            )
            res = _coerce_resolution(parsed, fake_conflict)
        else:
            res = ConflictResolution(
                conflict_index=idx,
                resolution="needs_human_review",
                confidence=0.0,
                reasoning="batch output unparseable",
                chosen_value=None,
            )
        res.conflict_index = idx
        out.append(res)
    # Stable order by conflict_index.
    out.sort(key=lambda r: r.conflict_index)
    return out


# --- Public entry point -------------------------------------------------------


def resolve_conflicts(
    conflicts: list[Conflict],
    openai_client: Optional[object] = None,
    model: str = DEFAULT_LLM_MODEL,
    use_batch_threshold: int = 5,
) -> list[ConflictResolution]:
    """Resolve N conflicts.

    Strategy:
      - If the conflict count is below ``use_batch_threshold``, use the
        synchronous chat.completions path (cheaper for small N).
      - Otherwise, use the Batch API.
      - If ``openai_client`` is None and we need to call OpenAI, this will
        use the default credentials. Tests should pass a mock client.
    """
    if not conflicts:
        return []
    if len(conflicts) < use_batch_threshold:
        return resolve_conflicts_sync(conflicts, openai_client=openai_client, model=model)
    # Batch path. Write to a temp JSONL and submit.
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tf:
        jsonl_path = tf.name
    try:
        batch_id = submit_conflict_batch(conflicts, jsonl_path, openai_client=openai_client, model=model)
        return poll_conflict_batch(batch_id, openai_client=openai_client, max_wait_s=10.0)
    finally:
        try:
            import os
            os.unlink(jsonl_path)
        except OSError:
            pass
