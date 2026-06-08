"""LLM-based header synonym resolver (Phase A of the Step 0.6 pipeline).

Strategy:
  1. Apply DEFAULT_SYNONYMS to every header. Anything that resolves, stays.
  2. Collect the *unknown* headers (passthrough in step 1).
  3. If there are no unknowns, return the seed result without calling the LLM.
  4. Otherwise, send ONE chat completion to OpenAI with:
       - The list of unknown headers.
       - The family_id + family_category as context.
       - An instruction to return JSON {original_header: canonical}.
  5. Parse the JSON, merge with the seed result (LLM takes precedence for
     keys it explicitly returns).
  6. Any header still unresolved passes through unchanged.

The single-call design is intentional: this is Phase A and runs at most
once per family. A typical family has 5-15 unknown headers, well within
the 8k context window of any modern LLM. No batching, no JSON-mode
specifics — just a clear system prompt and a regex-validated JSON parse.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from scripts.table_merge.classify import DEFAULT_SYNONYMS, resolve_header


logger = logging.getLogger(__name__)

DEFAULT_LLM_MODEL = "gpt-5.4-mini"


# --- Data model ---------------------------------------------------------------


@dataclass
class SynonymResolution:
    """The full result of one resolve_synonyms call.

    `seed_mappings` is what DEFAULT_SYNONYMS produced (may include passthroughs
    for unknown headers). `llm_mappings` is what the LLM produced, if called.
    `merged` is the final dict the merger uses: seed first, LLM overrides.
    `unknown_passthrough` lists headers that ended up unchanged (callers may
    want to flag these for human review).
    """
    family_id: str
    seed_mappings: dict[str, str]
    llm_mappings: dict[str, str] = field(default_factory=dict)
    merged: dict[str, str] = field(default_factory=dict)
    unknown_passthrough: list[str] = field(default_factory=list)
    llm_called: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "family_id": self.family_id,
            "seed_mappings": dict(self.seed_mappings),
            "llm_mappings": dict(self.llm_mappings),
            "merged": dict(self.merged),
            "unknown_passthrough": list(self.unknown_passthrough),
            "llm_called": self.llm_called,
            "notes": list(self.notes),
        }


# --- Helpers ------------------------------------------------------------------


def _seed_synonyms_for_headers(headers: list[str]) -> dict[str, str]:
    """Apply DEFAULT_SYNONYMS to every header.

    Returns ``{original_header: canonical}``. Unknown headers pass through
    unchanged (i.e. map to themselves).
    """
    return {h: resolve_header(h, DEFAULT_SYNONYMS) for h in headers}


def _extract_json_object(text: str) -> Optional[dict]:
    """Find the first JSON object in a string.

    LLMs often wrap JSON in ```json ... ``` fences or add preamble text. This
    helper extracts the first balanced { ... } block. Returns None if no
    valid JSON object is found.
    """
    text = text.strip()
    # Strip common markdown code fences.
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    # Direct parse first.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass
    # Fall back: find the first balanced { ... }.
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
                candidate = text[start : i + 1]
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        return obj
                except (json.JSONDecodeError, ValueError):
                    return None
    return None


def _build_prompt(headers: list[str], family_id: str, family_category: str) -> list[dict]:
    """Build the chat-completion messages for the synonym resolver."""
    system = (
        "You are a senior mechanical engineering catalog taxonomist. "
        "Your job is to normalize dimension/datasheet table column headers "
        "to short, machine-friendly canonical symbols (e.g. 'OD', 'ID', 'L', 'T', 'R'). "
        "Return ONLY a JSON object: {original_header: canonical_symbol}. "
        "Do not add commentary. Do not wrap the JSON in markdown."
    )
    user_lines = [
        f"Family ID: {family_id}",
        f"Family category: {family_category}",
        "",
        "Map each of these column headers to a canonical symbol:",
    ]
    for h in headers:
        user_lines.append(f"  - {h!r}")
    user_lines.append("")
    user_lines.append("Return the JSON mapping.")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(user_lines)},
    ]


# --- Main entry point ---------------------------------------------------------


def resolve_synonyms(
    headers: list[str],
    family_id: str,
    family_category: str,
    openai_client: Optional[object] = None,
    model: str = DEFAULT_LLM_MODEL,
) -> dict[str, str]:
    """Resolve a list of headers to canonical form, falling back to LLM when
    DEFAULT_SYNONYMS doesn't cover them.

    Returns ``{original_header: canonical}`` for every input header.

    The returned dict is also exposed as the ``.merged`` field of the
    SynonymResolution object, accessible via :func:`resolve_synonyms_full`.
    """
    full = resolve_synonyms_full(
        headers=headers,
        family_id=family_id,
        family_category=family_category,
        openai_client=openai_client,
        model=model,
    )
    return full.merged


def resolve_synonyms_full(
    headers: list[str],
    family_id: str,
    family_category: str,
    openai_client: Optional[object] = None,
    model: str = DEFAULT_LLM_MODEL,
) -> SynonymResolution:
    """Full result variant: returns the SynonymResolution object so callers
    can inspect seed_mappings, llm_mappings, etc. separately.
    """
    seed = _seed_synonyms_for_headers(headers)
    merged = dict(seed)

    # Call the LLM.
    llm_result: dict[str, str] = {}
    # Identify headers that did NOT resolve to a known canonical. A header
    # is "unknown" iff it passes through resolve_header unchanged AND the
    # canonical set doesn't already include that name. ("OD" -> "OD" is
    # NOT unknown, because "OD" is itself a canonical.)
    known_canonicals = set(DEFAULT_SYNONYMS.keys())
    unknown = [
        h for h in headers
        if h and (seed.get(h) == h) and (h not in known_canonicals)
    ]
    if not unknown:
        return SynonymResolution(
            family_id=family_id,
            seed_mappings=seed,
            merged=merged,
            unknown_passthrough=[],
            llm_called=False,
        )
    try:
        client = openai_client or _get_client()
        messages = _build_prompt(unknown, family_id, family_category)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        content = response.choices[0].message.content
        parsed = _extract_json_object(content or "")
        if parsed:
            for k, v in parsed.items():
                if isinstance(k, str) and isinstance(v, str) and k and v:
                    llm_result[k] = v.strip()
    except Exception as e:  # noqa: BLE001 - log and continue; resolve degrades gracefully
        logger.warning("resolve_synonyms: LLM call failed for %s: %s", family_id, e)

    # Merge: LLM overrides seed for keys the LLM explicitly returned.
    for k, v in llm_result.items():
        merged[k] = v

    passthrough = [h for h in unknown if merged.get(h) == h]
    notes = []
    if not llm_result:
        notes.append("LLM call produced no usable mappings; unknown headers passed through.")
    if passthrough:
        notes.append(f"{len(passthrough)} unknown header(s) passed through unchanged: {passthrough[:5]}")

    return SynonymResolution(
        family_id=family_id,
        seed_mappings=seed,
        llm_mappings=llm_result,
        merged=merged,
        unknown_passthrough=passthrough,
        llm_called=True,
        notes=notes,
    )


def _get_client():
    """Construct a default OpenAI client. Lazy import to keep tests fast."""
    import openai
    return openai.OpenAI()
