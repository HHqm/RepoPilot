"""Retrieval query planning and cross-query fusion for the coding agent.

A single "issue text" query under-retrieves when the issue names a symbol
that its prose does not describe, and it cannot see why a previous patch
failed. The planner derives up to three complementary queries
deterministically (no extra LLM call) and their rankings are merged with
Reciprocal Rank Fusion, so a chunk ranked well by any query surfaces.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..models import Evidence


_FILE_PATH = re.compile(
    r"(?<![\w/.-])[\w.-]+(?:/[\w.-]+)*\.(?:py|js|jsx|ts|tsx|java|go|rs|c|cc|cpp|h|hpp|cs|rb|php|kt|swift)\b"
)
_DOTTED_NAME = re.compile(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\b")
_SNAKE_CASE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
# Mixed-case identifiers with an inner capital: parseConfig, HTTPClient.
_CAMEL_CASE = re.compile(r"\b(?=\w*[a-z])[A-Za-z][a-z0-9]*[A-Z]\w*\b")
_TRACEBACK_FRAME = re.compile(r'File "([^"]+)", line \d+, in (\w+)')
_FAILURE_LINE = re.compile(
    r"^\s*(?:FAIL|ERROR):\s*(\w+)|^\s*(\w*(?:Error|Exception|Failure)\b.*)$",
    re.MULTILINE,
)
_MAX_SYMBOLS = 12
_MAX_QUERY_CHARS = 1_200
RRF_K = 60


@dataclass(frozen=True)
class PlannedQuery:
    purpose: str
    text: str

    def to_dict(self) -> dict[str, str]:
        return {"purpose": self.purpose, "query": self.text[:200]}


def extract_code_symbols(text: str) -> list[str]:
    """File paths, dotted names and snake/camel identifiers, in order."""

    found: list[str] = []
    for pattern in (_FILE_PATH, _DOTTED_NAME, _SNAKE_CASE, _CAMEL_CASE):
        for match in pattern.findall(text):
            if match not in found:
                found.append(match)
    # "calculator.py" is redundant next to "src/calculator.py".
    return [
        item
        for item in found
        if not any(
            other != item and other.endswith("/" + item) for other in found
        )
    ][:_MAX_SYMBOLS]


def failure_query(test_output: str, root_cause: str = "") -> str:
    """Condense a failing test run into the terms most likely to locate code."""

    terms: list[str] = []
    for path, function in _TRACEBACK_FRAME.findall(test_output):
        normalized = path.replace("\\", "/")
        if "/site-packages/" in normalized or "/lib/python" in normalized:
            continue
        for term in (normalized.rsplit("/", 1)[-1], function):
            if term not in terms and term != "<module>":
                terms.append(term)
    for test_name, failure in _FAILURE_LINE.findall(test_output):
        line = (test_name or failure).strip()[:200]
        if line and line not in terms:
            terms.append(line)
    if root_cause:
        terms.append(root_cause.strip()[:400])
    return "\n".join(terms)[:_MAX_QUERY_CHARS]


def plan_queries(
    issue: str,
    constraints: Sequence[str] = (),
    *,
    last_failure: dict[str, Any] | None = None,
) -> list[PlannedQuery]:
    queries = [
        PlannedQuery(
            "issue",
            "\n".join([issue, *constraints])[:_MAX_QUERY_CHARS],
        )
    ]
    symbols = extract_code_symbols(issue)
    if symbols:
        queries.append(PlannedQuery("symbols", " ".join(symbols)))
    if last_failure:
        text = failure_query(
            str(last_failure.get("test_failure", "")),
            str(last_failure.get("root_cause", "")),
        )
        if text.strip():
            queries.append(PlannedQuery("failure", text))
    return queries


def fuse_rankings(
    rankings: Sequence[Sequence[Evidence]],
    *,
    top_k: int,
) -> list[Evidence]:
    """Reciprocal Rank Fusion by chunk, renumbering evidence IDs E1..En."""

    scores: dict[str, float] = {}
    first_seen: dict[str, Evidence] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item.chunk_id] = scores.get(item.chunk_id, 0.0) + 1.0 / (
                RRF_K + rank
            )
            first_seen.setdefault(item.chunk_id, item)
    ordered = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)
    return [
        first_seen[chunk_id].model_copy(update={"evidence_id": f"E{rank}"})
        for rank, chunk_id in enumerate(ordered[:top_k], start=1)
    ]
