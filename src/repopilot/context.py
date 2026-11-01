from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .config import Settings


_CJK_PATTERN = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
    r"\u3040-\u30ff\uac00-\ud7af]"
)
_PATH_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_\-\u3400-\u9fff]+")


class ContextCompressionError(ValueError):
    """Raised when safety-critical context cannot fit without information loss."""


@dataclass
class _CompressionTracker:
    omitted_items: int = 0
    truncated_fields: int = 0


def estimate_text_tokens(text: str) -> int:
    """Conservative tokenizer-free estimate for mixed Chinese, prose, and code."""

    if not text:
        return 0
    cjk_count = len(_CJK_PATTERN.findall(text))
    non_cjk_count = max(0, len(text) - cjk_count)
    return max(1, math.ceil(cjk_count * 1.2 + non_cjk_count / 3.6))


def estimate_json_tokens(value: Any) -> int:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=False,
    )
    return estimate_text_tokens(serialized)


class ContextCompressor:
    """Build bounded, provenance-preserving payloads for each LLM agent.

    Compression is deterministic so it is testable and cannot invent facts.
    Paths, evidence IDs, hashes, plan structure, and every changed diff line are
    protected. Large prose, file excerpts, unchanged diff context, and test
    output are the only lossy fields.
    """

    _SCHEMA_BUDGET_KEYS = {
        "change_plan": "planner",
        "code_edits": "coder",
        "review_decision": "reviewer",
        "reflection": "reflection",
        "product_chat": "planner",
        "product_requirements": "planner",
        "requirements_workshop": "planner",
        "project_plan": "planner",
        "architecture_blueprint": "planner",
        "generated_project": "coder",
        "project_revision": "coder",
        "deployment_guide": "planner",
        "qa_report": "reviewer",
        "studio_review": "reviewer",
    }
    _SOURCE_COMPLETE_SCHEMAS = {
        "project_revision",
        "qa_report",
        "studio_review",
    }
    _PROTECTED_STRING_KEYS = {
        "path",
        "evidence_id",
        "chunk_id",
        "document_id",
        "original_sha256",
        "sha256",
        "step_id",
        "diffs",
    }

    def __init__(
        self,
        *,
        max_input_tokens: int,
        planner_tokens: int,
        coder_tokens: int,
        reviewer_tokens: int,
        reflection_tokens: int,
        max_evidence_item_tokens: int,
        max_file_item_tokens: int,
        max_diff_tokens: int,
        max_test_output_tokens: int,
    ) -> None:
        self.max_input_tokens = max(2_048, max_input_tokens)
        self.schema_budgets = {
            "planner": max(1_024, planner_tokens),
            "coder": max(1_024, coder_tokens),
            "reviewer": max(1_024, reviewer_tokens),
            "reflection": max(1_024, reflection_tokens),
        }
        self.max_evidence_item_tokens = max(128, max_evidence_item_tokens)
        self.max_file_item_tokens = max(256, max_file_item_tokens)
        self.max_diff_tokens = max(512, max_diff_tokens)
        self.max_test_output_tokens = max(256, max_test_output_tokens)

    @classmethod
    def from_settings(cls, settings: Settings) -> ContextCompressor:
        return cls(
            max_input_tokens=settings.context_max_input_tokens,
            planner_tokens=settings.context_planner_tokens,
            coder_tokens=settings.context_coder_tokens,
            reviewer_tokens=settings.context_reviewer_tokens,
            reflection_tokens=settings.context_reflection_tokens,
            max_evidence_item_tokens=(
                settings.context_max_evidence_item_tokens
            ),
            max_file_item_tokens=settings.context_max_file_item_tokens,
            max_diff_tokens=settings.context_max_diff_tokens,
            max_test_output_tokens=settings.context_max_test_output_tokens,
        )

    def compress(
        self,
        schema_name: str,
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        original_tokens = estimate_json_tokens(payload)
        budget_key = self._SCHEMA_BUDGET_KEYS.get(schema_name, "coder")
        budget_tokens = min(
            self.max_input_tokens,
            self.schema_budgets[budget_key],
        )
        # Reserve headroom for JSON keys and the stable user prompt wrapper.
        content_budget = max(768, int(budget_tokens * 0.86))
        tracker = _CompressionTracker()

        if schema_name == "change_plan":
            compressed = self._planner_payload(
                payload, content_budget, tracker
            )
        elif schema_name == "code_edits":
            compressed = self._coder_payload(payload, content_budget, tracker)
        elif schema_name == "review_decision":
            compressed = self._reviewer_payload(
                payload, content_budget, tracker
            )
        elif schema_name == "reflection":
            compressed = self._reflection_payload(
                payload, content_budget, tracker
            )
        else:
            compressed = deepcopy(payload)

        compressed_tokens = estimate_json_tokens(compressed)
        if compressed_tokens > budget_tokens:
            protected_keys = set(self._PROTECTED_STRING_KEYS)
            if schema_name in self._SOURCE_COMPLETE_SCHEMAS:
                protected_keys.add("content")
            compressed = self._shrink_noncritical_strings(
                compressed,
                budget_tokens,
                tracker,
                protected_keys=protected_keys,
            )
            compressed_tokens = estimate_json_tokens(compressed)
        if compressed_tokens > budget_tokens:
            raise ContextCompressionError(
                f"Context for {schema_name} requires {compressed_tokens} "
                f"estimated tokens after safe compression; budget is "
                f"{budget_tokens}. Increase REPOPILOT_CONTEXT_MAX_INPUT_TOKENS "
                "or reduce the requested change scope."
            )

        if tracker.omitted_items or tracker.truncated_fields:
            notice = {
                "deterministic_compression": True,
                "omitted_items": tracker.omitted_items,
                "truncated_fields": tracker.truncated_fields,
                "critical_ids_paths_and_diff_changes_preserved": True,
            }
            candidate = deepcopy(compressed)
            candidate["context_notice"] = notice
            candidate_tokens = estimate_json_tokens(candidate)
            if candidate_tokens <= budget_tokens:
                compressed = candidate
                compressed_tokens = candidate_tokens
            else:
                compact_candidate = deepcopy(compressed)
                compact_candidate["context_notice"] = (
                    "deterministically compressed; critical identifiers "
                    "and changed diff lines preserved"
                )
                compact_candidate_tokens = estimate_json_tokens(
                    compact_candidate
                )
                if compact_candidate_tokens <= budget_tokens:
                    compressed = compact_candidate
                    compressed_tokens = compact_candidate_tokens

        stats = {
            "schema": schema_name,
            "scope": "dynamic_payload_estimate",
            "budget_tokens": budget_tokens,
            "estimated_tokens_before": original_tokens,
            "estimated_tokens_after": compressed_tokens,
            "saved_tokens": max(0, original_tokens - compressed_tokens),
            "compression_ratio": round(
                compressed_tokens / max(1, original_tokens),
                4,
            ),
            "omitted_items": tracker.omitted_items,
            "truncated_fields": tracker.truncated_fields,
            "source_content_preserved": (
                schema_name in self._SOURCE_COMPLETE_SCHEMAS
            ),
        }
        return compressed, stats

    def _planner_payload(
        self,
        payload: dict[str, Any],
        budget: int,
        tracker: _CompressionTracker,
    ) -> dict[str, Any]:
        evidence = list(payload.get("evidence") or [])
        evidence_paths = {
            str(item.get("path", "")) for item in evidence if item.get("path")
        }
        attempts = payload.get("previous_attempts") or []
        # Failed attempts take their share from evidence: re-planning must see
        # why the last patch failed even if that costs some retrieved context.
        attempts_share = 0.15 if attempts else 0.0
        compact = {
            "issue": self._fit_text(
                str(payload.get("issue", "")),
                int(budget * 0.10),
                tracker,
            ),
            "constraints": self._fit_string_list(
                payload.get("constraints") or [],
                int(budget * 0.10),
                tracker,
            ),
            "repo_map": self._compact_repo_map(
                payload.get("repo_map") or [],
                int(budget * 0.25),
                evidence_paths,
                str(payload.get("issue", "")),
                tracker,
            ),
            "evidence": self._compact_evidence(
                evidence,
                int(budget * (0.55 - attempts_share)),
                tracker,
            ),
        }
        if attempts:
            compact["previous_attempts"] = self._fit_nested_value(
                attempts,
                int(budget * attempts_share),
                tracker,
            )
        return compact

    def _coder_payload(
        self,
        payload: dict[str, Any],
        budget: int,
        tracker: _CompressionTracker,
    ) -> dict[str, Any]:
        plan = payload.get("plan") or {}
        target_paths = {
            str(path)
            for step in plan.get("steps", [])
            for path in step.get("target_files", [])
        }
        evidence_ids = {
            str(evidence_id)
            for step in plan.get("steps", [])
            for evidence_id in step.get("evidence_ids", [])
        }
        feedback_keys = [
            key
            for key in ("review_feedback", "previous_attempts")
            if payload.get(key)
        ]
        # Feedback is carved out of the file budget, 6% per kind present.
        feedback_share = 0.06
        compact = {
            "issue": self._fit_text(
                str(payload.get("issue", "")),
                int(budget * 0.07),
                tracker,
            ),
            "constraints": self._fit_string_list(
                payload.get("constraints") or [],
                int(budget * 0.07),
                tracker,
            ),
            "plan": self._fit_nested_value(
                plan,
                int(budget * 0.12),
                tracker,
            ),
            "evidence": self._compact_evidence(
                payload.get("evidence") or [],
                int(budget * 0.20),
                tracker,
                priority_paths=target_paths,
                priority_ids=evidence_ids,
                omit_unrelated=True,
            ),
            "files": self._compact_files(
                payload.get("files") or [],
                int(budget * (0.54 - feedback_share * len(feedback_keys))),
                tracker,
            ),
        }
        for key in feedback_keys:
            compact[key] = self._fit_nested_value(
                payload[key],
                int(budget * feedback_share),
                tracker,
            )
        if payload.get("tool_error"):
            compact["tool_error"] = self._fit_text(
                str(payload["tool_error"]), 256, tracker
            )
        return compact

    def _reviewer_payload(
        self,
        payload: dict[str, Any],
        budget: int,
        tracker: _CompressionTracker,
    ) -> dict[str, Any]:
        plan = payload.get("plan") or {}
        target_paths = {
            str(path)
            for step in plan.get("steps", [])
            for path in step.get("target_files", [])
        }
        evidence_ids = {
            str(evidence_id)
            for step in plan.get("steps", [])
            for evidence_id in step.get("evidence_ids", [])
        }
        diff_budget = min(self.max_diff_tokens, int(budget * 0.48))
        return {
            "issue": self._fit_text(
                str(payload.get("issue", "")),
                int(budget * 0.08),
                tracker,
            ),
            "constraints": self._fit_string_list(
                payload.get("constraints") or [],
                int(budget * 0.08),
                tracker,
            ),
            "plan": self._fit_nested_value(
                plan,
                int(budget * 0.14),
                tracker,
            ),
            "evidence": self._compact_evidence(
                payload.get("evidence") or [],
                int(budget * 0.22),
                tracker,
                priority_paths=target_paths,
                priority_ids=evidence_ids,
                omit_unrelated=True,
            ),
            "diffs": self._compact_diffs(
                payload.get("diffs") or "",
                diff_budget,
                tracker,
            ),
        }

    def _reflection_payload(
        self,
        payload: dict[str, Any],
        budget: int,
        tracker: _CompressionTracker,
    ) -> dict[str, Any]:
        return {
            "issue": self._fit_text(
                str(payload.get("issue", "")),
                int(budget * 0.10),
                tracker,
            ),
            "plan": self._fit_nested_value(
                payload.get("plan") or {},
                int(budget * 0.16),
                tracker,
            ),
            "diffs": self._compact_diffs(
                payload.get("diffs") or [],
                min(self.max_diff_tokens, int(budget * 0.34)),
                tracker,
            ),
            "test_result": self._compact_test_result(
                payload.get("test_result") or {},
                min(self.max_test_output_tokens, int(budget * 0.30)),
                tracker,
            ),
            "verification": self._fit_nested_value(
                payload.get("verification") or {},
                int(budget * 0.10),
                tracker,
            ),
        }

    def _compact_repo_map(
        self,
        repo_map: Iterable[dict[str, Any]],
        budget: int,
        evidence_paths: set[str],
        issue: str,
        tracker: _CompressionTracker,
    ) -> list[dict[str, Any]]:
        issue_terms = {
            term.lower() for term in _PATH_TOKEN_PATTERN.findall(issue)
        }
        items = list(repo_map)

        def priority(item: dict[str, Any]) -> tuple[int, int, str]:
            path = str(item.get("path", ""))
            path_terms = {
                term.lower() for term in _PATH_TOKEN_PATTERN.findall(path)
            }
            return (
                0 if path in evidence_paths else 1,
                -len(issue_terms & path_terms),
                path,
            )

        compacted: list[dict[str, Any]] = []
        used = 0
        for item in sorted(items, key=priority):
            compact = {
                key: deepcopy(item[key])
                for key in (
                    "path",
                    "language",
                    "source_type",
                    "parser",
                    "size",
                    "chunks",
                    "redactions",
                )
                if key in item
            }
            symbols = [str(value) for value in item.get("symbols", [])]
            if len(symbols) > 32:
                tracker.omitted_items += len(symbols) - 32
                symbols = symbols[:32]
            if symbols:
                compact["symbols"] = symbols
            tokens = estimate_json_tokens(compact)
            if compacted and used + tokens > max(64, budget):
                continue
            compacted.append(compact)
            used += tokens
        tracker.omitted_items += max(0, len(items) - len(compacted))
        return compacted

    def _compact_evidence(
        self,
        evidence: Iterable[dict[str, Any]],
        budget: int,
        tracker: _CompressionTracker,
        *,
        priority_paths: set[str] | None = None,
        priority_ids: set[str] | None = None,
        omit_unrelated: bool = False,
    ) -> list[dict[str, Any]]:
        priority_paths = priority_paths or set()
        priority_ids = priority_ids or set()
        items = list(evidence)

        def is_related(item: dict[str, Any]) -> bool:
            return (
                str(item.get("path", "")) in priority_paths
                or str(item.get("evidence_id", "")) in priority_ids
            )

        if omit_unrelated and (priority_paths or priority_ids):
            related = [item for item in items if is_related(item)]
            tracker.omitted_items += len(items) - len(related)
            items = related
        items.sort(
            key=lambda item: (
                0 if is_related(item) else 1,
                -float(item.get("rerank_score", item.get("score", 0.0))),
                str(item.get("evidence_id", "")),
            )
        )
        if not items:
            return []

        per_item = min(
            self.max_evidence_item_tokens,
            max(128, budget // max(1, len(items))),
        )
        compacted: list[dict[str, Any]] = []
        used = 0
        for item in items:
            compact = {
                key: deepcopy(item[key])
                for key in (
                    "evidence_id",
                    "chunk_id",
                    "document_id",
                    "path",
                    "source_type",
                    "trust_level",
                    "document_version",
                    "language",
                    "start_line",
                    "end_line",
                    "page_start",
                    "page_end",
                    "heading_path",
                    "symbol",
                    "fusion_score",
                    "rerank_score",
                    "score",
                )
                if key in item
            }
            metadata_tokens = estimate_json_tokens(compact)
            text_budget = max(48, per_item - metadata_tokens)
            compact["text"] = self._fit_text(
                str(item.get("text", "")),
                text_budget,
                tracker,
            )
            tokens = estimate_json_tokens(compact)
            if compacted and used + tokens > max(128, budget):
                tracker.omitted_items += 1
                continue
            compacted.append(compact)
            used += tokens
        return compacted

    def _compact_files(
        self,
        files: Iterable[dict[str, Any]],
        budget: int,
        tracker: _CompressionTracker,
    ) -> list[dict[str, Any]]:
        items = list(files)
        if not items:
            return []
        per_file = min(
            self.max_file_item_tokens,
            max(256, budget // len(items)),
        )
        compacted: list[dict[str, Any]] = []
        for item in items:
            excerpts = [str(value) for value in item.get("excerpts", [])]
            excerpt_budget = max(128, per_file - 64)
            per_excerpt = max(
                64,
                excerpt_budget // max(1, len(excerpts)),
            )
            compacted.append(
                {
                    "path": str(item.get("path", "")),
                    "size": item.get("size"),
                    "excerpts": [
                        self._fit_text(
                            excerpt,
                            per_excerpt,
                            tracker,
                            head_ratio=0.72,
                        )
                        for excerpt in excerpts
                    ],
                }
            )
        return compacted

    def _compact_diffs(
        self,
        diffs: str | Iterable[str],
        budget: int,
        tracker: _CompressionTracker,
    ) -> str | list[str]:
        if isinstance(diffs, str):
            return self._compress_unified_diff(diffs, budget, tracker)
        values = [str(value) for value in diffs]
        if not values:
            return []
        estimates = [max(1, estimate_text_tokens(value)) for value in values]
        total = sum(estimates)
        result = []
        for value, estimate in zip(values, estimates, strict=True):
            item_budget = max(128, int(budget * estimate / total))
            result.append(
                self._compress_unified_diff(value, item_budget, tracker)
            )
        return result

    def _compress_unified_diff(
        self,
        diff: str,
        budget: int,
        tracker: _CompressionTracker,
    ) -> str:
        if estimate_text_tokens(diff) <= budget:
            return diff
        lines = diff.splitlines()
        critical = {
            index
            for index, line in enumerate(lines)
            if line.startswith(("diff --git ", "--- ", "+++ ", "@@", "+", "-"))
        }
        keep = set(critical)
        for index in critical:
            if index > 0:
                keep.add(index - 1)
            if index + 1 < len(lines):
                keep.add(index + 1)

        compressed_lines: list[str] = []
        previous = -2
        for index in sorted(keep):
            if index > previous + 1:
                compressed_lines.append(
                    "... [unchanged diff context compressed] ..."
                )
            compressed_lines.append(lines[index])
            previous = index
        compressed = "\n".join(compressed_lines)
        if diff.endswith("\n"):
            compressed += "\n"
        if estimate_text_tokens(compressed) > budget:
            critical_only = "\n".join(
                lines[index] for index in sorted(critical)
            )
            if estimate_text_tokens(critical_only) > budget:
                raise ContextCompressionError(
                    "All added/removed diff lines cannot fit within the "
                    f"{budget}-token reviewer budget; refusing a blind review."
                )
            compressed = critical_only + ("\n" if diff.endswith("\n") else "")
        tracker.truncated_fields += 1
        tracker.omitted_items += max(0, len(lines) - len(keep))
        return compressed

    def _compact_test_result(
        self,
        result: dict[str, Any],
        budget: int,
        tracker: _CompressionTracker,
    ) -> dict[str, Any]:
        compact = {
            key: deepcopy(value)
            for key, value in result.items()
            if key not in {"stdout", "stderr"}
        }
        output_budget = max(64, budget - estimate_json_tokens(compact))
        stdout = str(result.get("stdout", ""))
        stderr = str(result.get("stderr", ""))
        compact["stdout"] = self._fit_text(
            stdout,
            output_budget // 2,
            tracker,
            head_ratio=0.18,
        )
        compact["stderr"] = self._fit_text(
            stderr,
            output_budget - output_budget // 2,
            tracker,
            head_ratio=0.12,
        )
        return compact

    def _fit_string_list(
        self,
        values: Iterable[Any],
        budget: int,
        tracker: _CompressionTracker,
    ) -> list[str]:
        items = [str(value) for value in values]
        if not items:
            return []
        per_item = max(24, budget // len(items))
        compacted = [
            self._fit_text(value, per_item, tracker) for value in items
        ]
        while len(compacted) > 1 and estimate_json_tokens(compacted) > budget:
            compacted.pop()
            tracker.omitted_items += 1
        return compacted

    def _fit_nested_value(
        self,
        value: Any,
        budget: int,
        tracker: _CompressionTracker,
    ) -> Any:
        cloned = deepcopy(value)
        current = estimate_json_tokens(cloned)
        if current <= budget:
            return cloned
        ratio = max(0.08, budget / max(1, current))
        return self._scale_strings(cloned, ratio, tracker)

    def _shrink_noncritical_strings(
        self,
        payload: dict[str, Any],
        budget: int,
        tracker: _CompressionTracker,
        *,
        protected_keys: set[str] | None = None,
    ) -> dict[str, Any]:
        current = estimate_json_tokens(payload)
        compressed = deepcopy(payload)
        for _ in range(5):
            if current <= budget:
                break
            ratio = max(0.08, (budget * 0.92) / max(1, current))
            compressed = self._scale_strings(
                compressed,
                ratio,
                tracker,
                protected_keys=protected_keys,
            )
            next_size = estimate_json_tokens(compressed)
            if next_size >= current:
                break
            current = next_size
        return compressed

    def _scale_strings(
        self,
        value: Any,
        ratio: float,
        tracker: _CompressionTracker,
        *,
        key: str | None = None,
        protected_keys: set[str] | None = None,
    ) -> Any:
        protected_keys = protected_keys or self._PROTECTED_STRING_KEYS
        if isinstance(value, str):
            if key in protected_keys:
                return value
            tokens = estimate_text_tokens(value)
            if tokens <= 32:
                return value
            return self._fit_text(
                value,
                max(24, int(tokens * ratio)),
                tracker,
            )
        if isinstance(value, list):
            return [
                self._scale_strings(
                    item,
                    ratio,
                    tracker,
                    key=key,
                    protected_keys=protected_keys,
                )
                for item in value
            ]
        if isinstance(value, dict):
            return {
                item_key: self._scale_strings(
                    item_value,
                    ratio,
                    tracker,
                    key=item_key,
                    protected_keys=protected_keys,
                )
                for item_key, item_value in value.items()
            }
        return value

    def _fit_text(
        self,
        text: str,
        budget: int,
        tracker: _CompressionTracker,
        *,
        head_ratio: float = 0.68,
    ) -> str:
        budget = max(8, budget)
        if estimate_text_tokens(text) <= budget:
            return text
        marker = "\n... [context compressed deterministically] ...\n"
        low = 0
        high = len(text)
        best = marker
        while low <= high:
            keep = (low + high) // 2
            head = int(keep * head_ratio)
            tail = keep - head
            candidate = (
                text[:head]
                + marker
                + (text[-tail:] if tail else "")
            )
            if estimate_text_tokens(candidate) <= budget:
                best = candidate
                low = keep + 1
            else:
                high = keep - 1
        tracker.truncated_fields += 1
        return best
