"""Aggregate cost and latency for a run from its agent traces.

Every model call goes through ``AgentSuite._complete_json``, which records a
context-compression stats dict (``schema`` + ``estimated_tokens_before``) and,
for real providers, ``api_usage``. Repairs nest their own stats, so walking
the trace metadata counts every call exactly once, in mock mode too.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any


def _walk(value: Any, calls: list[dict[str, Any]], usage: Counter) -> None:
    if isinstance(value, dict):
        if "schema" in value and "estimated_tokens_before" in value:
            calls.append(value)
        api_usage = value.get("api_usage")
        if isinstance(api_usage, dict):
            usage.update(
                {
                    key: int(amount)
                    for key, amount in api_usage.items()
                    if isinstance(amount, int | float)
                }
            )
        for key, child in value.items():
            if key != "api_usage":
                _walk(child, calls, usage)
    elif isinstance(value, list):
        for child in value:
            _walk(child, calls, usage)


def summarize_run_metrics(traces: Iterable[dict[str, Any]]) -> dict[str, Any]:
    calls: list[dict[str, Any]] = []
    usage: Counter = Counter()
    latency: Counter = Counter()
    for trace in traces:
        latency[str(trace.get("agent", "unknown"))] += float(
            trace.get("latency_ms") or 0.0
        )
        _walk(trace.get("metadata") or {}, calls, usage)
    by_schema = Counter(str(call["schema"]) for call in calls)
    return {
        "llm_calls": len(calls),
        "llm_calls_by_schema": dict(sorted(by_schema.items())),
        "estimated_input_tokens": sum(
            int(call.get("estimated_tokens_after", 0)) for call in calls
        ),
        "token_usage": dict(sorted(usage.items())),
        "agent_latency_ms": {
            agent: round(total, 2)
            for agent, total in sorted(
                latency.items(), key=lambda item: item[1], reverse=True
            )
        },
        "total_agent_latency_ms": round(sum(latency.values()), 2),
    }


def render_run_metrics(metrics: dict[str, Any]) -> str:
    usage = metrics.get("token_usage") or {}
    usage_text = (
        ", ".join(f"{key}={value}" for key, value in usage.items())
        if usage
        else "provider did not report usage (mock or unsupported)"
    )
    slowest = list((metrics.get("agent_latency_ms") or {}).items())[:5]
    return "\n".join(
        [
            f"- LLM calls: {metrics.get('llm_calls', 0)} "
            f"({', '.join(f'{k}×{v}' for k, v in metrics.get('llm_calls_by_schema', {}).items()) or 'none'})",
            f"- Estimated input tokens after compression: "
            f"{metrics.get('estimated_input_tokens', 0)}",
            f"- Provider token usage: {usage_text}",
            f"- Agent time: {metrics.get('total_agent_latency_ms', 0)} ms; slowest: "
            + (
                ", ".join(f"{agent} {ms} ms" for agent, ms in slowest)
                or "none"
            ),
        ]
    )
