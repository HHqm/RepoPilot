from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

from repopilot.config import Settings
from repopilot.models import ApprovalDecision, IssueRequest
from repopilot.service import RepoPilotService


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    cases = json.loads((PROJECT_ROOT / "evals" / "cases.json").read_text("utf-8"))
    results = []
    for case in cases:
        with tempfile.TemporaryDirectory(prefix="repopilot-eval-") as directory:
            root = Path(directory)
            source = PROJECT_ROOT / case["repository"]
            target = root / "repo"
            shutil.copytree(source, target)
            settings = Settings(
                deepseek_api_key=None,
                deepseek_base_url="https://api.deepseek.com",
                deepseek_model="deepseek-v4-pro",
                llm_mode="mock",
                max_files=100,
                max_file_bytes=200_000,
                rag_top_k=6,
                test_timeout_seconds=30,
                runtime_dir=root / "runtime",
                allow_host_test_execution=True,
            )
            with RepoPilotService(settings) as service:
                accepted = service.start(
                    IssueRequest(
                        repo_path=str(target),
                        issue=case["issue"],
                        test_command=case["test_command"],
                        llm_mode="mock",
                    )
                )
                run = service.wait(accepted.run_id, timeout_seconds=30)
                retrieval_hit = any(
                    item["path"] == case["expected_retrieval_path"]
                    for item in run.state["evidence"]
                )
                resumed = service.decide(
                    run.run_id,
                    ApprovalDecision(
                        approved=True,
                        comment="Automated eval fixture",
                        artifact_digest=run.interrupt["artifact_digest"],
                    ),
                )
                completed = service.wait(
                    resumed.run_id,
                    statuses={"completed", "failed", "failed_rolled_back"},
                    timeout_seconds=30,
                )
            results.append(
                {
                    "case_id": case["case_id"],
                    "retrieval_hit_at_k": retrieval_hit,
                    "proposal_generated": bool(run.state["proposals"]),
                    "review_passed": run.state["review"]["approved"],
                    "test_passed": completed.state["test_result"]["success"],
                    "task_completed": completed.status == "completed",
                }
            )
    summary = {
        "cases": results,
        "task_completion_rate": (
            sum(item["task_completed"] for item in results) / len(results)
        ),
        "retrieval_hit_rate": (
            sum(item["retrieval_hit_at_k"] for item in results) / len(results)
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if all(item["task_completed"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
