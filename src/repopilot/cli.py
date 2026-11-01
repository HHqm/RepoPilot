from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from .config import Settings
from .models import ApprovalDecision, IssueRequest
from .rag import HybridRAG
from .service import RepoPilotService


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _print_run(run) -> None:
    output = {
        "run_id": run.run_id,
        "status": run.status,
        "review": run.state.get("review"),
        "proposals": [
            {"path": item["path"], "diff": item["diff"]}
            for item in run.state.get("proposals", [])
        ],
        "test_result": run.state.get("test_result"),
        "verification": run.state.get("verification"),
        "interrupt": run.interrupt,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


def command_run(args: argparse.Namespace) -> int:
    settings = Settings.from_env(args.runtime_dir)
    with RepoPilotService(settings) as service:
        accepted = service.start(
            IssueRequest(
                repo_path=args.repo,
                issue=args.issue,
                constraints=args.constraint,
                test_command=args.test_command,
                llm_mode=args.llm,
            )
        )
        run = service.wait(accepted.run_id, timeout_seconds=300)
        _print_run(run)
    return 0


def command_decide(args: argparse.Namespace) -> int:
    settings = Settings.from_env(args.runtime_dir)
    with RepoPilotService(settings) as service:
        checkpoint = service.get(args.run_id)
        resumed = service.decide(
            args.run_id,
            ApprovalDecision(
                approved=args.approve,
                comment=args.comment,
                actor_id="cli-user",
                artifact_digest=(
                    checkpoint.interrupt or {}
                ).get("artifact_digest"),
            ),
        )
        # A failed test can trigger a repair attempt whose new patch stops at
        # the approval gate again, so wait for any terminal or human state.
        run = service.wait(resumed.run_id, timeout_seconds=300)
        _print_run(run)
        if run.status == "awaiting_approval":
            print(
                "A repair attempt produced a new patch; review it and run "
                f"`repopilot decide --run-id {run.run_id} --approve` again."
            )
    return 0 if run.status in {"completed", "cancelled", "awaiting_approval"} else 1


def command_demo(args: argparse.Namespace) -> int:
    source = _project_root() / "examples" / "buggy_calculator"
    demo_parent = Path(tempfile.mkdtemp(prefix="repopilot-demo-"))
    target = demo_parent / "buggy_calculator"
    shutil.copytree(source, target)
    runtime = demo_parent / ".repopilot"
    base = Settings.from_env(runtime)
    # The safety defaults protect arbitrary user repositories. The demo runs
    # RepoPilot's own bundled fixture in a private temp copy, so it trusts
    # exactly that directory and nothing else.
    settings = replace(
        base,
        allowed_workspace_roots=(*base.allowed_workspace_roots, demo_parent),
        allow_host_test_execution=True,
    )
    with RepoPilotService(settings) as service:
        accepted = service.start(
            IssueRequest(
                repo_path=str(target),
                issue=(
                    "修复 safe_divide 在除数为 0 时错误地抛出 "
                    "ZeroDivisionError；应明确抛出 ValueError。"
                ),
                constraints=["保持 safe_divide 的公开函数签名不变"],
                test_command="python -m unittest discover -s tests",
                llm_mode="mock",
            )
        )
        run = service.wait(accepted.run_id, timeout_seconds=60)
        while run.status == "awaiting_approval":
            print("=== Proposal (repository is still unchanged) ===")
            _print_run(run)
            service.decide(
                run.run_id,
                ApprovalDecision(
                    approved=True,
                    comment="Approved by offline demo.",
                    actor_id="offline-demo",
                    artifact_digest=run.interrupt["artifact_digest"],
                ),
            )
            run = service.wait(run.run_id, timeout_seconds=60)
        print("=== Result after approval ===")
        _print_run(run)
    print(f"Demo workspace: {target}")
    return 0 if run.status == "completed" else 1


def command_rag_index(args: argparse.Namespace) -> int:
    settings = Settings.from_env(args.runtime_dir)
    rag = HybridRAG.from_settings(settings)
    try:
        index = rag.index_repository(args.repo)
        print(
            json.dumps(
                {
                    "repo_id": index.repo_id,
                    "collection": index.collection_name,
                    "files": len(index.files),
                    "chunks": index.chunk_count,
                    "new_or_changed_chunks": index.indexed_chunks,
                    "reused_files": index.reused_files,
                    "deleted_files": index.deleted_files,
                    "skipped_files": index.skipped_files,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        rag.close()
    return 0


def command_rag_search(args: argparse.Namespace) -> int:
    settings = Settings.from_env(args.runtime_dir)
    rag = HybridRAG.from_settings(settings)
    try:
        index = rag.index_repository(args.repo)
        source_types = None if args.source_type == "all" else [args.source_type]
        evidence = rag.search(
            index,
            args.query,
            top_k=args.top_k,
            source_types=source_types,
        )
        print(
            json.dumps(
                [item.model_dump() for item in evidence],
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        rag.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="repopilot")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Create a reviewed patch proposal")
    run_parser.add_argument("--repo", required=True)
    run_parser.add_argument("--issue", required=True)
    run_parser.add_argument("--constraint", action="append", default=[])
    run_parser.add_argument(
        "--test-command", default="python -m unittest discover -s tests"
    )
    run_parser.add_argument("--llm", choices=["mock", "deepseek"], default=None)
    run_parser.add_argument("--runtime-dir", default=".repopilot")
    run_parser.set_defaults(func=command_run)

    decide_parser = subparsers.add_parser(
        "decide", help="Resume an interrupted run with a human decision"
    )
    decide_parser.add_argument("--run-id", required=True)
    decision = decide_parser.add_mutually_exclusive_group(required=True)
    decision.add_argument("--approve", action="store_true")
    decision.add_argument("--reject", dest="approve", action="store_false")
    decide_parser.add_argument("--comment", default="")
    decide_parser.add_argument("--runtime-dir", default=".repopilot")
    decide_parser.set_defaults(func=command_decide)

    demo_parser = subparsers.add_parser("demo", help="Run the safe offline demo")
    demo_parser.set_defaults(func=command_demo)

    index_parser = subparsers.add_parser(
        "rag-index",
        help="Incrementally parse and index repository code and documents",
    )
    index_parser.add_argument("--repo", required=True)
    index_parser.add_argument("--runtime-dir", default=".repopilot")
    index_parser.set_defaults(func=command_rag_index)

    search_parser = subparsers.add_parser(
        "rag-search",
        help="Run dense+sparse RRF retrieval followed by reranking",
    )
    search_parser.add_argument("--repo", required=True)
    search_parser.add_argument("--query", required=True)
    search_parser.add_argument("--top-k", type=int, default=8)
    search_parser.add_argument(
        "--source-type",
        choices=["all", "code", "document"],
        default="all",
    )
    search_parser.add_argument("--runtime-dir", default=".repopilot")
    search_parser.set_defaults(func=command_rag_search)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        raise SystemExit(args.func(args))
    except Exception as exc:
        print(f"RepoPilot error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
