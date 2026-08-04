from __future__ import annotations

import argparse
import os
from pathlib import Path
from threading import RLock

from mcp.server.fastmcp import FastMCP

from .config import Settings
from .rag import HybridRAG
from .security import resolve_safe_path


def create_server(repo_path: str) -> FastMCP:
    settings = Settings.from_env()
    rag = HybridRAG.from_settings(settings)
    index = rag.index_repository(repo_path)
    index_lock = RLock()
    server = FastMCP(
        "RepoPilot Read-Only Repository Tools",
        instructions=(
            "Read-only repository map, hybrid code search, and bounded file reads. "
            "Writing and test execution require the main approval workflow."
        ),
    )

    @server.tool(description="Return the safe repository map and extracted symbols.")
    def repo_map() -> list[dict]:
        nonlocal index
        with index_lock:
            index = rag.index_repository(index.root)
            return index.repo_map()

    @server.tool(
        description="Hybrid lexical/vector search over indexed code and documents."
    )
    def search_code(query: str, top_k: int = 8) -> list[dict]:
        nonlocal index
        bounded = max(1, min(top_k, 20))
        with index_lock:
            index = rag.index_repository(index.root)
            return [
                item.model_dump()
                for item in rag.search(index, query, top_k=bounded)
            ]

    @server.tool(
        description="Read a bounded line range from a safe repository text file."
    )
    def read_file(path: str, start_line: int = 1, end_line: int = 200) -> dict:
        target = resolve_safe_path(index.root, path)
        start = max(1, start_line)
        end = max(start, min(end_line, start + 499))
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        content = "\n".join(lines[start - 1 : end])
        encoded = content.encode("utf-8")
        truncated = len(encoded) > 64_000
        if truncated:
            content = encoded[:64_000].decode("utf-8", errors="ignore")
        return {
            "path": path,
            "start_line": start,
            "end_line": min(end, len(lines)),
            "content": content,
            "truncated": truncated,
        }

    return server


def main() -> None:
    parser = argparse.ArgumentParser(prog="repopilot-mcp")
    parser.add_argument("--repo", required=True)
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
    )
    args = parser.parse_args()
    if (
        args.transport != "stdio"
        and os.getenv("REPOPILOT_MCP_ALLOW_REMOTE", "0").strip().lower()
        not in {"1", "true", "yes", "on"}
    ):
        parser.error(
            "HTTP MCP transports are disabled by default. Set "
            "REPOPILOT_MCP_ALLOW_REMOTE=1 only behind an authenticated gateway."
        )
    repo = str(Path(args.repo).resolve())
    create_server(repo).run(transport=args.transport)


if __name__ == "__main__":
    main()
