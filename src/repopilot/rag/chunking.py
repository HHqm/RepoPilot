from __future__ import annotations

import ast
import re
from hashlib import sha256

from .embeddings import multilingual_tokens
from .parsers import CODE_SUFFIXES, GENERIC_SYMBOL_PATTERNS
from .types import DocumentChunk, IndexedFile, ParsedBlock, ParsedDocument


HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


class StructureAwareChunker:
    def __init__(self, max_tokens: int = 512, overlap_tokens: int = 64):
        if max_tokens < 64:
            raise ValueError("RAG max chunk tokens must be at least 64")
        if overlap_tokens < 0 or overlap_tokens >= max_tokens:
            raise ValueError(
                "RAG chunk overlap must be non-negative and below max tokens"
            )
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(
        self,
        *,
        repo_id: str,
        indexed_file: IndexedFile,
        parsed: ParsedDocument,
    ) -> list[DocumentChunk]:
        suffix = "." + indexed_file.language.lower()
        if parsed.parser == "utf8-text-v2" and parsed.blocks:
            content = parsed.blocks[0].text
            if suffix == ".py":
                blocks = self._python_blocks(content)
            elif suffix == ".md":
                blocks = self._markdown_blocks(content)
            elif suffix in CODE_SUFFIXES:
                blocks = self._generic_code_blocks(content)
            else:
                blocks = self._split_block(parsed.blocks[0])
        else:
            blocks = [
                split
                for block in parsed.blocks
                for split in self._split_block(block)
            ]

        chunks: list[DocumentChunk] = []
        for chunk_index, block in enumerate(blocks, start=1):
            text = block.text.strip()
            if not text:
                continue
            content_hash = sha256(text.encode("utf-8")).hexdigest()
            chunk_id = sha256(
                (
                    f"{indexed_file.document_id}:{chunk_index}:"
                    f"{content_hash}:{block.page_start}:{block.start_line}"
                ).encode("utf-8")
            ).hexdigest()
            context = [f"path: {indexed_file.path}"]
            if block.heading_path:
                context.append("section: " + " > ".join(block.heading_path))
            if block.symbol:
                context.append(f"symbol: {block.symbol}")
            contextualized = "\n".join([*context, "", text])
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=indexed_file.document_id,
                    repo_id=repo_id,
                    path=indexed_file.path,
                    text=text,
                    contextualized_text=contextualized,
                    content_hash=content_hash,
                    document_version=indexed_file.sha256,
                    source_type=indexed_file.source_type,
                    language=indexed_file.language,
                    parser=indexed_file.parser,
                    chunk_index=chunk_index,
                    start_line=block.start_line,
                    end_line=block.end_line,
                    page_start=block.page_start,
                    page_end=block.page_end,
                    heading_path=block.heading_path,
                    symbol=block.symbol,
                )
            )
        return chunks

    def _python_blocks(self, content: str) -> list[ParsedBlock]:
        lines = content.splitlines()
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return self._split_lines(lines, 1)
        nodes = [
            node
            for node in tree.body
            if isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
            )
        ]
        if not nodes:
            return self._split_lines(lines, 1)

        blocks: list[ParsedBlock] = []
        cursor = 1
        for node in nodes:
            start = min(
                (item.lineno for item in node.decorator_list),
                default=node.lineno,
            )
            end = int(getattr(node, "end_lineno", node.lineno))
            if cursor < start:
                blocks.extend(
                    self._split_lines(
                        lines[cursor - 1 : start - 1],
                        cursor,
                    )
                )
            node_lines = lines[start - 1 : end]
            blocks.extend(
                self._split_lines(
                    node_lines,
                    start,
                    symbol=node.name,
                )
            )
            cursor = end + 1
        if cursor <= len(lines):
            blocks.extend(self._split_lines(lines[cursor - 1 :], cursor))
        return blocks

    def _generic_code_blocks(self, content: str) -> list[ParsedBlock]:
        lines = content.splitlines()
        boundaries: list[tuple[int, str]] = []
        for line_number, line in enumerate(lines, start=1):
            for pattern in GENERIC_SYMBOL_PATTERNS:
                match = pattern.match(line)
                if match:
                    boundaries.append((line_number, match.group(1)))
                    break
        if not boundaries:
            return self._split_lines(lines, 1)
        blocks: list[ParsedBlock] = []
        if boundaries[0][0] > 1:
            blocks.extend(
                self._split_lines(lines[: boundaries[0][0] - 1], 1)
            )
        for index, (start, symbol) in enumerate(boundaries):
            end = (
                boundaries[index + 1][0] - 1
                if index + 1 < len(boundaries)
                else len(lines)
            )
            blocks.extend(
                self._split_lines(lines[start - 1 : end], start, symbol=symbol)
            )
        return blocks

    def _markdown_blocks(self, content: str) -> list[ParsedBlock]:
        lines = content.splitlines()
        headings: list[str] = []
        groups: list[ParsedBlock] = []
        current: list[str] = []
        start_line = 1
        current_heading: tuple[str, ...] = ()

        def flush(end_line: int) -> None:
            nonlocal current
            if current:
                groups.append(
                    ParsedBlock(
                        text="\n".join(current),
                        start_line=start_line,
                        end_line=end_line,
                        heading_path=current_heading,
                    )
                )
                current = []

        for line_number, line in enumerate(lines, start=1):
            match = HEADING_PATTERN.match(line)
            if match:
                flush(line_number - 1)
                depth = len(match.group(1))
                headings = headings[: depth - 1]
                headings.append(match.group(2).strip())
                current_heading = tuple(headings)
                start_line = line_number
            current.append(line)
        flush(len(lines))
        return [
            split
            for group in groups
            for split in self._split_block(group)
        ]

    def _split_block(self, block: ParsedBlock) -> list[ParsedBlock]:
        lines = block.text.splitlines() or [""]
        base_line = block.start_line or 1
        return self._split_lines(
            lines,
            base_line,
            page_start=block.page_start,
            page_end=block.page_end,
            heading_path=block.heading_path,
            symbol=block.symbol,
        )

    def _split_lines(
        self,
        lines: list[str],
        base_line: int,
        *,
        page_start: int | None = None,
        page_end: int | None = None,
        heading_path: tuple[str, ...] = (),
        symbol: str | None = None,
    ) -> list[ParsedBlock]:
        if not lines:
            return []
        blocks: list[ParsedBlock] = []
        cursor = 0
        while cursor < len(lines):
            end = cursor
            token_count = 0
            while end < len(lines):
                line_tokens = max(1, len(multilingual_tokens(lines[end])))
                if end > cursor and token_count + line_tokens > self.max_tokens:
                    break
                token_count += line_tokens
                end += 1
            if end == cursor:
                end += 1
            text = "\n".join(lines[cursor:end])
            blocks.append(
                ParsedBlock(
                    text=text,
                    start_line=base_line + cursor if page_start is None else None,
                    end_line=base_line + end - 1 if page_start is None else None,
                    page_start=page_start,
                    page_end=page_end,
                    heading_path=heading_path,
                    symbol=symbol,
                )
            )
            if end >= len(lines):
                break
            overlap = 0
            next_cursor = end
            while next_cursor > cursor + 1 and overlap < self.overlap_tokens:
                next_cursor -= 1
                overlap += max(1, len(multilingual_tokens(lines[next_cursor])))
            cursor = next_cursor
        return blocks
