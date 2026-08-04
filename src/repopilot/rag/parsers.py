from __future__ import annotations

import ast
import re
from pathlib import Path

from .types import ParsedBlock, ParsedDocument


CODE_SUFFIXES = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".go",
    ".rs",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
}
TEXT_SUFFIXES = {
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".xml",
    ".csv",
    ".sql",
    ".sh",
    ".ps1",
}
RICH_DOCUMENT_SUFFIXES = {
    ".pdf",
    ".docx",
    ".doc",
    ".pptx",
    ".ppt",
    ".xlsx",
    ".xls",
    ".html",
    ".htm",
    ".odt",
}
ALLOWED_SUFFIXES = CODE_SUFFIXES | TEXT_SUFFIXES | RICH_DOCUMENT_SUFFIXES
SECRET_PATTERNS = (
    re.compile(r"(?i)\bsk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
        r".*?-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        re.DOTALL,
    ),
    re.compile(
        r"(?i)\b(api[_-]?key|access[_-]?token|client[_-]?secret|password)"
        r"(\s*[:=]\s*)[\"']?[A-Za-z0-9_./+=-]{12,}[\"']?"
    ),
)


class DocumentParsingError(RuntimeError):
    pass


def redact_document_secrets(
    document: ParsedDocument,
) -> tuple[ParsedDocument, int]:
    count = 0
    blocks = []
    for block in document.blocks:
        text = block.text
        for pattern in SECRET_PATTERNS:
            if pattern.groups >= 2:
                text, matches = pattern.subn(
                    lambda match: (
                        f"{match.group(1)}{match.group(2)}[REDACTED]"
                    ),
                    text,
                )
            else:
                text, matches = pattern.subn("[REDACTED]", text)
            count += matches
        blocks.append(
            ParsedBlock(
                text=text,
                start_line=block.start_line,
                end_line=block.end_line,
                page_start=block.page_start,
                page_end=block.page_end,
                heading_path=block.heading_path,
                symbol=block.symbol,
            )
        )
    return (
        ParsedDocument(
            parser=document.parser,
            source_type=document.source_type,
            language=document.language,
            blocks=tuple(blocks),
            symbols=document.symbols,
        ),
        count,
    )


def _python_symbols(content: str) -> tuple[str, ...]:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return ()
    return tuple(
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    )


def parse_text_document(path: Path, raw: bytes) -> ParsedDocument:
    content = raw.decode("utf-8", errors="replace")
    suffix = path.suffix.lower()
    source_type = "code" if suffix in CODE_SUFFIXES else "document"
    symbols = _python_symbols(content) if suffix == ".py" else ()
    line_count = max(1, len(content.splitlines()))
    return ParsedDocument(
        parser="utf8-text-v2",
        source_type=source_type,
        language=suffix.lstrip(".") or "text",
        blocks=(
            ParsedBlock(
                text=content,
                start_line=1,
                end_line=line_count,
            ),
        ),
        symbols=symbols,
    )


def _docling_available() -> bool:
    try:
        import docling  # noqa: F401
    except ImportError:
        return False
    return True


class DoclingDocumentParser:
    """Structure-aware parser for PDF, Office, HTML, scans, and tables."""

    name = "docling-v2"

    def __init__(self, artifacts_path: Path | None = None):
        self._converter = None
        self.artifacts_path = (
            artifacts_path.expanduser().resolve()
            if artifacts_path is not None
            else None
        )

    def _load(self):
        if self._converter is not None:
            return self._converter
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter
            from docling.document_converter import PdfFormatOption
        except ImportError as exc:
            raise DocumentParsingError(
                "Rich documents require Docling. Install with: "
                "pip install -e \".[documents]\""
            ) from exc
        format_options = None
        if self.artifacts_path is not None:
            if not self.artifacts_path.is_dir():
                raise DocumentParsingError(
                    "Docling artifacts directory does not exist: "
                    f"{self.artifacts_path}"
                )
            pipeline_options = PdfPipelineOptions(
                artifacts_path=self.artifacts_path
            )
            format_options = {
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        self._converter = DocumentConverter(format_options=format_options)
        return self._converter

    def parse(
        self,
        path: Path,
        max_file_bytes: int,
        max_num_pages: int,
    ) -> ParsedDocument:
        try:
            result = self._load().convert(
                source=path,
                max_file_size=max_file_bytes,
                max_num_pages=max_num_pages,
                raises_on_error=True,
            )
        except Exception as exc:
            raise DocumentParsingError(
                f"Docling failed to parse {path.name}: {exc}"
            ) from exc

        document = result.document
        blocks: list[ParsedBlock] = []
        headings: list[str] = []
        for item, level in document.iterate_items():
            label = str(getattr(item, "label", "")).lower()
            text = getattr(item, "text", None)
            if not text and hasattr(item, "export_to_markdown"):
                try:
                    text = item.export_to_markdown(doc=document)
                except (TypeError, ValueError):
                    text = None
            text = str(text or "").strip()
            if not text:
                continue
            if "title" in label or "heading" in label:
                depth = max(1, int(level or 1))
                headings = headings[: depth - 1]
                headings.append(text)

            pages: list[int] = []
            for provenance in getattr(item, "prov", ()) or ():
                page_no = getattr(provenance, "page_no", None)
                if page_no is not None:
                    pages.append(int(page_no))
            blocks.append(
                ParsedBlock(
                    text=text,
                    page_start=min(pages) if pages else None,
                    page_end=max(pages) if pages else None,
                    heading_path=tuple(headings),
                )
            )
        if not blocks:
            markdown = document.export_to_markdown().strip()
            if markdown:
                blocks.append(ParsedBlock(text=markdown))
        if not blocks:
            raise DocumentParsingError(f"No readable content found in {path.name}")
        return ParsedDocument(
            parser=self.name,
            source_type="document",
            language=path.suffix.lower().lstrip("."),
            blocks=tuple(blocks),
        )


def parse_pdf_fallback(path: Path, max_num_pages: int) -> ParsedDocument:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise DocumentParsingError(
            "PDF parsing requires pypdf or the Docling document extra."
        ) from exc
    try:
        reader = PdfReader(str(path))
        if len(reader.pages) > max_num_pages:
            raise DocumentParsingError(
                f"{path.name} has {len(reader.pages)} pages; "
                f"the configured limit is {max_num_pages}"
            )
        blocks = []
        for page_index, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                blocks.append(
                    ParsedBlock(
                        text=text,
                        page_start=page_index,
                        page_end=page_index,
                    )
                )
    except Exception as exc:
        raise DocumentParsingError(
            f"PDF text extraction failed for {path.name}: {exc}"
        ) from exc
    if not blocks:
        raise DocumentParsingError(
            f"{path.name} contains no extractable text; install the Docling "
            "extra to enable OCR and layout parsing."
        )
    return ParsedDocument(
        parser="pypdf-text-fallback-v1",
        source_type="document",
        language="pdf",
        blocks=tuple(blocks),
    )


class DocumentLoader:
    def __init__(
        self,
        parser_mode: str = "auto",
        docling_artifacts_path: Path | None = None,
    ):
        if parser_mode not in {"auto", "docling", "text-only"}:
            raise ValueError(
                "RAG document parser must be auto, docling, or text-only"
            )
        self.parser_mode = parser_mode
        self._docling = DoclingDocumentParser(docling_artifacts_path)

    def parse(
        self,
        path: Path,
        raw: bytes,
        max_file_bytes: int,
        max_num_pages: int,
    ) -> ParsedDocument:
        suffix = path.suffix.lower()
        if suffix in CODE_SUFFIXES or suffix in TEXT_SUFFIXES:
            return parse_text_document(path, raw)
        if suffix not in RICH_DOCUMENT_SUFFIXES:
            raise DocumentParsingError(f"Unsupported file type: {suffix}")
        self._validate_signature(path, raw)
        if self.parser_mode == "text-only":
            if suffix == ".pdf":
                return parse_pdf_fallback(path, max_num_pages)
            raise DocumentParsingError(
                f"{suffix} is disabled by REPOPILOT_RAG_DOCUMENT_PARSER=text-only"
            )
        if self.parser_mode == "docling" or _docling_available():
            return self._docling.parse(
                path,
                max_file_bytes,
                max_num_pages,
            )
        if suffix == ".pdf":
            return parse_pdf_fallback(path, max_num_pages)
        raise DocumentParsingError(
            f"{suffix} requires Docling. Install with: "
            "pip install -e \".[documents]\""
        )

    @staticmethod
    def _validate_signature(path: Path, raw: bytes) -> None:
        suffix = path.suffix.lower()
        if suffix == ".pdf" and not raw.startswith(b"%PDF-"):
            raise DocumentParsingError("PDF signature does not match its suffix")
        office = {".docx", ".pptx", ".xlsx", ".odt"}
        if suffix in office and not raw.startswith(b"PK"):
            raise DocumentParsingError(
                f"{suffix} file is not a valid ZIP-based office document"
            )


GENERIC_SYMBOL_PATTERNS = [
    re.compile(
        r"^\s*(?:export\s+)?(?:async\s+)?(?:function|class|interface|type)\s+"
        r"([A-Za-z_$][\w$]*)"
    ),
    re.compile(
        r"^\s*(?:public|private|protected|static|final|async|\s)*"
        r"(?:class|interface|enum|struct|fn|func)\s+([A-Za-z_]\w*)"
    ),
]


def generic_code_symbols(content: str) -> tuple[str, ...]:
    symbols: list[str] = []
    for line in content.splitlines():
        for pattern in GENERIC_SYMBOL_PATTERNS:
            match = pattern.match(line)
            if match:
                symbols.append(match.group(1))
                break
    return tuple(dict.fromkeys(symbols))
