from __future__ import annotations

import re
from pathlib import Path


SENSITIVE_PARTS = {
    ".git",
    ".env",
    ".ssh",
    ".aws",
    ".repopilot",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
}
SENSITIVE_BASENAMES = {
    "id_rsa",
    "id_ed25519",
    "credentials",
    "credentials.json",
    "secrets.json",
}
EXECUTABLE_CODE_PATTERNS = {
    r"\beval\s*\(": "Use of eval()",
    r"\bexec\s*\(": "Use of exec()",
    r"(?i)shell\s*=\s*true": "Subprocess shell=True",
    r"\bos\.system\s*\(": "Use of os.system()",
    r"\bpickle\.loads?\s*\(": "Unsafe pickle deserialization",
    r"(?i)\bverify\s*=\s*false": "TLS verification disabled",
    r"(?i)verify_signature[\"']?\s*[:=]\s*false": "JWT signature verification disabled",
    r"(?i)pytest\.skip|unittest\.skip": "Disabling tests",
}
UNIVERSAL_SECRET_PATTERNS = {
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----": "Embedded private key",
    r"(?i)\bsk-[A-Za-z0-9_-]{16,}": "Possible API key",
    r"\bAKIA[0-9A-Z]{16}\b": "Possible AWS access key",
}
DOCUMENTATION_SUFFIXES = {".md", ".mdx", ".rst", ".txt"}

# Backward-compatible aggregate for callers that inspect the pattern registry.
DANGEROUS_PATTERNS = {
    **EXECUTABLE_CODE_PATTERNS,
    **UNIVERSAL_SECRET_PATTERNS,
}


def resolve_safe_path(
    repo_root: str | Path, relative_path: str, *, must_exist: bool = True
) -> Path:
    root = Path(repo_root).resolve()
    candidate_input = Path(relative_path)
    if candidate_input.is_absolute():
        raise PermissionError("Absolute paths are not allowed")
    if any(part.lower() in SENSITIVE_PARTS for part in candidate_input.parts):
        raise PermissionError(f"Sensitive path is not allowed: {relative_path}")
    if candidate_input.name.lower() in SENSITIVE_BASENAMES:
        raise PermissionError(f"Sensitive file is not allowed: {relative_path}")
    candidate = (root / candidate_input).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise PermissionError(f"Path escapes repository: {relative_path}") from exc
    if must_exist and not candidate.is_file():
        raise FileNotFoundError(f"Existing file required: {relative_path}")
    return candidate


def scan_content(path: str, content: str) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    patterns = dict(UNIVERSAL_SECRET_PATTERNS)
    if Path(path).suffix.lower() not in DOCUMENTATION_SUFFIXES:
        patterns.update(EXECUTABLE_CODE_PATTERNS)
    for pattern, message in patterns.items():
        if re.search(pattern, content):
            findings.append(
                {"severity": "critical", "message": message, "path": path}
            )
    return findings
