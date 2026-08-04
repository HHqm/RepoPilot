from __future__ import annotations

import difflib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from hashlib import sha256
from pathlib import Path
from typing import Any

from .security import resolve_safe_path


def content_sha256(content: str) -> str:
    return sha256(content.encode("utf-8")).hexdigest()


def artifact_digest(value: Any) -> str:
    """Canonical digest binding a human decision to the displayed artifact."""

    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(serialized.encode("utf-8")).hexdigest()


def unified_diff(path: str, original: str, replacement: str) -> str:
    return "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            replacement.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )


class PatchApplier:
    def __init__(self, runtime_dir: Path):
        self.runtime_dir = runtime_dir

    def apply(
        self, repo_root: str | Path, run_id: str, proposals: list[dict[str, Any]]
    ) -> dict[str, Any]:
        root = Path(repo_root).resolve()
        backup_root = self.runtime_dir / "backups" / run_id
        prepared: list[tuple[dict[str, Any], Path, Path]] = []
        changed: list[str] = []

        # Preflight every optimistic lock before the first write. This prevents a
        # later stale file from leaving earlier files partially modified.
        for proposal in proposals:
            target = resolve_safe_path(root, proposal["path"])
            original_bytes = target.read_bytes()
            actual_hash = sha256(original_bytes).hexdigest()
            if actual_hash != proposal["original_sha256"]:
                raise RuntimeError(
                    f"Optimistic lock failed for {proposal['path']}: file changed"
                )
            relative = target.relative_to(root)
            backup = backup_root / relative
            prepared.append((proposal, target, backup))

        for _, target, backup in prepared:
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup)

        try:
            for proposal, target, _ in prepared:
                target.parent.mkdir(parents=True, exist_ok=True)
                fd, temporary_name = tempfile.mkstemp(
                    prefix=f".{target.name}.",
                    suffix=".repopilot.tmp",
                    dir=target.parent,
                )
                try:
                    with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                        handle.write(proposal["new_content"])
                    os.replace(temporary_name, target)
                finally:
                    if os.path.exists(temporary_name):
                        os.unlink(temporary_name)
                changed.append(proposal["path"])
        except Exception:
            for relative_path in reversed(changed):
                target = resolve_safe_path(root, relative_path)
                backup = backup_root / relative_path
                if backup.is_file():
                    shutil.copy2(backup, target)
            raise

        return {
            "success": True,
            "changed_files": changed,
            "applied_sha256": {
                proposal["path"]: content_sha256(proposal["new_content"])
                for proposal, _, _ in prepared
            },
            "backup_root": str(backup_root),
        }

    def rollback(
        self,
        repo_root: str | Path,
        run_id: str,
        changed_files: list[str],
        *,
        expected_applied_sha256: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        root = Path(repo_root).resolve()
        backup_root = self.runtime_dir / "backups" / run_id
        restored: list[str] = []
        for relative_path in changed_files:
            target = resolve_safe_path(root, relative_path)
            if expected_applied_sha256 and relative_path in expected_applied_sha256:
                actual = sha256(target.read_bytes()).hexdigest()
                expected = expected_applied_sha256[relative_path]
                if actual != expected:
                    raise RuntimeError(
                        "Rollback refused because the applied file changed after "
                        f"RepoPilot wrote it: {relative_path}"
                    )
            backup = (backup_root / relative_path).resolve()
            try:
                backup.relative_to(backup_root.resolve())
            except ValueError as exc:
                raise PermissionError("Backup path escaped backup root") from exc
            if not backup.is_file():
                raise FileNotFoundError(f"Missing backup for {relative_path}")
            shutil.copy2(backup, target)
            restored.append(relative_path)
        return {"success": True, "restored_files": restored}


class TestRunner:
    _DENIED_PYTEST_OPTIONS = {
        "-p",
        "--basetemp",
        "--rootdir",
        "--confcutdir",
        "--override-ini",
        "-c",
    }

    def __init__(
        self,
        timeout_seconds: int = 120,
        *,
        allow_host_execution: bool = False,
    ):
        self.timeout_seconds = timeout_seconds
        self.allow_host_execution = allow_host_execution

    def run(self, repo_root: str | Path, command: str) -> dict[str, Any]:
        root = Path(repo_root).resolve()
        arguments = self.validate_command(command, repo_root=root)
        if not self.allow_host_execution:
            raise PermissionError(
                "Host test execution is disabled. Run tests in an isolated CI/"
                "container, or explicitly set REPOPILOT_ALLOW_HOST_TEST_EXECUTION=1 "
                "for a trusted local repository."
            )
        with tempfile.TemporaryDirectory(prefix="repopilot-test-home-") as home:
            environment = self._sanitized_environment(Path(home))
            completed = subprocess.run(
                arguments,
                cwd=root,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                shell=False,
                encoding="utf-8",
                errors="replace",
                env=environment,
            )
        return {
            "success": completed.returncode == 0,
            "returncode": completed.returncode,
            "command": arguments,
            "stdout": completed.stdout[-20_000:],
            "stderr": completed.stderr[-20_000:],
        }

    @staticmethod
    def validate_command(
        command: str,
        *,
        repo_root: str | Path | None = None,
    ) -> list[str]:
        if any(marker in command for marker in ("|", "&&", "||", ">", "<", ";", "`")):
            raise PermissionError("Shell operators are not allowed in test commands")
        arguments = shlex.split(command, posix=os.name != "nt")
        if not arguments:
            raise ValueError("Test command must not be empty")
        executable = Path(arguments[0]).name.lower()
        if executable in {"python", "python.exe", "py", "py.exe"}:
            if len(arguments) < 3 or arguments[1] != "-m":
                raise PermissionError("Only 'python -m unittest/pytest' is allowed")
            if arguments[2] not in {"unittest", "pytest"}:
                raise PermissionError("Only unittest or pytest modules are allowed")
            arguments[0] = sys.executable
            TestRunner._validate_arguments(
                arguments[3:],
                repo_root,
                module=arguments[2],
            )
            return arguments
        if executable in {"pytest", "pytest.exe"}:
            normalized = [sys.executable, "-m", "pytest", *arguments[1:]]
            TestRunner._validate_arguments(
                normalized[3:],
                repo_root,
                module="pytest",
            )
            return normalized
        raise PermissionError("Test command is not in the allowlist")

    @classmethod
    def _validate_arguments(
        cls,
        arguments: list[str],
        repo_root: str | Path | None,
        *,
        module: str,
    ) -> None:
        for index, argument in enumerate(arguments):
            option = argument.split("=", 1)[0]
            if module == "pytest" and option in cls._DENIED_PYTEST_OPTIONS:
                raise PermissionError(f"Unsafe test option is not allowed: {option}")
            if argument == ".." or argument.startswith("../") or argument.startswith("..\\"):
                raise PermissionError("Test paths cannot escape the repository")
            candidate = Path(argument)
            if candidate.is_absolute():
                if repo_root is None:
                    raise PermissionError("Absolute test paths are not allowed")
                try:
                    candidate.resolve().relative_to(Path(repo_root).resolve())
                except ValueError as exc:
                    raise PermissionError(
                        "Test paths must stay inside the repository"
                    ) from exc
            if (
                index > 0
                and arguments[index - 1] in {"-s", "-t"}
                and repo_root is not None
            ):
                target = (Path(repo_root).resolve() / argument).resolve()
                try:
                    target.relative_to(Path(repo_root).resolve())
                except ValueError as exc:
                    raise PermissionError(
                        "Test discovery paths must stay inside the repository"
                    ) from exc

    @staticmethod
    def _sanitized_environment(temporary_home: Path) -> dict[str, str]:
        allowed_names = {
            "COMSPEC",
            "NUMBER_OF_PROCESSORS",
            "OS",
            "PATH",
            "PATHEXT",
            "PROCESSOR_ARCHITECTURE",
            "SYSTEMDRIVE",
            "SYSTEMROOT",
            "WINDIR",
        }
        environment = {
            key: value
            for key, value in os.environ.items()
            if key.upper() in allowed_names
        }
        environment.update(
            {
                "HOME": str(temporary_home),
                "USERPROFILE": str(temporary_home),
                "TEMP": str(temporary_home),
                "TMP": str(temporary_home),
                "PYTHONNOUSERSITE": "1",
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            }
        )
        return environment
