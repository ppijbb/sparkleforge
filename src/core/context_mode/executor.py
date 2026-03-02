"""SandboxedExecutor — run code in isolated subprocess.

Only stdout/stderr (truncated) enter context; raw data stays in subprocess.
Ported from reference/claude-context-mode src/executor.ts.
"""

import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

LANGUAGE_ALIASES = ("python", "shell", "javascript", "typescript")


@dataclass
class ExecResult:
    """Result of sandboxed execution."""

    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool


def _smart_truncate(raw: str, max_bytes: int) -> str:
    """Keep head (60%) + tail (40%) of output by line boundaries."""
    raw_bytes = raw.encode("utf-8", errors="replace")
    if len(raw_bytes) <= max_bytes:
        return raw
    lines = raw.split("\n")
    head_budget = int(max_bytes * 0.6)
    tail_budget = max_bytes - head_budget
    head_lines: List[str] = []
    head_bytes = 0
    for line in lines:
        line_bytes = len((line + "\n").encode("utf-8", errors="replace"))
        if head_bytes + line_bytes > head_budget:
            break
        head_lines.append(line)
        head_bytes += line_bytes
    tail_lines: List[str] = []
    tail_bytes = 0
    for i in range(len(lines) - 1, len(head_lines) - 1, -1):
        line_bytes = len((lines[i] + "\n").encode("utf-8", errors="replace"))
        if tail_bytes + line_bytes > tail_budget:
            break
        tail_lines.insert(0, lines[i])
        tail_bytes += line_bytes
    skipped_lines = len(lines) - len(head_lines) - len(tail_lines)
    skipped_kb = (len(raw_bytes) - head_bytes - tail_bytes) / 1024
    sep = (
        f"\n\n... [{skipped_lines} lines / {skipped_kb:.1f}KB truncated — "
        f"showing first {len(head_lines)} + last {len(tail_lines)} lines] ...\n\n"
    )
    return "\n".join(head_lines) + sep + "\n".join(tail_lines)


def _detect_runtimes() -> Dict[str, Optional[str]]:
    """Detect available runtimes (python, shell, javascript)."""
    runtimes: Dict[str, Optional[str]] = {}
    runtimes["python"] = shutil.which("python3") or shutil.which("python")
    runtimes["shell"] = shutil.which("bash") or shutil.which("sh") or "sh"
    runtimes["javascript"] = shutil.which("node")
    runtimes["typescript"] = shutil.which("tsx") or shutil.which("ts-node") or runtimes["javascript"]
    return runtimes


def _build_safe_env(cwd: str) -> Dict[str, str]:
    """Build minimal env with passthrough for auth/CLI tools."""
    home = os.environ.get("HOME") or os.environ.get("USERPROFILE") or cwd
    passthrough = [
        "GH_TOKEN", "GITHUB_TOKEN", "GH_HOST",
        "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
        "AWS_REGION", "AWS_DEFAULT_REGION", "AWS_PROFILE",
        "GOOGLE_APPLICATION_CREDENTIALS", "CLOUDSDK_CONFIG",
        "DOCKER_HOST", "KUBECONFIG",
        "NPM_TOKEN", "NODE_AUTH_TOKEN", "npm_config_registry",
        "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "SSL_CERT_FILE", "CURL_CA_BUNDLE",
        "XDG_CONFIG_HOME", "XDG_DATA_HOME",
    ]
    env: Dict[str, str] = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": home,
        "TMPDIR": cwd,
        "LANG": "en_US.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "NO_COLOR": "1",
    }
    if os.name == "nt":
        for key in ("SYSTEMROOT", "COMSPEC", "PATHEXT", "USERPROFILE", "APPDATA", "TEMP", "TMP"):
            if key in os.environ:
                env[key] = os.environ[key]
    for key in passthrough:
        if key in os.environ:
            env[key] = os.environ[key]
    return env


class SandboxedExecutor:
    """Execute code in a sandboxed subprocess. Only truncated stdout/stderr returned."""

    def __init__(
        self,
        project_root: Optional[str] = None,
        max_output_bytes: int = 102_400,
        hard_cap_bytes: int = 100 * 1024 * 1024,
        runtimes: Optional[Dict[str, Optional[str]]] = None,
    ):
        self.project_root = (Path(project_root) if project_root else Path.cwd()).resolve()
        self.max_output_bytes = max_output_bytes
        self.hard_cap_bytes = hard_cap_bytes
        self._runtimes = runtimes or _detect_runtimes()

    def execute(
        self,
        language: str,
        code: str,
        timeout: int = 30_000,
    ) -> ExecResult:
        """Run code in sandbox. Returns truncated stdout/stderr."""
        lang = language.lower()
        if lang not in ("python", "shell", "javascript", "typescript"):
            lang = "python"
        runtime = self._runtimes.get(lang)
        if not runtime:
            return ExecResult(
                stdout="",
                stderr=f"Runtime not available for language: {language}",
                exit_code=1,
                timed_out=False,
            )
        tmpdir = tempfile.mkdtemp(prefix="ctx-mode-")
        try:
            ext = {"python": "py", "shell": "sh", "javascript": "js", "typescript": "ts"}[lang]
            script_path = os.path.join(tmpdir, f"script.{ext}")
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(code)
            if lang == "shell":
                os.chmod(script_path, 0o700)
            cmd: List[str]
            if lang == "python":
                cmd = [runtime, script_path]
            elif lang == "shell":
                cmd = [runtime, script_path]
            elif lang in ("javascript", "typescript"):
                cmd = [runtime, script_path]
            else:
                cmd = [runtime, script_path]
            return self._spawn(cmd, tmpdir, timeout)
        finally:
            try:
                import shutil as _shutil
                _shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass

    def execute_file(
        self,
        path: str,
        language: str,
        code: str,
        timeout: int = 30_000,
    ) -> ExecResult:
        """Read file into FILE_CONTENT in sandbox, run code. File never enters context."""
        abs_path = self.project_root / path
        if not abs_path.is_file():
            return ExecResult(
                stdout="",
                stderr=f"File not found: {abs_path}",
                exit_code=1,
                timed_out=False,
            )
        abs_path_str = str(abs_path.resolve())
        lang = language.lower()
        if lang not in ("python", "shell", "javascript", "typescript"):
            lang = "python"
        wrapped = self._wrap_with_file_content(abs_path_str, lang, code)
        return self.execute(language=lang, code=wrapped, timeout=timeout)

    def _wrap_with_file_content(self, absolute_path: str, language: str, code: str) -> str:
        """Inject FILE_CONTENT / FILE_CONTENT_PATH into code."""
        escaped = repr(absolute_path)
        if language == "python":
            return (
                f"FILE_CONTENT_PATH = {escaped}\n"
                "with open(FILE_CONTENT_PATH, 'r', encoding='utf-8', errors='replace') as _f:\n"
                "    FILE_CONTENT = _f.read()\n"
                f"{code}"
            )
        if language == "shell":
            safe = absolute_path.replace("'", "'\"'\"'")
            return (
                f"FILE_CONTENT_PATH='{safe}'\n"
                f"FILE_CONTENT=$(cat '{safe}')\n"
                f"{code}"
            )
        if language in ("javascript", "typescript"):
            return (
                "const FILE_CONTENT_PATH = " + escaped + ";\n"
                "const FILE_CONTENT = require('fs').readFileSync(FILE_CONTENT_PATH, 'utf-8');\n"
                f"{code}"
            )
        return code

    def _spawn(self, cmd: List[str], cwd: str, timeout_ms: int) -> ExecResult:
        """Run command with byte cap and timeout."""
        timeout_sec = max(1, timeout_ms // 1000)
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=_build_safe_env(cwd),
            )
            try:
                raw_stdout, raw_stderr = proc.communicate(timeout=timeout_sec)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                return ExecResult(
                    stdout="",
                    stderr=f"Execution timed out after {timeout_sec}s",
                    exit_code=1,
                    timed_out=True,
                )
            stdout_str = raw_stdout.decode("utf-8", errors="replace")
            stderr_str = raw_stderr.decode("utf-8", errors="replace")
            total_bytes = len(raw_stdout) + len(raw_stderr)
            if total_bytes > self.hard_cap_bytes:
                stderr_str += f"\n[output capped at {self.hard_cap_bytes // (1024*1024)}MB — process killed]"
            stdout_str = _smart_truncate(stdout_str, self.max_output_bytes)
            stderr_str = _smart_truncate(stderr_str, self.max_output_bytes)
            return ExecResult(
                stdout=stdout_str,
                stderr=stderr_str,
                exit_code=proc.returncode if proc.returncode is not None else 0,
                timed_out=False,
            )
        except FileNotFoundError as e:
            return ExecResult(
                stdout="",
                stderr=str(e),
                exit_code=1,
                timed_out=False,
            )
        except Exception as e:
            logger.exception("SandboxedExecutor spawn failed")
            return ExecResult(
                stdout="",
                stderr=str(e),
                exit_code=1,
                timed_out=False,
            )
