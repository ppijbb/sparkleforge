"""OpenCode repair loop: retries `sparkleforge ci fix-issue` with self-verify/verify-command gating.

1:1 relocation of the "OpenCode repair loop" bash step from
.github/workflows/opencode-auto-fix.yml -- preserves its exact semantics,
including a quirk that looks unintentional but is kept as-is: a failing
self-verify command aborts the whole loop immediately (no retry), while a
failing verify command retries up to max_iterations.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from src.core import patch_ops
from src.utils.sparkleforge_history import (
    end_history_session,
    log_history_event,
    start_history_session,
)

_FIX_ISSUE_TIMEOUT_SECONDS = 600
_HISTORY_SESSION_ENV = "SPARKLEFORGE_HISTORY_SESSION_ID"


@dataclass
class AutofixResult:
    success: bool
    reason: str = ""
    attempts: int = 0


def _run(
    cmd: list[str], cwd: Path, timeout: int | None = None, env: dict | None = None
) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False, env=env
    )


def run_autofix_repair_loop(
    issue_context_path: Path,
    repo_root: Path,
    commit_title: str,
    max_iterations: int = 3,
    verify_command: str = "python -m compileall -q src scripts",
    self_verify_command: str | None = None,
) -> AutofixResult:
    max_iterations = max(1, min(5, max_iterations))

    issue_ref = None
    try:
        match = re.search(r"/issues/(\d+)", issue_context_path.read_text(encoding="utf-8"))
        issue_ref = match.group(1) if match else None
    except OSError:
        pass
    history_session_id = start_history_session("autofix", external_ref=issue_ref, title=commit_title)

    def _finish(result: AutofixResult) -> AutofixResult:
        end_history_session(
            history_session_id,
            "succeeded" if result.success else "failed",
            metadata={"reason": result.reason, "attempts": result.attempts},
        )
        return result

    extra_context_path = repo_root / "opencode-extra-context.md"
    extra_context_path.write_text("", encoding="utf-8")
    worker_error_path = repo_root / "opencode-worker-error.log"
    self_verify_log_path = repo_root / "opencode-self-verify.log"
    verify_log_path = repo_root / "opencode-verify.log"
    patch_path = repo_root / "opencode.patch"

    sparkleforge_entrypoint = Path(__file__).resolve().parents[3] / "main.py"
    if not sparkleforge_entrypoint.exists():
        sparkleforge_entrypoint = repo_root / "main.py"

    for attempt in range(1, max_iterations + 1):
        is_last_attempt = attempt == max_iterations
        print(f"OpenCode attempt {attempt} of {max_iterations}")
        log_history_event(
            history_session_id, "log", f"OpenCode attempt {attempt} of {max_iterations}"
        )

        proc = _run(
            [
                sys.executable,
                str(sparkleforge_entrypoint),
                "ci",
                "fix-issue",
                "--issue-context",
                str(issue_context_path),
                "--extra-context",
                str(extra_context_path),
            ],
            cwd=repo_root,
            timeout=_FIX_ISSUE_TIMEOUT_SECONDS,
            env={**os.environ, _HISTORY_SESSION_ENV: history_session_id},
        )
        worker_error_path.write_text(proc.stderr or "", encoding="utf-8")

        if proc.returncode != 0:
            lines = [
                "Previous OpenCode attempt failed before commit.",
                "",
                "git apply, no-op patch, or worker error:",
                proc.stderr or "",
            ]
            if patch_path.exists():
                lines += ["", "Rejected patch:", "\n".join(patch_path.read_text(encoding="utf-8").splitlines()[:220])]
            lines += ["", "Regenerate the patch against the current repository contents."]
            extra_context_path.write_text("\n".join(lines), encoding="utf-8")
            log_history_event(
                history_session_id, "error", proc.stderr or "OpenCode attempt failed", level="error"
            )

            if is_last_attempt:
                print(proc.stderr or "")
                return _finish(
                    AutofixResult(success=False, reason=proc.stderr or "OpenCode attempt failed", attempts=attempt)
                )
            continue

        # repository_change_signature() reads the excludes both this loop and the
        # old bash step already agreed on (issue-context.md, opencode.patch,
        # opencode-extra-context.md, opencode-verify.log, opencode-worker-error.log,
        # *.orig/*.rej) -- reuse it instead of reimplementing the pathspec exclude list.
        changed = bool(patch_ops.repository_change_signature())
        if not changed:
            message = f"OpenCode produced no repository changes on attempt {attempt}."
            print(message)
            worker_error_path.write_text(message, encoding="utf-8")
            extra_context_path.write_text(
                "Previous OpenCode attempt produced no repository changes.\n\n"
                "The response either generated a no-op patch or changed only files ignored by git.\n"
                "Regenerate a minimal patch that changes tracked repository files relevant to the issue.",
                encoding="utf-8",
            )
            log_history_event(history_session_id, "error", message, level="error")
            if is_last_attempt:
                return _finish(AutofixResult(success=False, reason=message, attempts=attempt))
            continue

        _run(["git", "add", "-u", "."], cwd=repo_root)
        untracked = _run(
            ["git", "ls-files", "--others", "--exclude-standard"], cwd=repo_root
        ).stdout.splitlines()
        if untracked:
            _run(["git", "add", "--", *untracked], cwd=repo_root)

        staged = _run(["git", "diff", "--cached", "--quiet"], cwd=repo_root)
        if staged.returncode == 0:
            return _finish(
                AutofixResult(
                    success=False,
                    reason="Repository changes were detected but no non-ignored files were staged.",
                    attempts=attempt,
                )
            )

        validator_script = Path(__file__).resolve().parents[3] / "scripts" / "validate_commit_messages.py"
        if not validator_script.exists():
            validator_script = repo_root / "scripts" / "validate_commit_messages.py"
        validate = _run([sys.executable, str(validator_script), "--message", commit_title], cwd=repo_root)
        if validate.returncode != 0:
            return _finish(
                AutofixResult(
                    success=False,
                    reason=f"Derived commit title failed validation: {commit_title!r}",
                    attempts=attempt,
                )
            )
        commit = _run(["git", "commit", "-m", commit_title], cwd=repo_root)
        if commit.returncode != 0:
            reason = f"git commit failed: {(commit.stderr or commit.stdout or '').strip()}"
            log_history_event(history_session_id, "error", reason, level="error")
            return _finish(AutofixResult(success=False, reason=reason, attempts=attempt))
        log_history_event(history_session_id, "commit", commit_title)

        if self_verify_command:
            print(f"Running self-verification: {self_verify_command}")
            self_verify = _run(["bash", "-lc", self_verify_command], cwd=repo_root)
            self_verify_log_path.write_text(
                (self_verify.stdout or "") + (self_verify.stderr or ""), encoding="utf-8"
            )
            if self_verify.returncode != 0:
                # Preserved as-is: unlike verify_command below, a failing self-verify
                # aborts the whole loop immediately, even on a non-final attempt.
                print("Self-verification failed.")
                print(self_verify_log_path.read_text(encoding="utf-8"))
                reason = f"Self-verification failed: {self_verify_command}"
                log_history_event(history_session_id, "error", reason, level="error")
                return _finish(AutofixResult(success=False, reason=reason, attempts=attempt))

        verify = _run(["bash", "-lc", verify_command], cwd=repo_root)
        verify_log_path.write_text((verify.stdout or "") + (verify.stderr or ""), encoding="utf-8")
        if verify.returncode == 0:
            print(f"Verification passed: {verify_command}")
            log_history_event(history_session_id, "log", f"Verification passed: {verify_command}")
            return _finish(AutofixResult(success=True, attempts=attempt))

        verify_output = (verify.stdout or "") + (verify.stderr or "")
        extra_context_path.write_text(
            "Previous attempt did not pass verification.\n\n"
            f"Command:\n{verify_command}\n\n"
            "Output:\n" + "\n".join(verify_output.splitlines()[-160:]),
            encoding="utf-8",
        )
        if is_last_attempt:
            print(f"Verification still failed after {max_iterations} attempts.")
            print(verify_output)
            reason = f"Verification still failed after {max_iterations} attempts: {verify_command}"
            log_history_event(history_session_id, "error", reason, level="error")
            return _finish(AutofixResult(success=False, reason=reason, attempts=attempt))

    return _finish(AutofixResult(success=False, reason="Exhausted all attempts.", attempts=max_iterations))
