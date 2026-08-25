"""Shared unified-diff application primitives.

Extracted so that both src/core/ci/fix_issue.py and src/core/nightwelding can
apply LLM-generated diffs without duplicating the patch-application strategy:

  1. git apply --3way --ignore-whitespace  (main path)
  2. patch --fuzz=3 -p1                   (fallback: tolerates +/-3 lines of offset)
  3. fail with full diagnostics

All functions here shell out to git against an explicit `cwd` (defaulting to
the caller's current working directory when omitted) and carry no
module-level state. Passing the target repo root explicitly matters whenever
the caller operates against a different checkout than the process's own cwd
(e.g. Nightwelding's disposable worktrees) -- omitting it silently applies
patches to and reads status from the wrong repository.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


def run(
    cmd: list[str], *, input_text: str | None = None, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
        cwd=cwd,
    )


def repo_snapshot(cwd: Path | None = None) -> str:
    files = run(["git", "ls-files"], cwd=cwd).stdout.splitlines()
    keep = [
        path
        for path in files
        if path.startswith((".github/", "src/", "scripts/", "tests/"))
        or path in {"pyproject.toml", "package.json", "README.md"}
    ]
    return "\n".join(keep[:600])


def repository_change_signature(cwd: Path | None = None) -> tuple[str, ...]:
    """Return meaningful git status lines, excluding worker runtime files."""
    ignored_runtime_files = {
        "issue-context.md",
        "opencode.patch",
        "opencode-extra-context.md",
        "opencode-verify.log",
        "opencode-worker-error.log",
    }
    status_lines = []
    for line in run(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=cwd
    ).stdout.splitlines():
        path = line[3:] if len(line) > 3 else ""
        if path in ignored_runtime_files or path.endswith((".orig", ".rej")):
            continue
        status_lines.append(line)
    return tuple(sorted(status_lines))


def extract_diff(text: str) -> str:
    fenced = re.search(r"```(?:diff|patch)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    candidate = fenced.group(1).strip() if fenced else text.strip()
    idx = candidate.find("diff --git ")
    if idx >= 0:
        return candidate[idx:].strip() + "\n"
    idx = candidate.find("--- ")
    if idx >= 0:
        return candidate[idx:].strip() + "\n"
    return ""


def requested_read_paths(text: str) -> list[str]:
    paths = re.findall(
        r"<parameter\s+name=[\"']file_path[\"']>\s*([^<]+?)\s*</parameter>",
        text,
        re.IGNORECASE,
    )
    return [path.strip() for path in paths if path.strip()]


def read_full_file(path: str, limit: int = 200_000) -> str:
    """Return full file contents with line numbers, truncated to limit chars."""
    file_path = Path(path)
    if not file_path.is_file():
        return ""
    content = file_path.read_text(encoding="utf-8")
    if len(content) > limit:
        lines = content[:limit].splitlines()
        numbered = [f"{i+1:5d}: {line}" for i, line in enumerate(lines)]
        return (
            f"--- {path} (truncated to {limit} chars) ---\n"
            + "\n".join(numbered)
            + "\n...[truncated]\n"
        )
    lines = content.splitlines()
    numbered = [f"{i+1:5d}: {line}" for i, line in enumerate(lines)]
    return f"--- {path} ---\n" + "\n".join(numbered) + "\n"


def build_prompt(
    *,
    snapshot: str,
    status: str,
    issue_context: str,
    file_contents_str: str,
    extra_context: str,
    tool_context: str = "",
    force_diff: bool = False,
) -> str:
    if force_diff:
        step_by_step = """This is your FINAL attempt. File requests will no longer be honored.
You MUST output a single unified git diff now, using the file contents already provided below."""
    else:
        step_by_step = """Step-by-step process:
1. Review the issue context and the repository snapshot.
2. If you need to see the contents of a file, request it using this format:
   <parameter name="file_path">path/to/file.py</parameter>
3. Once you have enough information, output a single unified git diff."""
    return f"""
You are an autonomous coding agent editing the SparkleForge repository.

Your goal is to fix the issue described below.

{step_by_step}

Rules:
- Output ONLY the unified diff{"" if force_diff else " or a file request"}. No prose.
- Do not change generated artifacts or lockfiles.
- Keep the patch minimal.
- CRITICAL: Always emit diffs in `git diff` format:
    diff --git a/path/to/file b/path/to/file
    --- a/path/to/file
    +++ b/path/to/file
  The `a/` and `b/` prefixes are MANDATORY. Never omit them.
- The file contents below include EXACT line numbers. Your diff MUST use
  the correct line numbers as shown. The context lines must match exactly.

Repository snapshot:
{snapshot}

Current git status:
{status}

Issue context:
{issue_context}

{file_contents_str}
{tool_context}
Additional review or verification context:
{extra_context or "None"}
""".strip()


def _detect_strip_level(diff_text: str) -> int:
    """Return the -p strip level implied by the diff path headers.

    1 -> paths have 'a/' / 'b/' prefix  (standard git diff)
    0 -> bare paths, no prefix
    """
    if re.search(r"^--- a/", diff_text, re.MULTILINE):
        return 1
    if re.search(r"^diff --git a/", diff_text, re.MULTILINE):
        return 1
    return 0


def _normalize_diff_paths(diff_text: str) -> str:
    """Rewrite bare path headers to standard 'a/' 'b/' prefix format.

    LLMs sometimes emit:
        --- .github/workflows/foo.yml
        +++ .github/workflows/foo.yml
    but git apply -p1 strips the first path component, turning
    '.github' into the prefix that gets removed, leaving 'workflows/...'
    which does not exist in the index.

    This function rewrites such headers to:
        --- a/.github/workflows/foo.yml
        +++ b/.github/workflows/foo.yml
    so that git apply -p1 correctly resolves the path.
    """
    if re.search(r"^--- a/", diff_text, re.MULTILINE):
        return diff_text  # already in standard format

    def _add_prefix(m: re.Match) -> str:
        sign = m.group(1)  # '---' or '+++'
        prefix = "a" if sign == "---" else "b"
        path = m.group(2)
        rest = m.group(3) or ""
        return f"{sign} {prefix}/{path}{rest}"

    return re.sub(
        r"^(---|\+\+\+) (?!a/|b/|/dev/null)(\S+)(.*)",
        _add_prefix,
        diff_text,
        flags=re.MULTILINE,
    )


def _format_hunk_range(start: str, count: int) -> str:
    if count == 1:
        return start
    return f"{start},{count}"


def _repair_hunk_headers(diff_text: str) -> str:
    """Recalculate unified-diff hunk line counts from the hunk body."""
    lines = diff_text.splitlines(keepends=True)
    repaired: list[str] = []
    i = 0
    hunk_header = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@(.*?)(\r?\n)?$")

    while i < len(lines):
        match = hunk_header.match(lines[i])
        if not match:
            repaired.append(lines[i])
            i += 1
            continue

        header_index = len(repaired)
        repaired.append(lines[i])
        old_count = 0
        new_count = 0
        i += 1

        while i < len(lines):
            line = lines[i]
            if line.startswith("@@ ") or line.startswith("diff --git "):
                break
            if line.startswith(("--- ", "+++ ")):
                break
            if line in {"\n", "\r\n"}:
                old_count += 1
                new_count += 1
                repaired.append(" " + line)
                i += 1
                continue
            if line.startswith("\\ No newline at end of file"):
                repaired.append(line)
                i += 1
                continue

            marker = line[:1]
            if marker == " ":
                old_count += 1
                new_count += 1
            elif marker == "-":
                old_count += 1
            elif marker == "+":
                new_count += 1
            repaired.append(line)
            i += 1

        newline = match.group(4) or ""
        repaired[header_index] = (
            f"@@ -{_format_hunk_range(match.group(1), old_count)} "
            f"+{_format_hunk_range(match.group(2), new_count)} @@{match.group(3)}{newline}"
        )

    return "".join(repaired)


def _normalize_diff(diff_text: str) -> str:
    return _repair_hunk_headers(_normalize_diff_paths(diff_text))


def _header_repo_path(raw_path: str) -> str | None:
    path = raw_path.split("\t", 1)[0].strip()
    if path == "/dev/null":
        return None
    if path.startswith(("a/", "b/")):
        return path[2:]
    return path


def _invalid_patch_path_reason(repo_path: str) -> str | None:
    if not repo_path:
        return "empty patch path"
    parts = [part for part in repo_path.split("/") if part]
    if repo_path.startswith("/") or ".." in parts:
        return f"path escapes repository: {repo_path}"
    if repo_path.startswith(("a/", "b/")):
        return (
            "diff-prefix path is embedded in repository path: "
            f"{repo_path}. Use tests/foo.py, not a/tests/foo.py."
        )
    return None


def _diff_git_header_paths(line: str) -> tuple[str, str] | None:
    prefix = "diff --git a/"
    if not line.startswith(prefix):
        return None

    rest = line[len(prefix) :].rstrip()
    separator = " b/"
    separator_index = rest.rfind(separator)
    if separator_index <= 0:
        return None

    left = rest[:separator_index]
    right = rest[separator_index + len(separator) :]
    if not left or not right:
        return None
    return left, right


def _validate_patch_paths(diff_text: str) -> str:
    invalid: list[str] = []

    for line in diff_text.splitlines():
        paths = _diff_git_header_paths(line)
        if paths is None:
            continue
        for repo_path in paths:
            reason = _invalid_patch_path_reason(repo_path)
            if reason:
                invalid.append(reason)

    for match in re.finditer(r"^(?:---|\+\+\+) (.+?)\s*$", diff_text, flags=re.MULTILINE):
        repo_path = _header_repo_path(match.group(1))
        if repo_path is None:
            continue
        reason = _invalid_patch_path_reason(repo_path)
        if reason:
            invalid.append(reason)

    if not invalid:
        return ""
    unique = list(dict.fromkeys(invalid))
    return "Invalid patch path(s):\n- " + "\n- ".join(unique)


def _split_multifile_patch(diff_text: str) -> list[tuple[str, str]]:
    """Split a multi-file diff into (filepath, patch_segment) pairs.

    GNU patch can't handle 'diff --git' headers inside a multi-file stream;
    it reports 'malformed patch at line N: diff --git ...'.  Splitting and
    applying each file's hunk individually avoids that problem entirely.
    """
    segments: list[tuple[str, str]] = []
    # Each segment starts at a 'diff --git' header
    parts = re.split(r"(?=^diff --git )", diff_text, flags=re.MULTILINE)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        first_line = part.splitlines()[0]
        paths = _diff_git_header_paths(first_line)
        if paths:
            filepath = paths[1]  # use b/ path (destination)
            segments.append((filepath, part + "\n"))
    return segments


def _apply_single_patch(diff_text: str, label: str = "", cwd: Path | None = None) -> tuple[bool, str]:
    """Apply a single-file patch using all available strategies.

    Returns (success, error_summary).
    """
    # Write to a temp file, inside `cwd` so a relative git-apply target resolves.
    tmp = (cwd or Path.cwd()) / "opencode-single.patch"
    diff_text = _normalize_diff(diff_text)
    path_error = _validate_patch_paths(diff_text)
    if path_error:
        return False, path_error
    tmp.write_text(diff_text, encoding="utf-8")
    strip = _detect_strip_level(diff_text)
    errors: list[str] = []

    # Try the strip level implied by the diff's own path headers first. A
    # naive `sorted({strip, 1 - strip})` always tries -p0 before -p1, and for
    # new-file hunks (--- /dev/null) `git apply -p0` has nothing to fail
    # against -- it happily creates the file at the literal 'b/...' path
    # instead of failing over to the correct strip level.
    for p in (strip, 1 - strip):
        proc = run(
            [
                "git",
                "apply",
                f"-p{p}",
                "--3way",
                "--ignore-whitespace",
                "--whitespace=nowarn",
                str(tmp),
            ],
            cwd=cwd,
        )
        if proc.returncode == 0:
            tmp.unlink(missing_ok=True)
            return True, ""
        err = proc.stderr.strip()
        errors.append(f"git apply -p{p}: {err}")
        print(f"[{label}][git apply -p{p}] failed: {err[:120]}", file=sys.stderr)

    patch_bin = run(["which", "patch"]).stdout.strip()
    if patch_bin:
        for p in (strip, 1 - strip):
            proc2 = run(
                ["patch", f"--strip={p}", "--fuzz=3", "--batch", "--forward"],
                input_text=diff_text,
                cwd=cwd,
            )
            if proc2.returncode == 0:
                print(f"[{label}][patch --strip={p}] succeeded.", file=sys.stderr)
                tmp.unlink(missing_ok=True)
                return True, ""
            fuzz_err = (proc2.stderr.strip() or proc2.stdout.strip())[:300]
            errors.append(f"patch --strip={p}: {fuzz_err}")
            print(f"[{label}][patch --strip={p}] failed: {fuzz_err[:120]}", file=sys.stderr)

    tmp.unlink(missing_ok=True)
    return False, "\n".join(errors)


def _apply_patch(patch_path: Path, cwd: Path | None = None) -> tuple[bool, str]:
    """Apply a (possibly multi-file) patch robustly.

    Strategy:
    1. Normalise bare paths -> a/b prefix.
    2. Try the whole patch at once with git apply (fast path).
    3. If that fails, split into per-file segments and apply each independently.
       - Every file segment must succeed. Partial application is a failed fix.
    4. Return (success, error_summary).
    """
    diff_text = patch_path.read_text(encoding="utf-8")

    # -- Step 1: normalise paths and hunk counts --------------------------------
    normalised = _normalize_diff(diff_text)
    if normalised != diff_text:
        print("[patch] Normalised diff paths and hunk headers.", file=sys.stderr)
        patch_path.write_text(normalised, encoding="utf-8")
        diff_text = normalised

    path_error = _validate_patch_paths(diff_text)
    if path_error:
        return False, path_error

    strip = _detect_strip_level(diff_text)

    # -- Step 2: try whole patch at once (fast path) ----------------------------
    # Detected strip level first -- see the comment in _apply_single_patch for
    # why trying -p0 before the detected level is unsafe for new-file hunks.
    for p in (strip, 1 - strip):
        proc = run(
            [
                "git",
                "apply",
                f"-p{p}",
                "--3way",
                "--ignore-whitespace",
                "--whitespace=nowarn",
                str(patch_path),
            ],
            cwd=cwd,
        )
        if proc.returncode == 0:
            print("[patch] Whole-patch git apply succeeded.", file=sys.stderr)
            return True, ""
        err = proc.stderr.strip()
        print(f"[git apply -p{p} --3way] failed: {err[:200]}", file=sys.stderr)

    # -- Step 3: split into per-file patches and apply individually -------------
    segments = _split_multifile_patch(diff_text)
    if len(segments) <= 1:
        # Single file -- surface all errors clearly
        ok, errs = _apply_single_patch(diff_text, label="single", cwd=cwd)
        return ok, errs

    print(
        f"[patch] Whole-patch failed; splitting into {len(segments)} per-file patches.",
        file=sys.stderr,
    )
    succeeded: list[str] = []
    failed: list[str] = []
    all_errors: list[str] = []

    for filepath, seg in segments:
        ok, errs = _apply_single_patch(seg, label=filepath, cwd=cwd)
        if ok:
            succeeded.append(filepath)
            print(f"  ✅ {filepath}", file=sys.stderr)
        else:
            failed.append(filepath)
            all_errors.append(f"FAILED {filepath}:\n{errs}")
            print(f"  ❌ {filepath}", file=sys.stderr)

    if succeeded:
        summary = f"Applied {len(succeeded)}/{len(segments)} file(s): {succeeded}"
        if failed:
            summary += f"  Skipped: {failed}"
        print(f"[patch] Partial success: {summary}", file=sys.stderr)
        if failed:
            return False, (
                "Patch only applied partially. "
                "All files in the generated diff must apply cleanly.\n"
                f"{summary}\n\n" + "\n\n".join(all_errors)
            )
        return True, ""

    return False, "\n\n".join(all_errors)
