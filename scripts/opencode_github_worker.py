"""GitHub Actions helper for low-cost OpenCode issue fixing.

This worker reads a GitHub issue context file, asks the local OpenCode agent for
a unified diff, and applies it to the checked-out branch.

Robust patch application strategy:
  1. git apply --3way --ignore-whitespace  (main path)
  2. patch --fuzz=3 -p1                   (fallback: tolerates ±3 lines of offset)
  3. fail with full diagnostics
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.cli_agents.open_code_agent import OpenCodeAgent

# CI 잡은 LLM_PROVIDER/LLM_MODEL과 API 키만 주입하므로, 설정 로더가 요구하는
# 나머지 필수 변수에 안전한 기본값을 채운다. 시크릿 성격의 키에는 기본값을 두지 않는다.
_CONFIG_ENV_DEFAULTS = {
    "LLM_TEMPERATURE": "0.2",
    "LLM_MAX_TOKENS": "8192",
    "BUDGET_LIMIT": "5.0",
    "ENABLE_COST_OPTIMIZATION": "true",
}
_MODEL_ROLE_KEYS = (
    "PLANNING_MODEL",
    "REASONING_MODEL",
    "VERIFICATION_MODEL",
    "GENERATION_MODEL",
    "COMPRESSION_MODEL",
)


def ensure_config_loaded() -> None:
    """MultiModelOrchestrator 생성 전에 전역 LLM 설정을 1회 로드한다."""
    from src.core import researcher_config

    if researcher_config.config is not None:
        return
    for key, value in _CONFIG_ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)
    default_model = os.getenv("LLM_MODEL")
    if default_model:
        for key in _MODEL_ROLE_KEYS:
            os.environ.setdefault(key, default_model)
    researcher_config.load_config_from_env()


def run(cmd: list[str], *, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
    )


def repo_snapshot() -> str:
    files = run(["git", "ls-files"]).stdout.splitlines()
    keep = [
        path
        for path in files
        if path.startswith((".github/", "src/", "scripts/", "tests/"))
        or path in {"pyproject.toml", "package.json", "README.md"}
    ]
    return "\n".join(keep[:600])


def repository_change_signature() -> tuple[str, ...]:
    """Return meaningful git status lines, excluding worker runtime files."""
    ignored_runtime_files = {
        "issue-context.md",
        "opencode.patch",
        "opencode-extra-context.md",
        "opencode-verify.log",
        "opencode-worker-error.log",
    }
    status_lines = []
    for line in run(["git", "status", "--porcelain", "--untracked-files=all"]).stdout.splitlines():
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


def _infer_relevant_files(issue_context: str, all_files: list[str]) -> list[str]:
    """Heuristically find files most relevant to the issue text."""
    relevant: list[str] = []
    # 1. Exact filename mentions
    for f in all_files:
        basename = Path(f).name
        if basename in issue_context or f in issue_context:
            relevant.append(f)
    # 2. Token-based match (function/class names)
    tokens = {
        token
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_.]{7,}", issue_context)
        if token not in {"github", "actions", "workflow", "unified", "repository"}
    }
    for f in all_files:
        if f in relevant:
            continue
        if not Path(f).is_file():
            continue
        try:
            content = Path(f).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if any(tok in content for tok in sorted(tokens, key=len, reverse=True)[:15]):
            relevant.append(f)
        if len(relevant) >= 8:
            break
    return relevant[:8]


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

    1 → paths have 'a/' / 'b/' prefix  (standard git diff)
    0 → bare paths, no prefix
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
        m = re.match(r"^diff --git a/(\S+) b/(\S+)", part)
        if m:
            filepath = m.group(2)  # use b/ path (destination)
            segments.append((filepath, part + "\n"))
    return segments


def _apply_single_patch(diff_text: str, label: str = "") -> tuple[bool, str]:
    """Apply a single-file patch using all available strategies.

    Returns (success, error_summary).
    """
    # Write to a temp file
    tmp = Path("opencode-single.patch")
    diff_text = _normalize_diff(diff_text)
    tmp.write_text(diff_text, encoding="utf-8")
    strip = _detect_strip_level(diff_text)
    errors: list[str] = []

    for p in sorted({strip, 1 - strip}):
        proc = run(
            [
                "git",
                "apply",
                f"-p{p}",
                "--3way",
                "--ignore-whitespace",
                "--whitespace=nowarn",
                str(tmp),
            ]
        )
        if proc.returncode == 0:
            tmp.unlink(missing_ok=True)
            return True, ""
        err = proc.stderr.strip()
        errors.append(f"git apply -p{p}: {err}")
        print(f"[{label}][git apply -p{p}] failed: {err[:120]}", file=sys.stderr)

    patch_bin = run(["which", "patch"]).stdout.strip()
    if patch_bin:
        for p in sorted({strip, 1 - strip}):
            proc2 = run(
                ["patch", f"--strip={p}", "--fuzz=3", "--batch", "--forward"],
                input_text=diff_text,
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


def _apply_patch(patch_path: Path) -> tuple[bool, str]:
    """Apply a (possibly multi-file) patch robustly.

    Strategy:
    1. Normalise bare paths → a/b prefix.
    2. Try the whole patch at once with git apply (fast path).
    3. If that fails, split into per-file segments and apply each independently.
       - Every file segment must succeed. Partial application is a failed fix.
    4. Return (success, error_summary).
    """
    diff_text = patch_path.read_text(encoding="utf-8")

    # ── Step 1: normalise paths and hunk counts ────────────────────────────
    normalised = _normalize_diff(diff_text)
    if normalised != diff_text:
        print("[patch] Normalised diff paths and hunk headers.", file=sys.stderr)
        patch_path.write_text(normalised, encoding="utf-8")
        diff_text = normalised

    strip = _detect_strip_level(diff_text)

    # ── Step 2: try whole patch at once (fast path) ────────────────────────
    for p in sorted({strip, 1 - strip}):
        proc = run(
            [
                "git",
                "apply",
                f"-p{p}",
                "--3way",
                "--ignore-whitespace",
                "--whitespace=nowarn",
                str(patch_path),
            ]
        )
        if proc.returncode == 0:
            print("[patch] Whole-patch git apply succeeded.", file=sys.stderr)
            return True, ""
        err = proc.stderr.strip()
        print(f"[git apply -p{p} --3way] failed: {err[:200]}", file=sys.stderr)

    # ── Step 3: split into per-file patches and apply individually ─────────
    segments = _split_multifile_patch(diff_text)
    if len(segments) <= 1:
        # Single file — surface all errors clearly
        ok, errs = _apply_single_patch(diff_text, label="single")
        return ok, errs

    print(
        f"[patch] Whole-patch failed; splitting into {len(segments)} per-file patches.",
        file=sys.stderr,
    )
    succeeded: list[str] = []
    failed: list[str] = []
    all_errors: list[str] = []

    for filepath, seg in segments:
        ok, errs = _apply_single_patch(seg, label=filepath)
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


async def fix_issue(issue_context_path: Path, extra_context_path: Path | None = None) -> int:
    issue_context = issue_context_path.read_text(encoding="utf-8")
    extra_context = ""
    if extra_context_path and extra_context_path.exists():
        extra_context = extra_context_path.read_text(encoding="utf-8").strip()
    snapshot = repo_snapshot()
    status = run(["git", "status", "--short"]).stdout

    all_files = snapshot.splitlines()

    # Always provide full file contents for relevant files (not just snippets)
    relevant_files = _infer_relevant_files(issue_context, all_files)
    relevant_contents = []
    for f in relevant_files:
        try:
            content = read_full_file(f)
            if content:
                relevant_contents.append(content)
        except Exception:
            pass

    file_contents_str = "\n\n".join(relevant_contents[:5])
    if file_contents_str:
        file_contents_str = (
            f"Relevant File Contents (with exact line numbers):\n{file_contents_str}\n"
        )

    agent = OpenCodeAgent()
    tool_context = ""
    response = ""
    diff = ""
    system_message = (
        "You are a careful coding agent working against a real repository. "
        "On each turn, output EITHER a file request using "
        '<parameter name="file_path">path/to/file.py</parameter> '
        "OR a git-apply compatible unified diff — never both, and no other "
        "prose, markdown narration, or tool calls. "
        "The diff context lines must match the file exactly."
    )
    max_llm_attempts = 3
    for llm_attempt in range(max_llm_attempts):
        is_final_attempt = llm_attempt == max_llm_attempts - 1
        prompt = build_prompt(
            snapshot=snapshot,
            status=status,
            issue_context=issue_context,
            file_contents_str=file_contents_str,
            extra_context=extra_context,
            tool_context=tool_context,
            force_diff=is_final_attempt,
        )
        result = await agent.execute_query(prompt, system_message=system_message)
        if not result.get("success"):
            print(
                result.get("response") or result.get("error") or "OpenCode failed", file=sys.stderr
            )
            return 1

        response = result.get("response", "")
        diff = extract_diff(response)
        if diff:
            break

        if is_final_attempt:
            break

        paths = requested_read_paths(response)
        if not paths:
            break

        requested_context = []
        for path in paths[:3]:
            if path in all_files and Path(path).is_file():
                requested_context.append(read_full_file(path))
        if not requested_context:
            break

        tool_context = (
            "Requested file contents are provided below (with exact line numbers). "
            "You must now return only a unified diff, with no tool calls or prose. "
            "Use the exact line numbers shown when writing the diff hunk headers.\n"
            + "\n\n".join(requested_context)
            + "\n"
        )

    if not diff:
        print("OpenCode did not return an applicable diff.", file=sys.stderr)
        print(response[:4000], file=sys.stderr)
        return 1

    patch_path = Path("opencode.patch")
    before_signature = repository_change_signature()
    patch_path.write_text(diff, encoding="utf-8")

    success, err = _apply_patch(patch_path)
    if not success:
        print(err, file=sys.stderr)
        print("--- Failed Patch ---", file=sys.stderr)
        print(diff[:4000], file=sys.stderr)
        return 1
    if repository_change_signature() == before_signature:
        print("OpenCode patch applied cleanly but produced no repository changes.", file=sys.stderr)
        print("--- No-op Patch ---", file=sys.stderr)
        print(diff[:4000], file=sys.stderr)
        return 1

    return 0


async def code_review(diff_path: Path) -> int:
    if not diff_path.exists():
        print(f"Error: Diff file {diff_path} not found.", file=sys.stderr)
        return 1
    diff = diff_path.read_text(encoding="utf-8")
    if not diff.strip():
        Path("review_result.txt").write_text("No changes or empty diff.", encoding="utf-8")
        return 0
    
    from src.core.llm_manager import MultiModelOrchestrator, TaskType
    ensure_config_loaded()
    orchestrator = MultiModelOrchestrator()
    prompt = f"You are an expert code reviewer. Read the git diff and summarize key issues, bugs, or style violations briefly.\n\nGit Diff:\n{diff}"
    system_message = "You are an expert code reviewer. If the primary model is unavailable, the system will fallback."
    
    result = await orchestrator.execute_with_model(
        prompt=prompt,
        task_type=TaskType.RESEARCH,
        system_message=system_message,
        use_cascade=False
    )
    
    Path("review_result.txt").write_text(result.content, encoding="utf-8")
    print("✅ Code review completed and saved to review_result.txt")
    return 0


async def issue_triage(
    review_path: Path,
    cerebras_path: Path | None = None,
    open_issues_path: Path | None = None,
) -> int:
    if not review_path.exists():
        print(f"Error: Review file {review_path} not found.", file=sys.stderr)
        return 1
    openrouter_review = review_path.read_text(encoding="utf-8")
    cerebras_review = ""
    if cerebras_path and cerebras_path.exists():
        cerebras_review = cerebras_path.read_text(encoding="utf-8")

    combined_review = f"OpenRouter Review:\n{openrouter_review}\n\nCerebras Review:\n{cerebras_review}"

    open_issues_section = "None known."
    if open_issues_path and open_issues_path.exists():
        try:
            import json as _json

            open_issues = _json.loads(open_issues_path.read_text(encoding="utf-8"))
            if open_issues:
                open_issues_section = "\n".join(
                    f"- #{item['number']}: {item['title']}" for item in open_issues
                )
        except Exception:
            pass

    from src.core.llm_manager import MultiModelOrchestrator, TaskType
    import datetime

    ensure_config_loaded()
    orchestrator = MultiModelOrchestrator()
    current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    prompt = f"""You are an elite architectural and security reviewer for SparkleForge. Based on the review, generate a JSON object with 'should_create_issue', 'title', and 'body' keys.
CRITICAL INSTRUCTION: DO NOT create an issue for stylistic changes, minor nitpicks, or uncertain hallucinations. Set should_create_issue to true ONLY if the review identifies a concrete correctness bug, security vulnerability, workflow failure, or critical architectural debt.
DEDUPLICATION: The following issues are currently OPEN in this repository:
{open_issues_section}
If the finding you would report is already substantially covered by one of the open issues above (even if it would be worded differently), set should_create_issue to false — do not file a near-duplicate.
If should_create_issue is true, the title MUST use the repository's plain Conventional Commit prefix style (e.g. 'fix: ...', 'feat: ...').
The 'body' MUST be a Markdown string that explicitly includes the following structured header at the very top:
> **Date**: {current_date}
> **Issue Type**: [Classify as Parent Issue or Sub-issue (Anvil Phase A)]
> **Related Milestone**: [Mention if related to a known milestone like #13 Anvil, otherwise N/A]
> **Metrics & Justification**: [Provide a concrete, logical proof or metric of why this code must be changed. Do not use simple judgments.]

Below the header, provide a highly specific explanation of what is wrong and an actionable plan to fix it.
If should_create_issue is false, title and body must be empty strings.

Review Content:
{combined_review}"""

    system_message = "You are an elite triage agent. Return only a JSON object."
    
    result = await orchestrator.execute_with_model(
        prompt=prompt,
        task_type=TaskType.ANALYSIS,
        system_message=system_message,
        use_cascade=False
    )
    
    raw_content = result.content.strip()
    if raw_content.startswith("```"):
        lines = raw_content.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        raw_content = "\n".join(lines).strip()
        
    try:
        import json
        json.loads(raw_content)
        Path("triage_result.json").write_text(raw_content, encoding="utf-8")
        print("✅ Triage completed and saved to triage_result.json")
        return 0
    except Exception as je:
        print(f"Error: Invalid JSON response: {je}", file=sys.stderr)
        print(f"Raw response:\n{result.content}", file=sys.stderr)
        return 1


async def merge_decision(pr_meta_path: Path, review_path: Path, cerebras_path: Path | None = None) -> int:
    if not pr_meta_path.exists():
        print(f"Error: PR metadata file {pr_meta_path} not found.", file=sys.stderr)
        return 1
    pr_meta = pr_meta_path.read_text(encoding="utf-8")
    
    openrouter_review = ""
    if review_path.exists():
        openrouter_review = review_path.read_text(encoding="utf-8")
        
    cerebras_review = ""
    if cerebras_path and cerebras_path.exists():
        cerebras_review = cerebras_path.read_text(encoding="utf-8")
        
    from src.core.llm_manager import MultiModelOrchestrator, TaskType
    import json

    ensure_config_loaded()
    orchestrator = MultiModelOrchestrator()
    prompt = f"""You are the final merge gate for a dev branch. Return a JSON object with keys should_merge and reason. Set should_merge to true only when the reviews do not identify concrete correctness, security, workflow, packaging, or test failures that require code changes. Set should_merge to false for unresolved risks, failing checks, unclear generated changes, or any concrete issue. Do not block solely because one provider skipped or returned no content if another review is available. Keep reason concise.

PR Metadata:
{pr_meta}

OpenRouter Review:
{openrouter_review}

Cerebras Review:
{cerebras_review}"""

    system_message = "You are a merge decision gate. Return only a JSON object."
    
    result = await orchestrator.execute_with_model(
        prompt=prompt,
        task_type=TaskType.VERIFICATION,
        system_message=system_message,
        use_cascade=False
    )
    
    raw_content = result.content.strip()
    if raw_content.startswith("```"):
        lines = raw_content.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        raw_content = "\n".join(lines).strip()
        
    try:
        json.loads(raw_content)
        Path("merge_decision.json").write_text(raw_content, encoding="utf-8")
        print("✅ Merge decision completed and saved to merge_decision.json")
        return 0
    except Exception as je:
        print(f"Error: Invalid JSON response: {je}", file=sys.stderr)
        print(f"Raw response:\n{result.content}", file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["fix-issue", "code-review", "issue-triage", "merge-decision"])
    parser.add_argument("--issue-context", default="issue-context.md")
    parser.add_argument("--extra-context", default=None)
    parser.add_argument("--diff", default="diff.txt")
    parser.add_argument("--review-file", default="review_result.txt")
    parser.add_argument("--cerebras-file", default="cerebras_result.txt")
    parser.add_argument("--pr-meta-file", default="pr_meta.json")
    parser.add_argument("--open-issues-file", default=None)
    args = parser.parse_args()

    if args.command == "fix-issue":
        extra_context = Path(args.extra_context) if args.extra_context else None
        return asyncio.run(fix_issue(Path(args.issue_context), extra_context))
    elif args.command == "code-review":
        return asyncio.run(code_review(Path(args.diff)))
    elif args.command == "issue-triage":
        cerebras_file = Path(args.cerebras_file) if args.cerebras_file else None
        open_issues_file = Path(args.open_issues_file) if args.open_issues_file else None
        return asyncio.run(issue_triage(Path(args.review_file), cerebras_file, open_issues_file))
    elif args.command == "merge-decision":
        cerebras_file = Path(args.cerebras_file) if args.cerebras_file else None
        return asyncio.run(merge_decision(Path(args.pr_meta_file), Path(args.review_file), cerebras_file))

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
