from pathlib import Path

import pytest

from scripts.opencode_github_worker import (
    _apply_patch,
    _budgeted_relevant_file_contents,
    _budgeted_requested_tool_context,
    _normalize_diff,
    _parse_triage_response,
    _per_file_context_limit,
    _prompt_fits_budget,
)
from src.core.patch_ops import _split_multifile_patch, _validate_patch_paths


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DummyOpenCodeAgent:
    def __init__(self, prompt_budget: int):
        self.prompt_budget = prompt_budget

    def prompt_context_budget(self) -> int:
        return self.prompt_budget


def test_normalize_diff_repairs_incorrect_hunk_counts() -> None:
    diff = """diff --git a/.github/workflows/auto-fix.yml b/.github/workflows/auto-fix.yml
--- a/.github/workflows/auto-fix.yml
+++ b/.github/workflows/auto-fix.yml
@@ -33,6 +35,21 @@ jobs:
           fi
           gh auth status
 
+      - name: Validate issue number
+        id: validate_issue
+        run: |
+          ISSUE_NUMBER="${{ github.event.inputs.issue_number }}"
+          if [ -z "$ISSUE_NUMBER" ] || ! [[ "$ISSUE_NUMBER" =~ ^[0-9]+$ ]]; then
+            echo "::error::Invalid issue_number: '$ISSUE_NUMBER'. Must be a non-empty numeric string."
+            echo "skip=true" >> "$GITHUB_OUTPUT"
+            exit 1
+          fi
+          echo "Validating issue #$ISSUE_NUMBER exists..."
+          if ! gh issue view "$ISSUE_NUMBER" --repo "${{ github.repository }}" >/dev/null 2>&1; then
+            echo "::error::Issue #$ISSUE_NUMBER does not exist in ${{ github.repository }}."
+            echo "skip=true" >> "$GITHUB_OUTPUT"
+            exit 1
+          fi
+          echo "skip=false" >> "$GITHUB_OUTPUT"
+
       - name: Dispatch OpenCode Auto Fix PR workflow
         env:
           GH_TOKEN: ${{ secrets.PAT }}
"""

    normalized = _normalize_diff(diff)

    assert "@@ -33,6 +35,23 @@ jobs:" in normalized
    assert "          gh auth status\n \n+      - name: Validate issue number" in normalized


def test_worker_git_apply_ignores_whitespace_warnings() -> None:
    # The git-apply invocation lives in src/core/patch_ops.py (extracted from this
    # worker script so src/core/nightwelding can reuse it without duplication).
    patch_ops = (PROJECT_ROOT / "src" / "core" / "patch_ops.py").read_text(
        encoding="utf-8"
    )

    assert "--whitespace=nowarn" in patch_ops


def test_normalize_diff_adds_missing_path_prefixes() -> None:
    diff = """--- .github/workflows/auto-fix.yml
+++ .github/workflows/auto-fix.yml
@@ -1,2 +1,2 @@
 name: Auto Fix Harness
-old
+new
"""

    normalized = _normalize_diff(diff)

    assert "--- a/.github/workflows/auto-fix.yml" in normalized
    assert "+++ b/.github/workflows/auto-fix.yml" in normalized


def test_apply_patch_rejects_partial_multifile_success(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    Path("README.md").write_text("old\n", encoding="utf-8")

    patch = tmp_path / "opencode.patch"
    patch.write_text(
        """diff --git a/README.md b/README.md
--- a/README.md
+++ b/README.md
@@ -1 +1 @@
-old
+new
diff --git a/scripts/setup.sh b/scripts/setup.sh
--- a/scripts/setup.sh
+++ b/scripts/setup.sh
@@ -1 +1 @@
-old
+new
""",
        encoding="utf-8",
    )

    success, error = _apply_patch(patch)

    assert success is False
    assert "Patch only applied partially" in error
    assert "scripts/setup.sh" in error


def test_apply_patch_rejects_embedded_diff_prefix_paths(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    patch = tmp_path / "opencode.patch"
    patch.write_text(
        """diff --git a/a/tests/test_bad_path.py b/a/tests/test_bad_path.py
--- a/a/tests/test_bad_path.py
+++ b/a/tests/test_bad_path.py
@@ -0,0 +1,2 @@
+def test_bad_path():
+    assert True
""",
        encoding="utf-8",
    )

    success, error = _apply_patch(patch)

    assert success is False
    assert "diff-prefix path is embedded" in error
    assert "a/tests/test_bad_path.py" in error
    assert not (tmp_path / "a" / "tests" / "test_bad_path.py").exists()


def test_validate_patch_paths_rejects_embedded_prefix_in_file_headers() -> None:
    diff = """diff --git a/tests/test_bad_path.py b/tests/test_bad_path.py
--- a/a/tests/test_bad_path.py
+++ b/a/tests/test_bad_path.py
@@ -0,0 +1,2 @@
+def test_bad_path():
+    assert True
"""

    error = _validate_patch_paths(diff)

    assert "diff-prefix path is embedded" in error
    assert "a/tests/test_bad_path.py" in error


def test_validate_patch_paths_rejects_parent_segments_without_trailing_slash() -> None:
    diff = """diff --git a/foo/.. b/foo/..
--- a/foo/..
+++ b/foo/..
@@ -1 +1 @@
-old
+new
"""

    error = _validate_patch_paths(diff)

    assert "path escapes repository: foo/.." in error


def test_validate_patch_paths_allows_spaces_in_filenames() -> None:
    diff = """diff --git a/docs/my note.md b/docs/my note.md
--- a/docs/my note.md
+++ b/docs/my note.md
@@ -1 +1 @@
-old
+new
"""

    assert _validate_patch_paths(diff) == ""


def test_validate_patch_paths_parses_diff_git_separator_from_right() -> None:
    diff = """diff --git a/docs/foo b/bar.txt b/docs/foo b/bar.txt
--- a/docs/foo b/bar.txt
+++ b/docs/foo b/bar.txt
@@ -1 +1 @@
-old
+new
"""

    assert _validate_patch_paths(diff) == ""


def test_validate_patch_paths_rejects_embedded_prefix_with_trailing_whitespace() -> None:
    diff = (
        "diff --git a/tests/test_bad_path.py b/tests/test_bad_path.py\n"
        "--- a/a/tests/test_bad_path.py   \n"
        "+++ b/a/tests/test_bad_path.py   \n"
        "@@ -0,0 +1,2 @@\n"
        "+def test_bad_path():\n"
        "+    assert True\n"
    )

    error = _validate_patch_paths(diff)

    assert "diff-prefix path is embedded" in error
    assert "a/tests/test_bad_path.py" in error


def test_split_multifile_patch_keeps_paths_with_spaces() -> None:
    diff = """diff --git a/docs/my note.md b/docs/my note.md
--- a/docs/my note.md
+++ b/docs/my note.md
@@ -1 +1 @@
-old
+new
"""

    assert _split_multifile_patch(diff) == [("docs/my note.md", diff)]


def test_per_file_context_limit_caps_large_model_budget() -> None:
    limit = _per_file_context_limit(
        DummyOpenCodeAgent(prompt_budget=1_000_000),
        1,
        snapshot="",
        status="",
        issue_context="short issue",
    )

    assert limit == 200_000


def test_per_file_context_limit_returns_zero_when_prompt_budget_is_exhausted() -> None:
    limit = _per_file_context_limit(
        DummyOpenCodeAgent(prompt_budget=100),
        3,
        snapshot="src/core/patch_ops.py",
        status="",
        issue_context="x" * 10_000,
    )

    assert limit == 0


def test_per_file_context_limit_shares_budget_across_files() -> None:
    one_file = _per_file_context_limit(
        DummyOpenCodeAgent(prompt_budget=50_000),
        1,
        snapshot="src/core/patch_ops.py",
        status="",
        issue_context="short issue",
    )
    five_files = _per_file_context_limit(
        DummyOpenCodeAgent(prompt_budget=50_000),
        5,
        snapshot="src/core/patch_ops.py",
        status="",
        issue_context="short issue",
    )

    assert five_files < one_file
    assert five_files > 0


def test_budgeted_relevant_file_contents_shrinks_until_final_prompt_fits(tmp_path) -> None:
    source = tmp_path / "large_module.py"
    source.write_text("\n".join(f"line_{i} = 'value'" for i in range(2_000)), encoding="utf-8")
    agent = DummyOpenCodeAgent(prompt_budget=3_000)

    file_contents_str = _budgeted_relevant_file_contents(
        agent,
        [str(source)],
        snapshot="",
        status="",
        issue_context="fix the large module",
        extra_context="",
    )

    assert file_contents_str
    assert "[truncated]" in file_contents_str
    assert _prompt_fits_budget(
        agent,
        snapshot="",
        status="",
        issue_context="fix the large module",
        file_contents_str=file_contents_str,
        extra_context="",
    )


def test_budgeted_requested_tool_context_counts_existing_file_context(tmp_path) -> None:
    requested = tmp_path / "requested.py"
    requested.write_text("\n".join(f"result_{i} = {i}" for i in range(2_000)), encoding="utf-8")
    agent = DummyOpenCodeAgent(prompt_budget=3_500)
    file_contents_str = "Relevant File Contents (with exact line numbers):\n" + (
        "context line\n" * 120
    )

    tool_context = _budgeted_requested_tool_context(
        agent,
        [str(requested)],
        snapshot="",
        status="",
        issue_context="fix the requested module",
        file_contents_str=file_contents_str,
        extra_context="",
    )

    assert tool_context
    assert _prompt_fits_budget(
        agent,
        snapshot="",
        status="",
        issue_context="fix the requested module",
        file_contents_str=file_contents_str,
        extra_context="",
        tool_context=tool_context,
    )


def test_parse_triage_response_accepts_fenced_json() -> None:
    parsed = _parse_triage_response(
        """```json
{"should_create_issue": true, "title": "fix: bug", "body": "details"}
```"""
    )

    assert parsed["should_create_issue"] is True
    assert parsed["title"] == "fix: bug"


def test_parse_triage_response_defaults_to_no_issue_for_prose() -> None:
    parsed = _parse_triage_response(
        "We need to decide if any listed issue qualifies, but this is not JSON."
    )

    assert parsed == {"should_create_issue": False, "title": "", "body": ""}


def test_parse_triage_response_rejects_json_missing_required_fields() -> None:
    with pytest.raises(ValueError, match="missing required"):
        _parse_triage_response('{"title": "fix: bug", "body": "details"}')


def test_parse_triage_response_rejects_malformed_json_shape() -> None:
    with pytest.raises(ValueError, match="looked like JSON"):
        _parse_triage_response('{"should_create_issue": true, "title": "fix: bug"')
