from pathlib import Path

from scripts.opencode_github_worker import _apply_patch, _normalize_diff
from src.core.patch_ops import _validate_patch_paths


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    # worker script so src/core/nightshift can reuse it without duplication).
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
