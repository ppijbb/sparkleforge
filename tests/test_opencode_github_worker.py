from pathlib import Path

from scripts.opencode_github_worker import _apply_patch, _normalize_diff


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
