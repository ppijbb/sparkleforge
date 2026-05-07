from scripts.opencode_github_worker import _normalize_diff


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
