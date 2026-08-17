"""
tests/test_verification_gate_scripts.py — Anvil Phase Σ-4 auto-fix verification gate (#511).

Covers the pure logic behind two merge-gate scripts:
- scripts/check_import_smoke.py
- scripts/check_no_diff_prefix_paths.py

The third, scripts/check_issue_scope_overlap.py, was moved into
src/core/ci/fix_substantiality.py (its logic now lives alongside the
checklist/file-type heuristic it complements) -- see
tests/test_ci_fix_substantiality.py for its coverage.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from check_import_smoke import check_imports  # noqa: E402
from check_no_diff_prefix_paths import find_diff_prefix_paths  # noqa: E402


class TestImportSmoke:
    def test_real_core_modules_import_cleanly(self):
        failures = check_imports(["main", "src.core.mcp_integration", "src.core.agent_harness"])
        assert failures == []

    def test_broken_module_is_reported_with_traceback(self):
        failures = check_imports(["this_module_does_not_exist_anywhere"])
        assert len(failures) == 1
        name, tb = failures[0]
        assert name == "this_module_does_not_exist_anywhere"
        assert "ModuleNotFoundError" in tb or "ImportError" in tb


class TestNoDiffPrefixPaths:
    def test_normal_paths_pass(self):
        assert find_diff_prefix_paths(["src/core/mcp/__init__.py", "tests/test_foo.py"]) == []

    def test_stray_b_prefixed_path_is_flagged(self):
        offenders = find_diff_prefix_paths(
            ["src/core/mcp/__init__.py", "b/src/core/mcp/openrouter_client.py"]
        )
        assert offenders == ["b/src/core/mcp/openrouter_client.py"]

    def test_stray_a_prefixed_path_is_flagged(self):
        assert find_diff_prefix_paths(["a/tests/test_foo.py"]) == ["a/tests/test_foo.py"]

    def test_legitimate_paths_starting_with_a_or_b_word_are_not_flagged(self):
        # Only an exact path component of "a" or "b" should trigger, not any
        # directory that happens to start with those letters.
        assert find_diff_prefix_paths(["agents/foo.py", "backend/bar.py"]) == []
