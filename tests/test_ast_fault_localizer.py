"""Tests for AST fault localizer and context slicer."""

import tempfile
from pathlib import Path
from src.core.analysis.ast_localizer import ASTFaultLocalizer


def test_ast_fault_localizer_basic():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        sample_py = root / "calculator.py"
        sample_py.write_text(
            "class Calculator:\n"
            "    """A simple calculator class."""\n"
            "    def add(self, a: int, b: int) -> int:\n"
            "        return a + b\n\n"
            "    def subtract(self, a: int, b: int) -> int:\n"
            "        return a - b\n"
        )

        localizer = ASTFaultLocalizer(codebase_path=root, token_budget=2000)
        localizer.index_codebase()

        result = localizer.localize("Calculator add", top_k=2)
        
        assert result["latency_ms"] <= 300.0
        assert result["token_count"] <= 2000
        assert result["within_budget"] is True
        assert len(result["slices"]) > 0
        
        found_add = any("add" in s["symbol"] for s in result["slices"])
        assert found_add, "Should recall the target add method"


def test_token_budget_cap():
    localizer = ASTFaultLocalizer(token_budget=500)
    assert localizer.token_budget == 500
    # Test empty or non-existent codebase indexing
    localizer.index_codebase()
    res = localizer.localize("nonexistent")
    assert res["token_count"] == 0
    assert res["within_budget"] is True
