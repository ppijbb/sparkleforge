"""Unit tests for CodebaseAgent attributes, analysis, and token optimization."""

import asyncio

from src.agents.codebase_agent import CodebaseAgent


def test_codebase_agent_defaults():
    agent = CodebaseAgent()
    assert agent.deep_swe_mode is True
    assert agent.token_minimizer_active is True


def test_optimize_token_overhead_filters_debug_lines():
    agent = CodebaseAgent()
    raw = "INFO start\nDEBUG internal detail\n  DEBUG indented debug\nERROR keep error\n"
    optimized = agent.optimize_token_overhead(raw)
    assert optimized == "INFO start\nERROR keep error"


def test_optimize_token_overhead_respects_flag_and_empty_input():
    agent = CodebaseAgent()
    assert agent.optimize_token_overhead("") == ""
    agent.token_minimizer_active = False
    raw = "DEBUG keep me\nINFO line"
    assert agent.optimize_token_overhead(raw) == raw


def test_analyze_codebase_parses_python_file(tmp_path):
    sample = tmp_path / "app.py"
    sample.write_text("import os\n\nclass Greeter:\n    pass\n\ndef main():\n    return 1\n", encoding="utf-8")

    agent = CodebaseAgent(tmp_path)
    structure = asyncio.run(agent.analyze_codebase())

    assert "app.py" in structure.files
    parsed = structure.files["app.py"]
    assert parsed.classes == ["Greeter"]
    assert parsed.functions == ["main"]
    assert "os" in parsed.imports
