"""Unit tests for TokenMinimizer in Forge Master."""

from src.core.forge_master.token_minimizer import TokenMinimizer


def test_compact_prompt():
    minimizer = TokenMinimizer()
    raw_prompt = "Please ensure that you provide a comprehensive and highly detailed response.\n\n\n\nFix bug in main.py"
    compacted = minimizer.compact_prompt(raw_prompt)

    assert "please ensure that you provide" not in compacted.lower()
    assert "Fix bug in main.py" in compacted


def test_extract_diff_context():
    minimizer = TokenMinimizer()
    orig = "def hello():\n    print('world')\n    return True"
    mod = "def hello():\n    print('SparkleForge Agent OS')\n    return True"

    diff = minimizer.extract_diff_context(orig, mod)
    assert "-     print('world')" in diff
    assert "+     print('SparkleForge Agent OS')" in diff


def test_distill_response():
    minimizer = TokenMinimizer()
    raw_output = "\x1b[32mSuccess\x1b[0m: Modified file src/main.py\nDone in 0.5s."
    distilled = minimizer.distill_response(raw_output)

    assert "\x1b[" not in distilled
    assert "Modified file src/main.py" in distilled


def test_estimate_token_reduction():
    minimizer = TokenMinimizer()
    orig = "This is a very long prompt with many unnecessary words and repetitive details that consume extra tokens."
    comp = "This is a short prompt."

    metrics = minimizer.estimate_token_reduction(orig, comp)
    assert metrics["original_word_count"] > metrics["compressed_word_count"]
    assert metrics["reduction_percentage"] > 50.0
