"""Unit tests for AdversarialEvaluator (Zero-Trust Evaluation Gate) in Forge Master."""

import pytest
from src.core.forge_master.adversarial_evaluator import AdversarialEvaluator


@pytest.mark.asyncio
async def test_adversarial_evaluator_passes_valid_code():
    evaluator = AdversarialEvaluator()
    valid_code_response = """Here is the fix:
```python
def add(a: int, b: int) -> int:
    return a + b
```
Successfully updated module.
"""
    exec_result = {
        "success": True,
        "response": valid_code_response,
        "confidence": 0.85,
    }

    result = await evaluator.evaluate_output(
        task_query="Fix addition function",
        agent_name="codex",
        execution_result=exec_result,
    )

    assert result.passed is True
    assert result.verdict == "PASSED"
    assert len(result.flaws_detected) == 0


@pytest.mark.asyncio
async def test_adversarial_evaluator_rejects_lazy_stub():
    evaluator = AdversarialEvaluator()
    lazy_response = """```python
def complex_algorithm():
    pass # TODO: implement
```"""
    exec_result = {
        "success": True,
        "response": lazy_response,
        "confidence": 0.99,
    }

    result = await evaluator.evaluate_output(
        task_query="Implement complex algorithm",
        agent_name="claude_code",
        execution_result=exec_result,
    )

    assert result.passed is False
    assert any("lazy placeholder" in flaw for flaw in result.flaws_detected)


@pytest.mark.asyncio
async def test_adversarial_evaluator_catches_invalid_syntax():
    evaluator = AdversarialEvaluator()
    broken_syntax_response = """```python
def invalid_fn(
    return 42
```"""
    exec_result = {
        "success": True,
        "response": broken_syntax_response,
        "confidence": 0.80,
    }

    result = await evaluator.evaluate_output(
        task_query="Write python function",
        agent_name="codex",
        execution_result=exec_result,
    )

    assert result.passed is False
    assert any("AST syntax parsing" in flaw for flaw in result.flaws_detected)
