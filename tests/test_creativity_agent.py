"""Regression guard for src/agents/creativity_agent.py.

Two production breakages slipped through with zero test coverage on this
module: a missing `field` import (NameError at import time) and a
`bottleneck_engine` class attribute read before it was ever declared
(AttributeError at first instantiation). Both would have been caught by
just importing the module and constructing the class.
"""

from src.agents.creativity_agent import CreativityAgent


def test_creativity_agent_imports_and_constructs() -> None:
    agent = CreativityAgent()
    assert agent.bottleneck_engine is not None


def test_bottleneck_engine_is_a_shared_singleton() -> None:
    first = CreativityAgent()
    second = CreativityAgent()
    assert first.bottleneck_engine is second.bottleneck_engine
