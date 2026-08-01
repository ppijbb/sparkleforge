"""Regression guard for PromptRouter._tokenize (src/core/prompt_router.py).

The tokenizer regex was ASCII-only ([a-z0-9_:+.-]+), so any non-ASCII
prompt (Korean, etc.) tokenized to an empty tuple. Every route candidate
scores 0 against an empty prompt, so the REPL silently fell through to
the "research" catch-all for every non-English query regardless of intent.
"""

from src.core.prompt_router import PromptRouter


def test_tokenize_handles_korean_text() -> None:
    tokens = PromptRouter()._tokenize("현재 프로젝트 분석")
    assert tokens == ("현재", "프로젝트", "분석")


def test_tokenize_still_handles_ascii_identifiers() -> None:
    tokens = PromptRouter()._tokenize("session list mcp:git_status file.py")
    assert tokens == ("session", "list", "mcp:git_status", "file.py")
