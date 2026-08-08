"""When a model guesses a directory name with a made-up extension (e.g. reads
"wal.rs" for the actual directory "wal/"), read_file must fall back to
listing the real directory instead of raising FileNotFoundError.

Regression for: coworker sessions on directory-per-module codebases (see
issue #1261) silently gave up on every such module and fabricated its
description from the name alone, because read_file's failure had no recovery
path and the model never retried with list_files.
"""

import pytest

from src.core.mcp_integration.executors.file import _execute_file_tool


@pytest.mark.asyncio
async def test_read_file_on_guessed_extension_falls_back_to_list_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    module_dir = tmp_path / "wal"
    module_dir.mkdir()
    (module_dir / "segment.rs").write_text("pub struct Segment;")

    result = await _execute_file_tool("filesystem", {"operation": "read", "path": "wal.rs"})

    assert result.success
    assert result.data["directory"].endswith("wal")
    assert any(f["name"] == "segment.rs" for f in result.data["files"])


@pytest.mark.asyncio
async def test_read_file_still_fails_when_nothing_matches(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = await _execute_file_tool("filesystem", {"operation": "read", "path": "missing.rs"})

    assert not result.success
    assert "not found" in result.error.lower()
