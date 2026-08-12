"""Issue #1338: edit_file's "Old string not found" error gave no clue why the
match failed (whitespace? wrong file content? stale read?), so the model's
only recovery path was a full re-read -- an extra iteration the momentum
guard (#1307) then flags as a stall. The error should carry enough of the
actual file content for the model to retry in one shot.
"""

import pytest

from src.core.mcp_integration.executors.file import _execute_file_tool


@pytest.mark.asyncio
async def test_edit_mismatch_includes_closest_match_with_line_numbers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "buffer.rs"
    target.write_text("fn flush(&mut self) {\n    self.buf.clear();\n}\n")

    # Same code, but with a trailing space the model didn't know about --
    # a plausible real "Old string not found" cause (issue's Rust example).
    result = await _execute_file_tool(
        "edit_file",
        {
            "file_path": str(target),
            "old_string": "fn flush(&mut self) {\n    self.buf.clear(); \n}\n",
            "new_string": "fn flush(&mut self) {\n    self.buf.clear();\n    self.dirty = false;\n}\n",
        },
    )

    assert not result.success
    assert "Old string not found" in result.error
    assert "Closest match in file" in result.error
    assert "self.buf.clear();" in result.error
    # Line numbers let the model locate the mismatch without a fresh read.
    assert "1: fn flush" in result.error


@pytest.mark.asyncio
async def test_edit_mismatch_with_no_similar_text_shows_file_start(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "empty_ish.py"
    target.write_text("x = 1\ny = 2\n")

    result = await _execute_file_tool(
        "edit_file",
        {
            "file_path": str(target),
            "old_string": "completely unrelated content that appears nowhere",
            "new_string": "z = 3",
        },
    )

    assert not result.success
    assert "No similar text found" in result.error
    assert "x = 1" in result.error


@pytest.mark.asyncio
async def test_successful_edit_is_unaffected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "ok.py"
    target.write_text("a = 1\n")

    result = await _execute_file_tool(
        "edit_file",
        {"file_path": str(target), "old_string": "a = 1", "new_string": "a = 2"},
    )

    assert result.success
    assert target.read_text() == "a = 2\n"
