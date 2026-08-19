"""Issue #1509: ActionJournal._load()'s per-line handler only caught
json.JSONDecodeError, so a line that's syntactically valid JSON but fails
JournalEntry(**data) (e.g. a missing required field, or a torn read racing
a concurrent writer) raised TypeError/ValueError instead, escaped to the
outer handler, and silently aborted loading every entry after it."""

import json

from src.core.guard.action_journal import ActionJournal


def test_load_skips_a_line_that_fails_entry_validation_and_keeps_loading(tmp_path, caplog):
    journal_path = tmp_path / "action_journal.jsonl"
    valid_entry = {
        "entry_id": "e1",
        "agent_id": "system",
        "action": "read_file",
        "description": "test",
        "metadata": {},
    }
    lines = [
        json.dumps({"entry_id": "missing-required-fields"}),  # valid JSON, invalid JournalEntry
        json.dumps(valid_entry),
    ]
    journal_path.write_text("\n".join(lines) + "\n")

    import logging

    with caplog.at_level(logging.WARNING):
        journal = ActionJournal(journal_path=str(journal_path), _force_new=True)

    assert len(journal._entries) == 1
    assert journal._entries[0].entry_id == "e1"
    assert any("Skipping corrupt journal line" in r.message for r in caplog.records)
    assert not any("Failed to load journal" in r.message for r in caplog.records)
