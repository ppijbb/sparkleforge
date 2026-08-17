from pathlib import Path

from src.core.ci.todo_inventory import (
    Priority,
    categorize_todo,
    determine_priority,
    extract_todos,
    generate_json_inventory,
)


def test_determine_priority_critical_keyword():
    assert determine_priority("fix this security hole", "src/core/foo.py") == Priority.CRITICAL


def test_determine_priority_high_keyword():
    assert determine_priority("improve api integration", "src/core/foo.py") == Priority.HIGH


def test_determine_priority_medium_keyword():
    assert determine_priority("refactor this later", "src/utils/foo.py") == Priority.MEDIUM


def test_determine_priority_falls_back_to_high_for_core_files():
    assert determine_priority("nothing special here", "src/core/orchestrator/graph.py") == Priority.HIGH


def test_determine_priority_defaults_to_low():
    assert determine_priority("nothing special here", "src/utils/misc.py") == Priority.LOW


def test_categorize_todo_memory():
    assert categorize_todo("fix the memory cache") == "Memory/Storage"


def test_categorize_todo_other_fallback():
    assert categorize_todo("just do something") == "Other"


def test_extract_todos_from_file(tmp_path):
    project_root = tmp_path
    src_file = tmp_path / "src" / "core" / "foo.py"
    src_file.parent.mkdir(parents=True)
    src_file.write_text(
        "x = 1\n"
        "# TODO: fix the security issue here\n"
        "y = 2\n"
        "# FIXME improve api integration\n",
        encoding="utf-8",
    )

    todos = extract_todos(src_file, project_root)
    assert len(todos) == 2
    assert todos[0].file_path == "src/core/foo.py"
    assert todos[0].line_number == 2
    assert todos[0].priority == Priority.CRITICAL
    assert todos[1].line_number == 4
    assert todos[1].priority == Priority.HIGH


def test_generate_json_inventory_groups_by_priority_and_category():
    from src.core.ci.todo_inventory import TodoItem

    todos = [
        TodoItem("a.py", 1, "security bug", Priority.CRITICAL, "Other", "TODO"),
        TodoItem("b.py", 2, "cleanup", Priority.MEDIUM, "Refactoring", "FIXME"),
    ]
    inventory = generate_json_inventory(todos)
    assert inventory["total_count"] == 2
    assert len(inventory["by_priority"]["Critical"]) == 1
    assert len(inventory["by_priority"]["Medium"]) == 1
    assert "Refactoring" in inventory["by_category"]
