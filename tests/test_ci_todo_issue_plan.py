from src.core.ci.todo_issue_plan import known_anchors, plan_todo_issues


def _inventory(*items):
    return {"all_items": list(items)}


def _item(file="src/core/foo.py", line=10, priority="Critical", category="Other", issue_type="TODO", content="fix this"):
    return {"file": file, "line": line, "priority": priority, "category": category, "issue_type": issue_type, "content": content}


def test_known_anchors_extracted_from_existing_issue_bodies():
    existing = [
        {"body": "<!-- todo-debt:src/core/foo.py:10 -->\nsome text"},
        {"body": "no anchor here"},
    ]
    assert known_anchors(existing) == {"src/core/foo.py:10"}


def test_plans_critical_and_high_priority_items():
    inventory = _inventory(_item(priority="Critical"), _item(line=20, priority="High"), _item(line=30, priority="Medium"))
    plan = plan_todo_issues(inventory, [])
    assert len(plan) == 2
    assert {p.anchor for p in plan} == {"src/core/foo.py:10", "src/core/foo.py:20"}


def test_skips_low_and_medium_priority():
    inventory = _inventory(_item(priority="Low"), _item(priority="Medium"))
    plan = plan_todo_issues(inventory, [])
    assert plan == []


def test_dedups_against_known_anchor():
    inventory = _inventory(_item(priority="Critical"))
    existing = [{"body": "<!-- todo-debt:src/core/foo.py:10 -->"}]
    plan = plan_todo_issues(inventory, existing)
    assert plan == []


def test_planned_issue_title_is_lowercase_conventional_commit_style():
    inventory = _inventory(_item(file="src/Core/Foo.py", issue_type="FIXME"))
    plan = plan_todo_issues(inventory, [])
    assert plan[0].title == "chore: resolve fixme at src/core/foo.py:10"


def test_planned_issue_body_contains_anchor_and_location():
    inventory = _inventory(_item())
    plan = plan_todo_issues(inventory, [])
    body = plan[0].body
    assert "<!-- todo-debt:src/core/foo.py:10 -->" in body
    assert "- File: `src/core/foo.py`" in body
    assert "- Line: 10" in body
