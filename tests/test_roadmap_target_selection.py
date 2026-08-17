from src.core.roadmap.target_selection import (
    AnvilTarget,
    render_planning_context,
    select_anvil_target,
    target_file_contents,
)


def test_no_milestone_returns_not_found():
    target = select_anvil_target(None, [])
    assert target == AnvilTarget(None, None)
    assert not target.found


def test_picks_first_open_sub_issue():
    sub_status = [
        {"number": 10, "title": "closed one", "state": "CLOSED"},
        {"number": 11, "title": "next up", "state": "OPEN"},
        {"number": 12, "title": "also open", "state": "OPEN"},
    ]
    target = select_anvil_target({"number": 5, "title": "Milestone: Phase M"}, sub_status)
    assert target == AnvilTarget(11, "next up")


def test_all_closed_returns_not_found():
    sub_status = [{"number": 10, "title": "done", "state": "CLOSED"}]
    target = select_anvil_target({"number": 5, "title": "Milestone: Phase M"}, sub_status)
    assert not target.found


def test_target_file_contents_format():
    assert target_file_contents(AnvilTarget(11, "next up")) == "11|next up"
    assert target_file_contents(AnvilTarget(None, None)) == ""


def test_render_planning_context_no_milestone():
    assert render_planning_context(None, []) == "- No open Anvil phase milestone found."


def test_render_planning_context_lists_sub_issues_and_target():
    milestone = {"number": 5, "title": "Milestone: Phase M"}
    sub_status = [
        {"number": 10, "title": "closed one", "state": "CLOSED"},
        {"number": 11, "title": "next up", "state": "OPEN"},
    ]
    rendered = render_planning_context(milestone, sub_status)
    assert "- Open milestone: #5 Milestone: Phase M" in rendered
    assert "  - #10 [CLOSED] closed one" in rendered
    assert "- Next open Anvil roadmap sub-issue: #11 next up" in rendered
