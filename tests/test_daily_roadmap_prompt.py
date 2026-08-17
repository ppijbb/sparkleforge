from src.core.daily_roadmap import build_daily_roadmap_mission_brief


def test_mission_brief_contains_required_sections_and_today() -> None:
    brief = build_daily_roadmap_mission_brief("2026-08-16")

    assert "2026-08-16" in brief
    for section in (
        "## Why now",
        "## Proposed change",
        "## Implementation notes",
        "## Acceptance criteria",
        "## Validation",
    ):
        assert section in brief
