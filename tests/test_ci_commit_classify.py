from src.core.ci.commit_classify import classify_conventional_commit


def test_plain_title_defaults_to_fix():
    result = classify_conventional_commit("Something broke in the parser")
    assert result.type == "fix"
    assert result.subject == "something broke in the parser"


def test_bare_feat_prefix():
    result = classify_conventional_commit("feat: add new dashboard widget")
    assert result.type == "feat"
    assert result.subject == "add new dashboard widget"


def test_capitalized_fix_prefix():
    result = classify_conventional_commit("Fix: the login button is broken")
    assert result.type == "fix"
    assert result.subject == "the login button is broken"


def test_emoji_refactor_prefix():
    result = classify_conventional_commit("♻️ refactor: simplify the retry loop")
    assert result.type == "refactor"
    assert result.subject == "simplify the retry loop"


def test_bracket_tag_feature_maps_to_feat():
    result = classify_conventional_commit("[feature] Add dark mode toggle")
    assert result.type == "feat"
    assert result.subject == "add dark mode toggle"


def test_bracket_tag_unknown_type_maps_to_chore():
    result = classify_conventional_commit("[weird] Some odd label")
    assert result.type == "chore"
    assert result.subject == "some odd label"


def test_bracket_tag_outside_recognized_second_pass_types_leaves_prefix_embedded():
    # Known pre-existing quirk (see module docstring): "docs" isn't one of the
    # types the second-pass prefix match recognizes, so the "docs: " prefix
    # from the bracket normalization survives into the subject unstripped.
    result = classify_conventional_commit("[docs] Update the README")
    assert result.type == "docs"
    assert result.subject == "docs: update the readme"


def test_strips_leading_date_prefix():
    result = classify_conventional_commit("fix: 2026-01-02 - the scheduler drifted")
    assert result.subject == "the scheduler drifted"


def test_strips_trailing_issue_reference():
    result = classify_conventional_commit("fix: the scheduler drifted (#123)")
    assert result.subject == "the scheduler drifted"


def test_collapses_internal_whitespace():
    result = classify_conventional_commit("fix:   too    many     spaces")
    assert result.subject == "too many spaces"


def test_strips_embedded_issue_reference_not_just_trailing():
    result = classify_conventional_commit(
        "pre-existing test failures on main (baseline, unrelated to #1365)"
    )
    assert "#" not in result.subject


def test_strips_leading_quote_punctuation():
    result = classify_conventional_commit(
        "'Failed to load journal: Expecting value...' warning prints on every "
        "single CLI invocation before any real output"
    )
    assert result.subject[0].isalnum()
    assert result.subject.startswith("failed to load journal")
