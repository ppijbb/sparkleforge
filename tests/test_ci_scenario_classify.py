from src.core.ci.scenario_classify import classify_scenario_outcome


def test_healthy_run_is_not_an_infra_outage():
    report = {
        "overall_score": 0.8,
        "scenarios": {
            "s1": {"stdout_excerpt": "all good"},
            "s2": {"stdout_excerpt": "all good"},
        },
    }
    outcome = classify_scenario_outcome(report)
    assert outcome.is_infra_outage is False
    assert outcome.infra_failed == 0
    assert outcome.total == 2


def test_low_overall_score_alone_triggers_infra_outage():
    report = {"overall_score": 0.1, "scenarios": {"s1": {"stdout_excerpt": "real failure"}}}
    outcome = classify_scenario_outcome(report)
    assert outcome.is_infra_outage is True


def test_high_infra_failure_ratio_triggers_infra_outage():
    # 10/11 infra-failed => ratio > 0.9, even though overall_score alone
    # wouldn't cross the 0.3 threshold.
    scenarios = {f"s{i}": {"stdout_excerpt": "No available models."} for i in range(10)}
    scenarios["s_ok"] = {"stdout_excerpt": "unrelated pass"}
    report = {"overall_score": 0.5, "scenarios": scenarios}
    outcome = classify_scenario_outcome(report)
    assert outcome.infra_failed == 10
    assert outcome.total == 11
    assert outcome.is_infra_outage is True


def test_moderate_score_and_low_infra_ratio_is_not_outage():
    report = {
        "overall_score": 0.6,
        "scenarios": {
            "s1": {"stdout_excerpt": "No available models."},
            "s2": {"stdout_excerpt": "real regression, not infra"},
            "s3": {"stdout_excerpt": "real regression, not infra"},
            "s4": {"stdout_excerpt": "real regression, not infra"},
        },
    }
    outcome = classify_scenario_outcome(report)
    assert outcome.infra_ratio == 0.25
    assert outcome.is_infra_outage is False


def test_no_scenarios_defaults_infra_ratio_to_one():
    report = {"overall_score": 0.0, "scenarios": {}}
    outcome = classify_scenario_outcome(report)
    assert outcome.total == 0
    assert outcome.infra_ratio == 1.0
    assert outcome.is_infra_outage is True
