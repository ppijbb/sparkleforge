"""Issue #1152: agent_workflow_result_to_public_dict was dropped from
agent_orchestrator.py during a refactor, breaking anything still importing
the old public API by name. It must keep its original signature (an
optional `context` kwarg) and its original legacy field mapping -- not just
echo the input dict back unchanged.
"""

from src.core.agent_orchestrator import agent_workflow_result_to_public_dict


def test_maps_internal_result_to_legacy_public_shape():
    internal_result = {
        "plan": "do the thing",
        "tasks": ["a", "b"],
        "results": "done",
        "success": True,
        "detailed_results": {"internal": "detail"},
        "metadata": {"internal": "meta"},
    }

    public = agent_workflow_result_to_public_dict(internal_result)

    assert public == {
        "plan": "do the thing",
        "tasks": ["a", "b"],
        "results": "done",
        "success": True,
    }


def test_accepts_optional_context_kwarg_for_signature_compatibility():
    # Old callers pass `context=` positionally/by keyword; the function must
    # not raise TypeError just because a caller still supplies it.
    result = agent_workflow_result_to_public_dict({"success": True}, context={"foo": "bar"})
    assert result["success"] is True


def test_missing_fields_default_sensibly():
    assert agent_workflow_result_to_public_dict({}) == {
        "plan": "",
        "tasks": [],
        "results": "",
        "success": False,
    }
