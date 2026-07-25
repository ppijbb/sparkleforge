"""OS Integrity Proof Suite (issue #910).

Issue #715 audited whether the Anvil OS planes actually do what their
architecture docs claim in production -- not just in unit tests -- and found
9+ instances of the same failure shape: a governance component exists, is
exercised only by test fixtures, and is never invoked on the real execution
path (e.g. #777 CapabilityManager grants, #779 ModeController plan_first,
#780 TaskDashboard, #775 SessionControl). Each was fixed individually, but
nothing continuously re-checks that the fix holds as new code lands.

This suite drives the *real* production entry points identified by that
audit -- real constructors, not `object.__new__` bypasses; real
`InvocationGateway`/`CapabilityManager`/`ActionJournal`/`TaskDashboard`/
`SessionControl` APIs, not mocks -- and asserts their production-critical
side effects actually occur. It is the regression gate for the #715 bug
class, and the concrete artifact backing the "Agent OS" claim: green here
means the governance layer is live today, not just present in source.
"""

import os

import pytest

from src.core.guard.action_journal import ActionJournal
from src.core.guard.anomaly_detector import AnomalyDetector
from src.core.guard.capability_manager import CapabilityManager
from src.core.guard.invocation_gateway import (
    SYSTEM_ACTOR,
    InvocationGateway,
    InvocationKind,
)
from src.core.session_control import SessionControl, TaskStatus
from src.core.surface.task_dashboard import TaskDashboard


@pytest.fixture(autouse=True)
def reset_singletons():
    """Isolate each test from the module-level singletons these components use."""
    CapabilityManager._instance = None
    ActionJournal._instance = None
    AnomalyDetector._instance = None
    TaskDashboard._instance = None
    yield
    CapabilityManager._instance = None
    ActionJournal._instance = None
    AnomalyDetector._instance = None
    TaskDashboard._instance = None


def _real_gateway(tmp_path):
    """Construct a real InvocationGateway wired to tmp-scoped, real (non-mock) backends."""
    cm = CapabilityManager(state_path=str(tmp_path / "caps.json"))
    journal = ActionJournal(journal_path=str(tmp_path / "journal.jsonl"), _force_new=True)
    gateway = InvocationGateway(capability_manager=cm, action_journal=journal)
    return gateway, cm, journal


def test_gateway_bootstrap_grant_is_real_not_just_grantable(tmp_path):
    """#777: SYSTEM_ACTOR must actually hold its default capabilities the
    moment a gateway is constructed -- the original bug was that
    CapabilityManager.grant_agent existed and was unit-tested, but no
    production code path ever called it, so agent_has() always returned
    False for real agents.
    """
    _, cm, _ = _real_gateway(tmp_path)
    for capability in ("execute_shell", "write_file", "read_file", "network_request"):
        assert cm.agent_has(SYSTEM_ACTOR, capability), (
            f"SYSTEM_ACTOR is missing '{capability}' immediately after gateway "
            "construction -- the #777 bootstrap grant is not firing"
        )


def test_gateway_authorize_allows_and_journals_together(tmp_path):
    """The guard decision and the audit trail must be the same real call --
    #715's finding was that components which look wired in isolation can
    still be disconnected from each other in the actual call path.
    """
    gateway, _, journal = _real_gateway(tmp_path)

    decision = gateway.authorize(
        kind=InvocationKind.MCP_TOOL,
        actor=SYSTEM_ACTOR,
        target="execute_shell",
        description="os-plane-integrity-proof smoke check",
        required_capability="execute_shell",
    )

    assert decision.allowed
    entries = journal.recent(limit=10)
    assert any("execute_shell" in e.description and "ALLOWED" in e.description for e in entries), (
        "authorize() reported allowed=True but no matching entry reached the "
        "real ActionJournal -- the guard decision and the audit trail have "
        "come apart"
    )


def test_gateway_denies_and_journals_when_capability_missing(tmp_path):
    """A denial must be just as real and just as journaled as an approval --
    otherwise the audit trail only tells half the story."""
    gateway, cm, journal = _real_gateway(tmp_path)
    cm.revoke_agent(SYSTEM_ACTOR, "execute_shell")

    decision = gateway.authorize(
        kind=InvocationKind.MCP_TOOL,
        actor=SYSTEM_ACTOR,
        target="execute_shell",
        description="should be denied",
        required_capability="execute_shell",
    )

    assert not decision.allowed
    entries = journal.recent(limit=10)
    assert any("DENIED" in e.description for e in entries)


def test_task_dashboard_snapshot_reflects_real_submissions():
    """#780: TaskDashboard.snapshot() is what the CLI and web surface both
    read from -- if submit()/start()/complete() don't land in it, both
    surfaces silently show nothing, exactly as #780 found.
    """
    dashboard = TaskDashboard()
    task = dashboard.submit(
        name="proof-suite task",
        description="os plane integrity check",
        agent_id="proof_agent",
    )
    dashboard.start(task.task_id)
    dashboard.complete(task.task_id, result={"ok": True})

    snapshot = dashboard.snapshot()
    matching = [t for t in snapshot["tasks"] if t["task_id"] == task.task_id]
    assert len(matching) == 1, "submitted task did not reach TaskDashboard.snapshot()"
    assert matching[0]["status"] == "success"
    assert snapshot["summary"]["total"] >= 1


def test_session_control_registers_and_tracks_real_task_state():
    """#775: SessionControl's per-task tracking must reflect real
    register_task()/update_task_status() calls, not just no-op against them.
    """
    control = SessionControl()
    session_id = "os-plane-proof-session"

    control.register_task(
        session_id=session_id,
        task_id="task-1",
        task_type="proof_check",
        description="os plane integrity check",
    )
    control.update_task_status(session_id, "task-1", TaskStatus.RUNNING)

    task_info = control.get_task(session_id, "task-1")
    assert task_info is not None, "registered task is not retrievable -- tracking is a no-op"
    assert task_info.status == TaskStatus.RUNNING


async def test_agent_harness_constructs_real_wiring(monkeypatch, tmp_path):
    """Construct AgentHarness through its actual __init__ (not
    object.__new__, the pattern every existing harness/loop test uses to
    sidestep the constructor) and assert the components #715 flagged are the
    real, live objects the harness will use during execution -- catching a
    'the field is never assigned' bug (the same shape as this PR's
    VerificationNode._mode_controller fix) before it reaches production.
    """
    monkeypatch.setenv("LLM_MODEL", "gemini/gemini-2.0-flash-lite")
    monkeypatch.setenv("GOOGLE_API_KEY", "proof-suite-placeholder-key")
    monkeypatch.chdir(tmp_path)

    from src.core.anvil.mode_controller import ModeController
    from src.core.researcher_config import load_config_from_env
    from src.core.tools.registry import registry

    load_config_from_env()

    from src.core.agent_harness import AgentHarness

    harness = AgentHarness()
    try:
        assert isinstance(harness.mode_controller, ModeController)
        assert isinstance(harness.dashboard, TaskDashboard)
        # _register_tools() must have actually run against the real registry,
        # not merely be reachable in isolation.
        for tool_name in ("quarantine_file", "revoke_capability", "control_iot_device"):
            assert registry.get_tool_info(tool_name) is not None, (
                f"AgentHarness.__init__ did not register '{tool_name}' into the "
                "real tool registry"
            )
    finally:
        await harness.aclose()


def test_anomaly_detector_has_no_reachable_single_node_entrypoint():
    """Known, tracked gap (found while building this suite, not yet fixed):
    AnomalyDetector.observe() is only ever called from
    GuardPlane.check_and_execute(), whose only in-repo caller is
    WorkerNode.handle_execute() -- and WorkerNode is only ever instantiated in
    tests/test_coordinator.py, never in production code. In a single-node run
    (the CLI / AgentHarness / AgentLoop path this repo actually ships),
    nothing calls AnomalyDetector.observe() at all.

    This pins that fact as a scan over `src/`, not a docstring claim: it
    fails the day a real (non-test) call site constructs a WorkerNode, which
    is exactly when someone should come update this test (and the anomaly
    detector's reachability) instead of the gap silently persisting or
    silently closing unnoticed.
    """
    import ast
    from pathlib import Path

    src_root = Path(__file__).resolve().parent.parent / "src"
    real_instantiation_sites = []

    for path in src_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "WorkerNode"
            ):
                real_instantiation_sites.append(f"{path}:{node.lineno}")

    assert real_instantiation_sites == [], (
        "WorkerNode is now constructed in production code "
        f"({real_instantiation_sites}) -- GuardPlane.check_and_execute() and "
        "AnomalyDetector may have gained a real single-node entrypoint; "
        "update/replace this test to exercise it directly instead of "
        "asserting it stays unreachable."
    )
