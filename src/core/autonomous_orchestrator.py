"""LangGraph Orchestrator (v3.0 - Modularized Architecture)

Modular architecture refactored from the monolithic 165KB orchestrator.
Delegates core logic to src.core.orchestrator packages.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict
import sys
from pathlib import Path

from src.core.orchestrator import create_orchestrator_graph
from src.core.researcher_config import (
    get_agent_config,
    get_llm_config,
    get_mcp_config,
    get_research_config,
)

logger = logging.getLogger(__name__)


def _autopilot_mode_enabled(context: Dict[str, Any] | None = None) -> bool:
    """Return whether autonomous runs should avoid interactive clarification waits."""
    if context and "autopilot_mode" in context:
        return bool(context["autopilot_mode"])

    explicit = os.getenv("SPARKLEFORGE_AUTOPILOT_MODE")
    if explicit is not None:
        return explicit.lower() not in {"0", "false", "no", "off"}

    # Default to autonomous execution. Interactive clarification must be explicitly enabled
    # by setting SPARKLEFORGE_AUTOPILOT_MODE=false.
    return True


class LivenessWatchdog:
    """24x7 liveness watchdog detecting stagnation during autonomous runs.

    Tracks the most recent commit timestamp and orchestrator heartbeat. When
    the autonomous run stays inactive for longer than the configured stagnation
    threshold, the watchdog emits a liveness event so the self-recovery planner
    can restart ``research_planner`` along an alternative execution path.
    """

    DEFAULT_STAGNATION_HOURS = 1
    REPORT_DIR = Path("results/agent_reports")

    def __init__(
        self,
        stagnation_hours: float | None = None,
        report_dir: str | Path | None = None,
    ) -> None:
        env_hours = os.getenv("SPARKLEFORGE_LIVENESS_STAGNATION_HOURS")
        if stagnation_hours is not None:
            self.stagnation_hours = float(stagnation_hours)
        elif env_hours is not None:
            self.stagnation_hours = float(env_hours)
        else:
            self.stagnation_hours = float(self.DEFAULT_STAGNATION_HOURS)
        self.report_dir = Path(report_dir) if report_dir else self.REPORT_DIR
        self._last_commit_at: datetime | None = None
        self._last_heartbeat_at: datetime | None = None
        self._recovery_attempts: list[Dict[str, Any]] = []

    def record_commit(self, timestamp: datetime | None = None) -> None:
        self._last_commit_at = timestamp or datetime.now()

    def record_heartbeat(self, timestamp: datetime | None = None) -> None:
        self._last_heartbeat_at = timestamp or datetime.now()

    def last_activity_at(self) -> datetime | None:
        candidates = [ts for ts in (self._last_commit_at, self._last_heartbeat_at) if ts]
        return max(candidates) if candidates else None

    def is_stagnant(self, now: datetime | None = None) -> bool:
        last = self.last_activity_at()
        if last is None:
            return True
        now = now or datetime.now()
        return (now - last).total_seconds() >= self.stagnation_hours * 3600

    def emit_event(self, reason: str, now: datetime | None = None) -> Dict[str, Any]:
        now = now or datetime.now()
        event = {
            "event": "liveness_stagnation_detected",
            "reason": reason,
            "timestamp": now.isoformat(),
            "last_commit_at": self._last_commit_at.isoformat() if self._last_commit_at else None,
            "last_heartbeat_at": (
                self._last_heartbeat_at.isoformat() if self._last_heartbeat_at else None
            ),
            "stagnation_hours": self.stagnation_hours,
        }
        self._write_report(event, now)
        return event

    def _write_report(self, event: Dict[str, Any], now: datetime) -> Path | None:
        try:
            self.report_dir.mkdir(parents=True, exist_ok=True)
            report_path = self.report_dir / f"liveness_{now.strftime('%Y%m%d_%H%M%S')}.md"
            lines = [
                "# Liveness Watchdog Event",
                "",
                f"- Timestamp: {event['timestamp']}",
                f"- Reason: {event['reason']}",
                f"- Stagnation threshold (hours): {event['stagnation_hours']}",
                f"- Last commit at: {event['last_commit_at']}",
                f"- Last heartbeat at: {event['last_heartbeat_at']}",
                "",
                "## Recovery Attempts",
                "",
            ]
            if not self._recovery_attempts:
                lines.append("- (none yet)")
            for attempt in self._recovery_attempts:
                lines.append(
                    f"- {attempt.get('timestamp')}: {attempt.get('action')} "
                    f"-> {attempt.get('status')}"
                )
            report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return report_path
        except Exception as exc:  # pragma: no cover - best-effort telemetry
            logger.debug(f"Failed to write liveness report: {exc}")
            return None

    def record_recovery(self, action: str, status: str) -> None:
        self._recovery_attempts.append(
            {
                "timestamp": datetime.now().isoformat(),
                "action": action,
                "status": status,
            }
        )


class SelfRecoveryPlanner:
    """Autonomous self-recovery planner.

    When the liveness watchdog detects stagnation, this planner restarts the
    research planner along an alternative execution path and records the
    recovery attempt for transparent telemetry.
    """

    def __init__(self, watchdog: LivenessWatchdog | None = None) -> None:
        self.watchdog = watchdog or LivenessWatchdog()

    def diagnose(self, now: datetime | None = None) -> Dict[str, Any]:
        stagnant = self.watchdog.is_stagnant(now)
        reason = (
            "no recent commits or heartbeats within stagnation window"
            if stagnant
            else "active"
        )
        diagnosis = {
            "stagnant": stagnant,
            "reason": reason,
            "last_activity_at": (
                self.watchdog.last_activity_at().isoformat()
                if self.watchdog.last_activity_at()
                else None
            ),
        }
        if stagnant:
            self.watchdog.emit_event(reason, now)
        return diagnosis

    def plan_recovery(self, diagnosis: Dict[str, Any]) -> Dict[str, Any]:
        if not diagnosis.get("stagnant"):
            return {"action": "none", "reason": diagnosis.get("reason", "active")}
        self.watchdog.record_recovery("restart_research_planner", "initiated")
        return {
            "action": "restart_research_planner",
            "alternative_path": "research_planner",
            "reason": diagnosis.get("reason", "stagnation"),
        }


class AutonomousOrchestrator:
    """Modularized LangGraph Orchestrator delegating to specialized nodes."""

    def __init__(self):
        """초기화 및 의존성 주입."""
        self.llm_config = get_llm_config()
        self.liveness_watchdog = LivenessWatchdog()
        self.self_recovery_planner = SelfRecoveryPlanner(self.liveness_watchdog)
        self.agent_config = get_agent_config()
        self.research_config = get_research_config()
        self.mcp_config = get_mcp_config()

        # 스트리밍 매니저
        from src.core.streaming_manager import get_streaming_manager

        self.streaming_manager = get_streaming_manager()

        # 의존 시스템
        from src.agents.creativity_agent import CreativityAgent
        from src.core.adaptive_research_depth import AdaptiveResearchDepth
        from src.core.context_loader import ContextLoader
        from src.core.recursive_context_manager import get_recursive_context_manager
        from src.storage.hybrid_storage import HybridStorage

        self.hybrid_storage = HybridStorage()
        self.creativity_agent = CreativityAgent()
        self.context_loader = ContextLoader()
        self.context_manager = get_recursive_context_manager()

        # Research Depth
        depth_config = (
            self.research_config.research_depth
            if hasattr(self.research_config, "research_depth")
            else {}
        )
        self.research_depth = AdaptiveResearchDepth(depth_config)

        # Graph assembly
        self.graph = create_orchestrator_graph(
            creativity_agent=self.creativity_agent,
            context_manager=self.context_manager,
            streaming_manager=self.streaming_manager,
            hybrid_storage=self.hybrid_storage,
            context_loader=self.context_loader,
            research_depth=self.research_depth,
            llm_config=self.llm_config,
            agent_config=self.agent_config,
        )
        self.graph.recursion_limit = 100

    async def aclose(self) -> None:
        """Close the underlying SQLite checkpointer connection."""
        checkpointer = getattr(self.graph, "checkpointer", None)
        if checkpointer is not None and hasattr(checkpointer, "conn"):
            await checkpointer.conn.close()

    def _is_interactive_tty(self) -> bool:
        """Return True when stdout is a TTY (so live stage progress is useful)."""
        try:
            return bool(sys.stdout.isatty())
        except Exception:
            return False

    async def _stream_graph(self, initial_state, config):
        """Stream the LangGraph run, printing each node name as it completes.

        Uses ``stream_mode="updates"`` to yield ``{node_name: delta}`` per
        completed node, then ``stream_mode="values"`` to collect the final
        accumulated state (equivalent to ``ainvoke``'s return value).
        """
        final_state: Dict[str, Any] = {}
        show_progress = self._is_interactive_tty()
        async for chunk in self.graph.astream(initial_state, config, stream_mode="updates"):
            if show_progress and isinstance(chunk, dict):
                from rich import get_console

                console = get_console()
                for node_name in chunk:
                    console.print(f"[dim]▶ {node_name}[/dim]")
        async for chunk in self.graph.astream(initial_state, config, stream_mode="values"):
            if isinstance(chunk, dict):
                final_state = chunk
        return final_state

    async def execute(
        self, request: str, context: Dict[str, Any] = None, objective_id: str = None
    ) -> Dict[str, Any]:
        """연구 실행 워크플로우 기동.

        Args:
            objective_id: 기존 실행을 재개하려면 이전에 사용된 objective_id를 전달.
                생략하면 새 objective_id를 생성해 새 실행을 시작.
        """
        resuming = objective_id is not None
        objective_id = objective_id or f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        config = {"configurable": {"thread_id": objective_id}}

        # Setup Supabase Real-time Logging Handler
        # Isolated session logging is handled by the SupabaseRealtimeHandler instance per objective_id.
        supabase_handler = None
        root_logger = logging.getLogger()
        try:
            from src.utils.supabase_realtime_logger import SupabaseRealtimeHandler
            supabase_handler = SupabaseRealtimeHandler(objective_id)
            supabase_handler.setLevel(logging.INFO)
            root_logger.addHandler(supabase_handler)
            logger.debug(f"Registered SupabaseRealtimeHandler for session '{objective_id}'")
        except Exception as handler_err:
            logger.debug(f"Failed to register SupabaseRealtimeHandler: {handler_err}")

        try:
            from src.utils.supabase_realtime_logger import redirect_stdout_to_supabase
            if resuming:
                checkpoint = await self.graph.aget_state(config)
                if checkpoint and checkpoint.values:
                    logger.info(f"↩️  Resuming orchestrator run '{objective_id}' from checkpoint")
                    with redirect_stdout_to_supabase(objective_id):
                        final_state = await self._stream_graph(None, config)
                    return final_state
                logger.warning(
                    f"No checkpoint found for objective_id='{objective_id}', starting fresh"
                )

            logger.info(f"🚀 Starting modularized autonomous research: {request[:50]}...")
            initial_state = {
                "user_request": request,
                "context": context or {},
                "autopilot_mode": _autopilot_mode_enabled(context),
                "objective_id": objective_id,
                "iteration": 0,
                "max_iterations": 10,
                "should_continue": True,
                "current_step": "analyze_objectives",
                "innovation_stats": {},
                "messages": [],
            }
            self.liveness_watchdog.record_heartbeat()
            with redirect_stdout_to_supabase(objective_id):
                final_state = await self._stream_graph(initial_state, config)
            self.liveness_watchdog.record_commit()
            return final_state
        except Exception as e:
            logger.error(f"❌ Orchestrator execution failed: {e}")
            diagnosis = self.self_recovery_planner.diagnose()
            if diagnosis.get("stagnant"):
                recovery = self.self_recovery_planner.plan_recovery(diagnosis)
                logger.warning(
                    "Liveness watchdog triggered self-recovery: %s", recovery.get("action")
                )
            return {"error": str(e), "success": False, "recovery": recovery if 'recovery' in locals() else None}
        finally:
            if supabase_handler:
                try:
                    root_logger.removeHandler(supabase_handler)
                    from src.utils.supabase_realtime_logger import stop_supabase_logger_worker
                    stop_supabase_logger_worker()
                    logger.debug(f"Unregistered SupabaseRealtimeHandler for session '{objective_id}'")
                except Exception as cleanup_err:
                    logger.debug(f"Failed to clean up Supabase logging: {cleanup_err}")

    async def run_research(
        self,
        user_request: str,
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        if final_state is None:
            raise RuntimeError("run_research returned None from execute(); expected a state dictionary")
        """Legacy alias for execute()."""
        return await self.execute(user_request, context)

    def ensure_legacy_langgraph_workflow(self) -> None:
        """Backward compatibility helper.

        The current orchestrator builds its graph during initialization, so this
        method intentionally has no side effects.
        """
