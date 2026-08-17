import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any

from src.cli.cli_result import cli_result_succeeded, extract_cli_result_content
from src.core.autonomous_research_system import (
    _load_autonomous_orchestrator,
    project_root,
)

logger = logging.getLogger(__name__)

def _ensure_database_driver_for_cli() -> None:
    from src.core.db.database_driver import get_database_driver, set_database_driver
    from src.core.db.sqlite_driver import SQLiteDriver

    if get_database_driver() is None:
        sqlite_db_path = project_root / "data" / "sparkleforge.db"
        set_database_driver(SQLiteDriver(str(sqlite_db_path)))
        logger.info("✅ SQLite database driver initialized: %s", sqlite_db_path)

async def _resolve_run_session(args) -> tuple[str, str | None]:
    resume_id = getattr(args, "session_id", None)
    should_continue = getattr(args, "continue_session", False)
    from src.core.session_control import get_session_control

    if not resume_id and not should_continue:
        new_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            get_session_control().register_active_session(new_session_id, getattr(args, "query", ""))
        except RuntimeError as e:
            return "", f"❌ {e}"
        return new_session_id, None

    session_control = get_session_control()
    target_id = resume_id
    if not target_id:
        recent = await session_control.search_sessions(limit=1)
        if not recent:
            return "", "❌ No previous session found to continue (--continue)."
        target_id = recent[0].session_id

    restored_state = await session_control.restore_session(target_id)
    if restored_state is None:
        return "", f"❌ Session not found: {target_id}"

    prior_query = restored_state.get("user_query")
    if prior_query:
        args.query = (
            f"[Continuing session {target_id}]\nPrevious request: {prior_query}\n\n"
            f"New request: {args.query}"
        )
    logger.info(f"↩️  Resumed session: {target_id}")
    return target_id, None

def _persist_run_session(session_id: str, query: str, output_text: str | None) -> None:
    try:
        from src.core.session_manager import get_session_manager
        get_session_manager().save_session(
            session_id,
            agent_state={
                "user_query": query,
                "last_output_preview": (output_text or "")[:2000],
            },
            metadata={"tags": ["cli-run"]},
        )
    except Exception:
        logger.debug("Session persistence skipped for %s", session_id, exc_info=True)

async def run_command(args, config):
    from src.cli.commands.work import work_command_from_query
    if getattr(args, "mode", "research") == "work":
        return await work_command_from_query(args)

    def _apply_runtime_overrides() -> None:
        model_override = getattr(args, "model", None)
        if model_override:
            os.environ["OPEN_CODE_MODEL_PATH"] = model_override
            if config.llm.provider == "opencode":
                config.llm.open_code_model_path = model_override
            else:
                os.environ["LLM_MODEL"] = model_override
                for key in ("PLANNING_MODEL", "REASONING_MODEL", "VERIFICATION_MODEL", "GENERATION_MODEL", "COMPRESSION_MODEL"):
                    os.environ[key] = model_override
                config.llm.primary_model = model_override
                config.llm.planning_model = model_override
                config.llm.reasoning_model = model_override
                config.llm.verification_model = model_override
                config.llm.generation_model = model_override
                config.llm.compression_model = model_override

        max_tokens = getattr(args, "max_tokens", None)
        if max_tokens is not None:
            os.environ["LLM_MAX_TOKENS"] = str(max_tokens)
            config.llm.max_tokens = max_tokens

    original_query = getattr(args, "query", "")
    task_label = getattr(args, "task", None)
    if task_label:
        args.query = f"[{task_label}] {original_query}"

    session_id, session_error = await _resolve_run_session(args)
    if session_error:
        logger.error(session_error)
        return 1

    from src.core.observe.system_collector import check_disk_space_safety, check_network_connectivity
    from src.core.session_control import get_session_control

    try:
        disk_ok, disk_message = check_disk_space_safety()
        if not disk_ok:
            logger.error(disk_message)
            return 1

        network_ok, network_message = check_network_connectivity()
        if not network_ok:
            logger.warning(network_message)

        _apply_runtime_overrides()
        logger.info(f"🔬 Starting research: {args.query}")
        _ensure_database_driver_for_cli()
        orchestrator = _load_autonomous_orchestrator()()

        should_interact = (
            hasattr(sys, "stdin") and sys.stdin is not None and sys.stdin.isatty()
            and os.getenv("SPARKLEFORGE_CLI_INTERACTIVE", "true").lower() == "true"
        )

        def _needs_clarification(content: str) -> bool:
            if not content: return False
            keywords = ("추가로 사용자가 제공해야 할 단서", "후보 확정을 위한 추가 정보 요청", "추가 질문", "추가 단서", "식별을 위해")
            return any(k in content for k in keywords) and ("확정" in content or "단정" in content or "미확정" in content)

        def _extract_clarification_block(content: str) -> str:
            markers = ("추가로 사용자가 제공해야 할 단서", "후보 확정을 위한 추가 정보 요청")
            for m in markers:
                idx = content.find(m)
                if idx != -1:
                    end = content.find("\n---", idx)
                    if end == -1: end = min(len(content), idx + 1400)
                    return content[idx : end].strip()
            return ""

        base_query = args.query
        user_addendum = ""
        max_rounds = 3 if should_interact else 1
        result: dict[str, Any] | None = None

        for round_idx in range(max_rounds):
            active_query = base_query if not user_addendum else f"{base_query}\n\n{user_addendum}"
            result = await orchestrator.run_research(user_request=active_query, context={})
            content = extract_cli_result_content(result)
            if should_interact and _needs_clarification(content) and round_idx < max_rounds - 1:
                clarification_block = _extract_clarification_block(content)
                if clarification_block:
                    print("\n[추가 단서 요청 감지]\n")
                    print(clarification_block)
                try:
                    user_input = input("\n위 요청에 맞는 추가 단서(자유 입력)를 넣고 Enter를 누르면 SparkleForge가 이어서 재시도합니다. (원하면 빈 입력으로 건너뜁니다): ").strip()
                except EOFError:
                    user_input = ""
                if not user_input: break
                user_addendum = f"[사용자 추가 단서]\n{user_input}"
                continue
            break

        output_path = args.output
        if not output_path:
            output_dir = project_root / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = ".json" if args.format == "json" else ".md"
            output_path = str(output_dir / f"query_{ts}{ext}")

        text = extract_cli_result_content(result)
        succeeded = cli_result_succeeded(result, text)
        _persist_run_session(session_id, base_query, text)

        with open(output_path, "w", encoding="utf-8") as f:
            if args.format == "json":
                json.dump(result, f, ensure_ascii=False, indent=2)
            else:
                f.write(text)
        logger.info(f"✅ Results saved to {output_path}")
        if text: print(text)
        else: print(result)

        if not succeeded:
            logger.error("❌ Research completed with failure state")
            return 1
    except Exception as e:
        logger.error(f"❌ Research failed: {e}")
        return 1
    finally:
        get_session_control().release_active_session(session_id)
    return 0
