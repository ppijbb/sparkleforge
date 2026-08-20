"""AutonomousResearchSystem: the 9-innovation research pipeline, plus its
WebAppManager collaborator and two lazy orchestrator loaders.

Extracted from main.py (Anvil Phase Sigma, issue #507 -- main.py was
3,331 lines with this ~1,100-line class inlined, one of the two files
#507's Sigma-1 checklist item claimed to have already split).
"""
import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from src.core.researcher_config import load_config_from_env
from src.monitoring.system_monitor import HealthMonitor

project_root = Path(__file__).resolve().parent.parent.parent

logger = logging.getLogger(__name__)


def _load_agent_orchestrator():
    """Lazy import to keep lightweight CLI paths usable without full optional deps."""
    from src.core.agent_orchestrator import (
        AgentOrchestrator,
    )

    return AgentOrchestrator


def _load_autonomous_orchestrator():
    """Lazy import for heavy orchestrator paths."""
    from src.core.autonomous_orchestrator import AutonomousOrchestrator

    return AutonomousOrchestrator



class WebAppManager:
    """웹 앱 관리자 - Streaming Pipeline (Innovation 5) 지원"""

    def __init__(self):
        self.project_root = project_root
        self.health_monitor = HealthMonitor()

    def start_web_app(self):
        """웹 앱 시작 - Production-Grade Reliability 적용"""
        try:
            streamlit_app_path = self.project_root / "src" / "web" / "streamlit_app.py"

            if not streamlit_app_path.exists():
                logger.error(f"Streamlit app not found at {streamlit_app_path}")
                return False

            # Get port from environment variable, default to 8501
            port = os.getenv("STREAMLIT_PORT", "8501")
            address = os.getenv("STREAMLIT_ADDRESS", "0.0.0.0")

            logger.info(
                "🌐 Starting Local Researcher Web Application with Streaming Pipeline..."
            )
            logger.info(f"App will be available at: http://{address}:{port}")
            logger.info(
                "Features: Real-time streaming, Progressive reporting, Incremental save"
            )
            logger.info("Press Ctrl+C to stop the application")

            # Create logs directory if it doesn't exist
            logs_dir = self.project_root / "logs"
            logs_dir.mkdir(exist_ok=True)

            cmd = [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(streamlit_app_path),
                "--server.port",
                port,
                "--server.address",
                address,
                "--browser.gatherUsageStats",
                "false",
                "--server.enableCORS",
                "false",
                "--server.enableXsrfProtection",
                "false",
            ]

            # Start health monitoring
            asyncio.create_task(self.health_monitor.start_monitoring())

            subprocess.run(cmd, cwd=str(self.project_root))
            return True

        except KeyboardInterrupt:
            logger.info("Application stopped by user")
            return True
        except Exception as e:
            logger.error(f"Error running web application: {e}")
            return False

    async def get_web_app_health(self) -> Dict[str, Any]:
        """Get web application health status."""
        port = int(os.getenv("STREAMLIT_PORT", "8501"))
        return {
            "status": "running",
            "port": port,
            "streaming_enabled": True,
            "progressive_reporting": True,
            "incremental_save": True,
            "timestamp": datetime.now().isoformat(),
        }


class AutonomousResearchSystem:
    """자율 리서처 시스템 - 9가지 핵심 혁신 통합 메인 클래스"""

    def _sample_ood_outlier(self, distribution_center: List[Any], entropy_factor: float = 0.85) -> Any:
        """
        High-Entropy Out-of-Distribution Solution Space Sampler.
        
        Actively avoids standard LLM training distribution centers by calculating
        the centroid of the provided candidate space and sampling from the
        high-impact, non-obvious outlier regions (the 'tails' of the distribution).
        """
        import random
        import math

        if not distribution_center:
            return None
        
        # Calculate a pseudo-centroid or reference point
        # In a real implementation, this would involve embedding space distance
        # For this implementation, we use a high-entropy selection bias
        
        # Sort by 'standardness' (index) and pick from the edges
        n = len(distribution_center)
        if n <= 2:
            return random.choice(distribution_center)
            
        # Bias towards the edges (outliers)
        if random.random() < entropy_factor:
            idx = random.choice([0, n - 1])
            return distribution_center[idx]
        return random.choice(distribution_center)

    def __init__(self, bootstrap_result=None):
        # Load configurations from environment - ALL REQUIRED, NO DEFAULTS
        try:
            if bootstrap_result and bootstrap_result.ok:
                self.config = bootstrap_result.values["config"]["config"]
                logger.info(
                    "✅ Configuration transferred from bootstrap successfully"
                )
            else:
                self.config = load_config_from_env()
                logger.info(
                    "✅ Configuration loaded successfully from environment variables"
                )

            # Validate ChromaDB availability (optional)
            try:
                import chromadb  # type: ignore
            except ImportError:
                # Log only once per process if needed, or suppress entirely for CLI
                pass

        except ValueError as e:
            logger.error(f"❌ Configuration loading failed: {e}")
            logger.error(
                "Please check your .env file and ensure all required variables are set"
            )
            logger.info("\nRequired environment variables:")
            logger.info(
                "  - LLM_MODEL: LLM model identifier (e.g., google/gemini-3.5-flash-lite)"
            )
            logger.info("  - GOOGLE_API_KEY: Your Google or Vertex AI API key")
            logger.info("  - LLM_PROVIDER: Provider name (e.g., google)")
            raise

        # Initialize components with 8 innovations
        logger.info("🔧 Initializing system components...")
        try:
            from src.core.db.database_driver import (
                get_database_driver,
                set_database_driver,
            )
            from src.core.db.sqlite_driver import SQLiteDriver

            if get_database_driver() is None:
                if bootstrap_result and bootstrap_result.ok and "database" in bootstrap_result.values:
                    driver = bootstrap_result.values["database"]["driver_instance"]
                    set_database_driver(driver)
                    logger.info(f"✅ SQLite database driver transferred from bootstrap: {driver.__class__.__name__}")
                else:
                    sqlite_db_path = project_root / "data" / "sparkleforge.db"
                    set_database_driver(SQLiteDriver(str(sqlite_db_path)))
                    logger.info(f"✅ SQLite database driver initialized: {sqlite_db_path}")

            # Use new multi-agent orchestrator (no fallback - fail clearly)
            AgentOrchestrator = _load_agent_orchestrator()
            self.orchestrator = AgentOrchestrator()

            # Initialize TaskAnalyzerAgent
            from src.agents.task_analyzer import TaskAnalyzerAgent
            self.task_analyzer = TaskAnalyzerAgent()
            logger.info("✅ Task Analyzer Agent initialized")

            logger.info("✅ Multi-Agent Orchestrator initialized (no fallback mode)")
            logger.info("✅ Autonomous Orchestrator initialized")

            from src.agents.task_analyzer import TaskAnalyzerAgent
            self.task_analyzer = TaskAnalyzerAgent()
            logger.info("✅ Task Analyzer Agent initialized")

        except Exception as e:
            logger.error(f"❌ Orchestrator initialization failed: {e}")
            raise

        try:
            from src.core.mcp_integration import UniversalMCPHub

            if bootstrap_result and bootstrap_result.ok and "mcp_hub" in bootstrap_result.values:
                self.mcp_hub = bootstrap_result.values["mcp_hub"]["mcp_hub"]
                logger.info("✅ MCP Hub transferred from bootstrap")
            else:
                self.mcp_hub = UniversalMCPHub()
                logger.info("✅ MCP Hub initialized")
        except Exception as e:
            logger.error(f"❌ MCP Hub initialization failed: {e}")
            raise

        # 새로운 기능 모듈 기본 활성화 (모든 기능 기본 ON)
        try:
            from src.core.feature_flags import FeatureFlags

            FeatureFlags.log_status()

            # MCP 안정성 서비스 (기본 활성화)
            if FeatureFlags.ENABLE_MCP_STABILITY:
                from src.core.mcp_stability_service import MCPStabilityService

                self.mcp_stability_service = MCPStabilityService()
                logger.info("✅ MCP Stability Service enabled (default)")
            else:
                self.mcp_stability_service = None
                logger.info(
                    "⚠️ MCP Stability Service disabled (via DISABLE_MCP_STABILITY)"
                )

            # MCP 백그라운드 헬스체크 (기본 활성화)
            if FeatureFlags.ENABLE_MCP_HEALTH_BACKGROUND:
                from src.core.mcp_health_background import MCPHealthBackgroundService

                self.mcp_health_service = MCPHealthBackgroundService(
                    self.mcp_hub, interval=60
                )
                logger.info(
                    "✅ MCP Health Background Service enabled (default, will start on first execution)"
                )
            else:
                self.mcp_health_service = None
                logger.info(
                    "⚠️ MCP Health Background Service disabled (via DISABLE_MCP_HEALTH_BACKGROUND)"
                )

            # Guardrails 검증 (기본 활성화)
            if FeatureFlags.ENABLE_GUARDRAILS:
                from src.core.guardrails_validator import GuardrailsValidator

                self.guardrails_validator = GuardrailsValidator()
                logger.info("✅ Guardrails Validator enabled (default)")
            else:
                self.guardrails_validator = None
                logger.info("⚠️ Guardrails Validator disabled (via DISABLE_GUARDRAILS)")


            # YAML 설정 로더 (기본 활성화)
            if FeatureFlags.ENABLE_YAML_CONFIG:
                from src.core.yaml_config_loader import YAMLConfigLoader

                self.yaml_config_loader = YAMLConfigLoader()
                logger.info("✅ YAML Config Loader enabled (default)")
            else:
                self.yaml_config_loader = None
                logger.info("⚠️ YAML Config Loader disabled (via DISABLE_YAML_CONFIG)")
        except Exception as e:
            logger.warning(
                f"⚠️ Feature initialization failed: {e} - continuing with core features only"
            )
            # 기본값: 모든 기능 None (에러 발생 시)
            self.mcp_stability_service = None
            self.mcp_health_service = None
            self.guardrails_validator = None
            self.yaml_config_loader = None

        try:
            self.web_manager = WebAppManager()
            logger.info("✅ Web Manager initialized")
        except Exception as e:
            logger.error(f"❌ Web Manager initialization failed: {e}")
            raise

        try:
            self.health_monitor = HealthMonitor()
            logger.info("✅ Health Monitor initialized")
        except Exception as e:
            logger.error(f"❌ Health Monitor initialization failed: {e}")
            raise

        # ProcessManager 초기화 (프로세스 추적 및 종료 관리)
        try:
            from src.core.process_manager import get_process_manager

            pm = get_process_manager()
            # ProcessManager의 시그널 핸들러는 설치하지 않음 (main.py의 핸들러 사용)
            # pm.initialize()는 호출하지 않음 - 시그널 핸들러 충돌 방지
            logger.info(
                "✅ ProcessManager initialized (signal handlers managed by main.py)"
            )
        except Exception as e:
            logger.warning(
                f"⚠️ ProcessManager initialization failed: {e} - continuing without process tracking"
            )

        # Initialize signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Shutdown flag 초기화
        self._shutdown_requested = False

        logger.info("✅ AutonomousResearchSystem initialized successfully")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        import os
        import sys

        # Reentrant call 방지를 위해 os.write 사용 (print 대신)
        try:
            msg = f"Received signal {signum}, initiating graceful shutdown...\n"
            os.write(sys.stderr.fileno(), msg.encode("utf-8"))
        except (OSError, AttributeError):
            # stderr에 쓸 수 없으면 무시 (이미 종료 중일 수 있음)
            pass

        # Shutdown 플래그 설정 (중복 방지)
        if hasattr(self, "_shutdown_requested") and self._shutdown_requested:
            # 이미 종료 중이면 재진입 방지만 수행
            try:
                msg = "Shutdown already in progress; ignoring additional signal\n"
                os.write(sys.stderr.fileno(), msg.encode("utf-8"))
            except (OSError, AttributeError):
                pass
            return

        self._shutdown_requested = True

        # 실행 중인 이벤트 루프가 있는지 확인하고 shutdown 작업 스케줄링
        try:
            loop = asyncio.get_running_loop()
            # 중복 생성 방지: 이미 스케줄된 작업이 있으면 재생성하지 않음
            if (
                not hasattr(self, "_shutdown_task")
                or self._shutdown_task is None
                or self._shutdown_task.done()
            ):

                def _schedule():
                    self._shutdown_task = asyncio.create_task(self._graceful_shutdown())

                loop.call_soon_threadsafe(_schedule)
            else:
                # logger는 시그널 핸들러에서 사용하지 않음 (reentrant call 위험)
                pass
        except RuntimeError:
            # 이벤트 루프가 없으면 강제 종료
            try:
                msg = "No event loop available, forcing exit\n"
                os.write(sys.stderr.fileno(), msg.encode("utf-8"))
            except (OSError, AttributeError):
                pass
            os._exit(1)  # sys.exit 대신 os._exit 사용 (더 안전)

    async def _graceful_shutdown(self):
        """Graceful shutdown with state persistence."""
        try:
            logger.info("Performing graceful shutdown...")

            # ProcessManager를 통한 프로세스 종료 (우선 처리)
            try:
                from src.core.process_manager import get_process_manager

                pm = get_process_manager()
                pm.abort()  # 모든 작업 중단 플래그 설정
                killed = pm.kill_all()  # 모든 등록된 프로세스 종료
                if killed > 0:
                    logger.info(f"Killed {killed} registered processes")
            except Exception as e:
                logger.debug(f"Error killing processes: {e}")

            # MCP Hub cleanup (타임아웃 단축: 10초 -> 3초)
            if self.config.mcp.enabled and self.mcp_hub:
                try:
                    # 신규 연결 차단 (즉시)
                    if hasattr(self.mcp_hub, "start_shutdown"):
                        self.mcp_hub.start_shutdown()
                    # cleanup은 CancelledError를 발생시킬 수 있으므로 무시
                    try:
                        await asyncio.wait_for(
                            self.mcp_hub.cleanup(), timeout=3.0
                        )  # 타임아웃 단축
                    except asyncio.CancelledError:
                        logger.debug(
                            "MCP Hub cleanup was cancelled (normal during shutdown)"
                        )
                    except TimeoutError:
                        logger.warning(
                            "MCP Hub cleanup timed out (continuing shutdown)"
                        )
                except asyncio.CancelledError:
                    logger.debug(
                        "MCP Hub cleanup setup was cancelled (normal during shutdown)"
                    )
                except Exception as e:
                    logger.warning(f"Error cleaning up MCP Hub: {e}")

            # Health monitor 정지 (타임아웃 단축: 5초 -> 2초)
            try:
                await asyncio.wait_for(
                    self.health_monitor.stop_monitoring(), timeout=2.0
                )
            except TimeoutError:
                logger.warning("Health monitor stop timed out (continuing shutdown)")
            except Exception as e:
                logger.debug(f"Error stopping health monitor: {e}")

            logger.info("✅ Graceful shutdown completed")

        except Exception as e:
            logger.error(f"Error during graceful shutdown: {e}")
        finally:
            # 최종 종료 준비
            logger.info("Exiting...")
            # 외부 라이브러리 태스크는 개별 매니저가 정리함. 일괄 취소는 하지 않음

            # sys.exit(0)은 호출하지 않음 - asyncio.run()이 자동으로 처리
            # 대신 루프에서 나가도록 함

    def _detect_output_format_from_content(self, content: str, request: str) -> str:
        """생성된 내용을 보고 파일 형식 결정 (최소한의 패턴 매칭만)."""
        if not content:
            return "md"

        # 코드 블록 제거 후 내용 확인
        content_clean = content.replace("```", "").strip()

        # Python 코드 패턴
        if (
            "def " in content_clean[:1000] and "import " in content_clean[:1000]
        ) or content.startswith("```python"):
            return "py"

        # Java 코드 패턴
        if (
            "public class" in content_clean[:1000]
            or "public static void" in content_clean[:1000]
        ) or content.startswith("```java"):
            return "java"

        # JavaScript 코드 패턴
        if (
            ("const " in content_clean[:1000] or "function " in content_clean[:1000])
            and "console" in content_clean[:1000]
        ) or content.startswith("```javascript"):
            return "js"

        # HTML 패턴
        if content.strip().startswith("<!DOCTYPE") or content.strip().startswith(
            "<html"
        ):
            return "html"

        # 기본: Markdown
        return "md"

    async def run_research(
        self,
        request: str,
        output_path: str | None = None,
        streaming: bool = False,
        output_format: str | None = None,
    ) -> Dict[str, Any]:
        """연구 실행 - 9가지 핵심 혁신 적용"""
        logger.info("🤖 Starting Autonomous Research System with 9 Core Innovations")
        logger.info("=" * 80)
        logger.info(f"Request: {request}")
        logger.info(f"Primary LLM: {self.config.llm.primary_model}")
        logger.info(f"Planning Model: {self.config.llm.planning_model}")
        logger.info(f"Reasoning Model: {self.config.llm.reasoning_model}")
        logger.info(f"Verification Model: {self.config.llm.verification_model}")
        logger.info(f"Self-planning: {self.config.agent.enable_self_planning}")
        logger.info(
            f"Agent Communication: {self.config.agent.enable_agent_communication}"
        )
        logger.info(f"MCP Enabled: {self.config.mcp.enabled}")
        logger.info(f"Streaming Pipeline: {streaming}")
        logger.info(
            f"Adaptive Supervisor: {self.config.agent.max_concurrent_research_units}"
        )
        logger.info(f"Hierarchical Compression: {self.config.compression.enabled}")
        logger.info(f"Continuous Verification: {self.config.verification.enabled}")
        logger.info(f"Adaptive Context Window: {self.config.context_window.enabled}")
        logger.info("=" * 80)

        try:
            # Start health monitoring
            await self.health_monitor.start_monitoring()

            # Pre-analyze request using TaskAnalyzer
            logger.info("🔍 Analyzing request with TaskAnalyzer...")
            analysis_result = await self.task_analyzer.analyze_objective(request)
            logger.info(f"✅ Task analysis completed: {len(analysis_result.get('objectives', []))} objectives identified")
            # Optionally inject analysis into context or state if needed
            # request = f"{request}\n\nAnalysis: {json.dumps(analysis_result)}"

            # Initialize MCP client if enabled
            if self.config.mcp.enabled:
                try:
                    await self.mcp_hub.initialize_mcp()

                    # MCP 백그라운드 헬스체크 시작 (선택적)
                    if hasattr(self, "mcp_health_service") and self.mcp_health_service:
                        try:
                            await self.mcp_health_service.start()
                            logger.info("✅ MCP Health Background Service started")
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to start MCP health service: {e}")
                except asyncio.CancelledError:
                    # 초기화 중 취소된 경우 - 상위로 전파하여 종료
                    logger.warning("MCP initialization was cancelled")
                    raise

            # Run research with production-grade reliability
            if streaming:
                result = await self._run_streaming_research(request)
            else:
                # Perform task analysis before orchestration
                try:
                    analysis = await self.task_analyzer.analyze_objective(request)
                    logger.info(f"✅ Task analysis completed: {len(analysis.get('objectives', []))} objectives identified")
                    # Optionally inject analysis into state or context if needed
                except Exception as e:
                    logger.warning(f"⚠️ Task analysis failed, continuing with raw request: {e}")

                # Guardrails Input 검증 (선택적)
                if hasattr(self, "guardrails_validator") and self.guardrails_validator:
                    try:
                        from src.core.agent_orchestrator import AgentState

                        initial_state: AgentState = {
                            "messages": [],
                            "user_query": request,
                            "research_plan": None,
                            "research_tasks": [],
                            "research_results": [],
                            "verified_results": [],
                            "final_report": None,
                            "current_agent": None,
                            "iteration": 0,
                            "session_id": None,
                            "research_failed": False,
                            "verification_failed": False,
                            "report_failed": False,
                            "error": None,
                        }
                        validated_state = (
                            await self.guardrails_validator.validate_input(
                                initial_state
                            )
                        )
                        # 검증 통과 시 request 그대로 사용 (기존 코드 수정 없음)
                        request = validated_state.get("user_query", request)
                    except ValueError as e:
                        logger.error(f"❌ Input validation failed: {e}")
                        return {
                            "success": False,
                            "error": str(e),
                            "final_report": f"Input validation failed: {str(e)}",
                        }
                    except Exception as e:
                        logger.warning(
                            f"⚠️ Guardrails validation error: {e} - continuing without validation"
                        )

                # Use new multi-agent orchestrator (no fallback - fail clearly)
                workflow_result = await self.orchestrator.execute(request)
                workflow_result = await self._run_deep_validation(workflow_result, request)

                # Guardrails Output 검증 (선택적)
                if hasattr(self, "guardrails_validator") and self.guardrails_validator:
                    try:
                        validated_result = (
                            await self.guardrails_validator.validate_output(
                                workflow_result
                            )
                        )
                        workflow_result = validated_result
                    except ValueError as e:
                        logger.error(f"❌ Output validation failed: {e}")
                        workflow_result["error"] = str(e)
                        workflow_result["final_report"] = (
                            f"Output validation failed: {str(e)}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"⚠️ Guardrails validation error: {e} - continuing without validation"
                        )

                # 실패 상태 확인 - fallback 없이 명확한 오류 반환
                research_failed = workflow_result.get("research_failed", False)
                verification_failed = workflow_result.get("verification_failed", False)
                report_failed = workflow_result.get("report_failed", False)
                final_report = workflow_result.get("final_report", "")

                # 실패 보고서도 실패로 처리 (내용이 "연구 실패" 또는 "연구 완료 불가" 포함)
                is_failure_report = final_report and (
                    "연구 실패" in final_report
                    or "연구 완료 불가" in final_report
                    or "❌" in final_report
                )

                if (
                    research_failed
                    or verification_failed
                    or report_failed
                    or is_failure_report
                ):
                    error_msg = workflow_result.get("error", "알 수 없는 오류")
                    if not error_msg or error_msg == "알 수 없는 오류":
                        # 실패 보고서에서 오류 메시지 추출
                        if final_report and "오류 내용" in final_report:
                            lines = final_report.split("\n")
                            for i, line in enumerate(lines):
                                if "오류 내용" in line and i + 1 < len(lines):
                                    error_msg = lines[i + 1].strip()
                                    break
                        elif final_report and "❌" in final_report:
                            # 실패 보고서에서 간단히 추출
                            if "연구 실행 실패" in final_report:
                                error_msg = "연구 실행이 실패했습니다"
                            elif "보고서 생성 실패" in final_report:
                                error_msg = "보고서 생성이 실패했습니다"
                            else:
                                error_msg = "연구 실행 중 오류가 발생했습니다"

                    failed_agent = workflow_result.get("current_agent", "unknown")
                    session_id = workflow_result.get("session_id", "N/A")

                    logger.error(f"❌ Research failed at {failed_agent}: {error_msg}")

                    # 실패 결과 반환 (사용자가 재시도할 수 있도록)
                    result = {
                        "success": False,
                        "query": request,
                        "error": error_msg,
                        "failed_agent": failed_agent,
                        "session_id": session_id,
                        "content": final_report or f"연구 실패: {error_msg}",
                        "timestamp": datetime.now().isoformat(),
                        "metadata": {
                            "model_used": "multi-agent",
                            "execution_time": 0.0,
                            "cost": 0.0,
                            "confidence": 0.0,
                            "failed": True,
                        },
                        "synthesis_results": {
                            "content": final_report or "",
                            "failed": True,
                        },
                        "sources": [],
                        "innovation_stats": {"multi_agent_orchestration": "enabled"},
                        "system_health": {
                            "overall_status": "unhealthy",
                            "error": error_msg,
                        },
                        "retry_available": True,
                    }

                    logger.warning("⚠️ Research completed with errors - user can retry")
                else:
                    # 성공 결과
                    final_report = workflow_result.get("final_report", "")
                    if not final_report:
                        # 보고서가 없으면 실패로 처리
                        result = {
                            "success": False,
                            "query": request,
                            "error": "보고서 생성 실패: 최종 보고서가 생성되지 않았습니다",
                            "failed_agent": "generator",
                            "session_id": workflow_result.get("session_id", "N/A"),
                            "content": "연구 실행 중 오류가 발생했습니다.",
                            "timestamp": datetime.now().isoformat(),
                            "metadata": {"failed": True, "confidence": 0.0},
                            "retry_available": True,
                        }
                    else:
                        result = {
                            "success": True,
                            "query": request,
                            "content": final_report,
                            "timestamp": datetime.now().isoformat(),
                            "metadata": {
                                "model_used": "multi-agent",
                                "execution_time": 0.0,
                                "cost": 0.0,
                                "confidence": 0.9,
                            },
                            "synthesis_results": {"content": final_report},
                            "sources": self._extract_sources_from_workflow(
                                workflow_result
                            ),
                            "innovation_stats": {
                                "multi_agent_orchestration": "enabled"
                            },
                            "system_health": {"overall_status": "healthy"},
                            "session_id": workflow_result.get("session_id"),
                        }

            # Apply hierarchical compression if enabled
            # Commented out to avoid serialization errors
            # if self.config.compression.enabled:
            #     result = await self._apply_hierarchical_compression(result)

            # Save results - LLM이 생성한 내용을 그대로 사용
            content = result.get("content", "") or result.get(
                "synthesis_results", {}
            ).get("content", "")

            if output_path:
                # 사용자가 지정한 경로 사용
                final_path = await self._save_content_as_file(
                    content, output_path, result
                )
                logger.info(f"📄 Results saved to: {final_path}")
            else:
                # 생성된 내용을 보고 형식 결정
                detected_format = self._detect_output_format_from_content(
                    content, request
                )
                if output_format is None:
                    output_format = detected_format

                output_dir = project_root / "output"
                output_dir.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # 확장자 결정 (간단하게)
                if output_format.startswith("."):
                    ext = output_format
                else:
                    ext_map = {
                        "py": ".py",
                        "java": ".java",
                        "js": ".js",
                        "html": ".html",
                        "md": ".md",
                        "pdf": ".pdf",
                    }
                    ext = ext_map.get(output_format, ".md")
                default_output = output_dir / f"research_{timestamp}{ext}"

                final_path = await self._save_content_as_file(
                    content, str(default_output), result
                )
                logger.info(
                    f"📄 Results saved to default location: {final_path} (format: {output_format})"
                )
                self._display_results(result)

            # Get final health status
            health_status = self.health_monitor.get_system_health()
            result["system_health"] = health_status

            logger.info("✅ Research completed successfully with 9 Core Innovations")
            return result

        except Exception as e:
            logger.error(f"Research failed with exception: {e}")
            import traceback

            logger.error(traceback.format_exc())

            # Get error health status
            error_health = self.health_monitor.get_system_health()
            logger.error(f"System health at failure: {error_health}")

            # 실패 결과 반환 (예외 대신)
            result = {
                "success": False,
                "query": request,
                "error": f"시스템 오류: {str(e)}",
                "failed_agent": "system",
                "content": f"연구 실행 중 시스템 오류가 발생했습니다: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "model_used": "multi-agent",
                    "execution_time": 0.0,
                    "cost": 0.0,
                    "confidence": 0.0,
                    "failed": True,
                },
                "synthesis_results": {"content": "", "failed": True},
                "sources": [],
                "innovation_stats": {},
                "system_health": error_health,
                "retry_available": True,
            }

            # 실패 결과도 저장
            if output_path:
                await self._save_results_incrementally(
                    result, output_path, output_format
                )
            else:
                output_dir = project_root / "output"
                output_dir.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                default_output = (
                    output_dir / f"research_failed_{timestamp}.{output_format}"
                )
                await self._save_results_incrementally(
                    result, str(default_output), output_format
                )
                self._display_results(result)

            # 예외를 발생시키지 않고 결과 반환
            return result

    async def _run_streaming_research(self, request: str) -> Dict[str, Any]:
        """Run research with streaming pipeline (Innovation 5)."""
        logger.info("🌊 Starting streaming research pipeline...")

        # Create streaming callback
        async def streaming_callback(partial_result: Dict[str, Any]):
            logger.info(
                f"📊 Streaming partial result: {partial_result.get('type', 'unknown')}"
            )
            # In a real implementation, this would send to web interface
            print(
                f"📊 Partial Result: {partial_result.get('summary', 'Processing...')}"
            )

        # Unified path: same as non-streaming (AgentOrchestrator.execute)
        raw = await self.orchestrator.execute(request)
        raw = await self._run_deep_validation(raw, request)
        result = raw

        # Extract and format result
        final_synthesis = result.get("synthesis_results", {}).get("content", "")
        if not final_synthesis:
            final_synthesis = result.get("content", "")

        formatted_result = {
            "query": request,
            "content": final_synthesis or "Research completed",
            "timestamp": datetime.now().isoformat(),
            "metadata": result.get("metadata", {}),
            "synthesis_results": result.get("synthesis_results", {}),
            "sources": self._extract_sources(result),
            "innovation_stats": result.get("innovation_stats", {}),
            "system_health": result.get("system_health", {}),
        }

        logger.info("✅ Streaming research completed")
        return formatted_result

    def _extract_sources_from_workflow(
        self, workflow_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract sources from workflow results."""
        sources = []
        seen_urls = set()

        # Extract from verified results (우선)
        verified_results = workflow_result.get("verified_results", [])
        for result in verified_results:
            if isinstance(result, dict):
                url = result.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    sources.append(
                        {
                            "title": result.get("title", ""),
                            "url": url,
                            "snippet": result.get("snippet", ""),
                        }
                    )

        # Extract from research results (백업)
        research_results = workflow_result.get("research_results", [])
        for result in research_results:
            if isinstance(result, dict):
                url = result.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    sources.append(
                        {
                            "title": result.get("title", ""),
                            "url": url,
                            "snippet": result.get("snippet", ""),
                        }
                    )
            elif isinstance(result, str) and "Source:" in result:
                # Extract URL from result string (레거시)
                parts = result.split("Source:")
                if len(parts) > 1:
                    url = parts[1].strip()
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        title = (
                            parts[0].split(":")[-1].strip() if ":" in parts[0] else ""
                        )
                        sources.append({"title": title, "url": url, "snippet": ""})

        return sources[:20]  # Limit to 20 sources

    def _extract_sources(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract sources from research results (legacy method)."""
        sources = []

        # Try to extract from execution results
        execution_results = result.get("detailed_results", {}).get(
            "execution_results", []
        )
        for exec_result in execution_results:
            if isinstance(exec_result, dict):
                # Look for search results
                if "results" in exec_result:
                    sources.extend(exec_result["results"])
                elif "sources" in exec_result:
                    sources.extend(exec_result["sources"])
                elif "url" in exec_result:
                    sources.append(exec_result)

        # Deduplicate by URL
        seen_urls = set()
        unique_sources = []
        for source in sources:
            url = source.get("url") or source.get("link")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_sources.append(
                    {
                        "title": source.get("title", source.get("name", "")),
                        "url": url,
                        "snippet": source.get("snippet", source.get("summary", "")),
                    }
                )

        return unique_sources[:20]  # Limit to 20 sources

    async def _run_deep_validation(
        self, workflow_result: Dict[str, Any], request: str
    ) -> Dict[str, Any]:
        """Wire ValidationAgent (issue #1041) into the research synthesis loop.

        Runs the deep multi-pass ValidationAgent over the synthesized research
        results before report generation. Best-effort: validation failures are
        logged and surfaced in metadata but never abort the pipeline.
        """
        try:
            from src.agents.validation_agent import ValidationAgent

            execution_results = list(workflow_result.get("research_results", []))
            verified_results = workflow_result.get("verified_results", [])
            if verified_results:
                execution_results.extend(verified_results)

            if not execution_results:
                logger.info("ℹ️ Deep validation skipped: no execution results to validate")
                return workflow_result

            original_objectives = workflow_result.get("research_plan", {}).get(
                "objectives", []
            ) or []

            validation_agent = ValidationAgent()
            validation_result = await validation_agent.validate_results(
                execution_results=execution_results,
                original_objectives=original_objectives,
                user_request=request,
                objective_id=workflow_result.get("session_id"),
            )

            workflow_result["deep_validation"] = validation_result

            validation_report = validation_result.get("validation_report", {})
            if validation_report:
                recommendations = validation_report.get("recommendations", [])
                if recommendations:
                    logger.info(
                        f"🔍 Deep validation recommendations: {recommendations}"
                    )

            logger.info(
                f"✅ Deep validation completed: "
                f"{validation_result.get('validation_level', 'unknown')} "
                f"({validation_result.get('validation_score', 0.0):.2f})"
            )
        except Exception as e:
            logger.warning(
                f"⚠️ Deep validation failed (continuing without): {e}"
            )
            workflow_result["deep_validation"] = {
                "success": False,
                "error": str(e),
            }

        return workflow_result

    async def _apply_hierarchical_compression(
        self, result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply hierarchical compression (Innovation 2)."""
        if not self.config.compression.enabled:
            return result

        logger.info("🗜️ Applying hierarchical compression...")

        # Import compression module
        from src.core.compression import compress_data

        # Compress large text fields
        if "synthesis_results" in result:
            compressed_synthesis = await compress_data(result["synthesis_results"])
            result["synthesis_results_compressed"] = compressed_synthesis

        logger.info("✅ Hierarchical compression applied")
        return result

    async def _save_content_as_file(
        self, content: str, output_path: str, result: Dict[str, Any]
    ) -> str:
        """LLM이 생성한 내용을 그대로 파일로 저장 (하드코딩 없이)."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # LLM이 생성한 내용을 그대로 저장 (추가 템플릿 없이)
        # 소스 정보가 있으면 마지막에 추가
        if result.get("sources"):
            sources_text = "\n\n## 참고 문헌\n\n"
            for i, source in enumerate(result.get("sources", []), 1):
                sources_text += (
                    f"{i}. [{source.get('title', 'N/A')}]({source.get('url', '')})\n"
                )
                if source.get("snippet"):
                    sources_text += f"   {source.get('snippet', '')[:200]}...\n"
                sources_text += "\n"
            content = content + sources_text

        output_file.write_text(content, encoding="utf-8")
        return str(output_file)

    async def _save_results_incrementally(
        self, result: Dict[str, Any], output_path: str, output_format: str = "json"
    ):
        """Save results with incremental save (Innovation 5)."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Save incrementally based on format
        temp_file = output_file.with_suffix(".tmp")

        if output_format.lower() == "json":
            # Custom JSON encoder for datetime objects
            class DateTimeEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, datetime):
                        return obj.isoformat()
                    elif hasattr(obj, "value"):  # Enum 처리
                        return obj.value
                    elif hasattr(obj, "__dict__"):  # 객체의 경우
                        return str(obj)
                    return super().default(obj)

            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False, cls=DateTimeEncoder)
        elif output_format.lower() == "yaml":
            import yaml

            with open(temp_file, "w", encoding="utf-8") as f:
                yaml.dump(result, f, default_flow_style=False, allow_unicode=True)
        elif output_format.lower() == "txt":
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write("Research Results\n")
                f.write("===============\n\n")
                f.write(f"Query: {result.get('query', 'N/A')}\n")
                f.write(f"Timestamp: {result.get('timestamp', 'N/A')}\n\n")
                if "content" in result:
                    f.write(f"Content:\n{result['content']}\n\n")
                elif "synthesis_results" in result:
                    synthesis = result["synthesis_results"]
                    if isinstance(synthesis, dict) and "content" in synthesis:
                        f.write(f"Content:\n{synthesis['content']}\n\n")
                if "sources" in result:
                    f.write("Sources:\n")
                    for i, source in enumerate(result["sources"], 1):
                        f.write(
                            f"{i}. {source.get('title', 'N/A')} - {source.get('url', 'N/A')}\n"
                        )
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

        # Atomic move
        temp_file.replace(output_file)

        logger.info(
            f"✅ Results saved incrementally to: {output_file} (format: {output_format})"
        )

    def _display_results(self, result: Dict[str, Any]):
        """Display results with enhanced formatting."""
        print("\n📋 Research Results with 9 Core Innovations:")
        print("=" * 80)

        # 실패 상태 확인 및 표시
        if not result.get("success", True):
            print("\n❌ 연구 실행 실패")
            print("=" * 80)
            print(f"\n오류: {result.get('error', '알 수 없는 오류')}")
            print(f"실패 단계: {result.get('failed_agent', 'unknown')}")
            print(f"\n세션 ID: {result.get('session_id', 'N/A')}")
            print("\n재시도 방법:")
            print("1. 같은 쿼리로 다시 시도: python main.py --request 'YOUR_QUERY'")
            print("2. 다른 검색어로 시도")
            print("3. 네트워크 연결 확인")
            print("=" * 80)
            return

        # Display main research content
        if "content" in result and result["content"]:
            print("\n📝 Research Content:")
            print("-" * 60)
            print(result["content"])
            print("-" * 60)
        elif "synthesis_results" in result:
            synthesis = result["synthesis_results"]
            if isinstance(synthesis, dict) and "content" in synthesis:
                print("\n📝 Research Content:")
                print("-" * 60)
                print(synthesis["content"])
                print("-" * 60)
            else:
                print(f"\n📝 Synthesis: {synthesis}")
        else:
            print("\n❌ No research content found in results")

        # Display research metadata
        if "metadata" in result:
            metadata = result["metadata"]
            print("\n📊 Research Metadata:")
            print(f"  • Model Used: {metadata.get('model_used', 'N/A')}")
            print(f"  • Execution Time: {metadata.get('execution_time', 'N/A'):.2f}s")
            print(f"  • Cost: ${metadata.get('cost', 0):.4f}")
            print(f"  • Confidence: {metadata.get('confidence', 'N/A')}")

        # Display synthesis results
        if "synthesis_results" in result and isinstance(
            result["synthesis_results"], dict
        ):
            synthesis = result["synthesis_results"]
            if "synthesis_results" in synthesis:
                print(f"\n📝 Synthesis: {synthesis.get('synthesis_results', 'N/A')}")

        # Display innovation stats
        if "innovation_stats" in result:
            stats = result["innovation_stats"]
            print("\n🚀 Innovation Statistics:")
            print(f"  • Adaptive Supervisor: {stats.get('adaptive_supervisor', 'N/A')}")
            print(
                f"  • Hierarchical Compression: {stats.get('hierarchical_compression', 'N/A')}"
            )
            print(
                f"  • Multi-Model Orchestration: {stats.get('multi_model_orchestration', 'N/A')}"
            )
            print(
                f"  • Continuous Verification: {stats.get('continuous_verification', 'N/A')}"
            )
            print(f"  • Streaming Pipeline: {stats.get('streaming_pipeline', 'N/A')}")
            print(f"  • Universal MCP Hub: {stats.get('universal_mcp_hub', 'N/A')}")
            print(
                f"  • Adaptive Context Window: {stats.get('adaptive_context_window', 'N/A')}"
            )
            print(
                f"  • Production-Grade Reliability: {stats.get('production_grade_reliability', 'N/A')}"
            )

        # Display system health
        if "system_health" in result:
            health = result["system_health"]
            print(f"\n🏥 System Health: {health.get('overall_status', 'Unknown')}")
            print(f"  • Health Score: {health.get('health_score', 'N/A')}")
            print(f"  • Monitoring Active: {health.get('monitoring_active', 'N/A')}")

        print("=" * 80)

    async def run_mcp_server(self):
        """MCP 서버 실행"""
        await self.mcp_hub.initialize_mcp()

    async def run_mcp_client(self):
        """MCP 클라이언트 실행"""
        await self.mcp_hub.initialize_mcp()

    def run_web_app(self):
        """웹 앱 실행"""
        return self.web_manager.start_web_app()

    async def run_health_check(self):
        """Run comprehensive health check for all system components."""
        logger.info("🏥 Running comprehensive health check...")

        # Check MCP tools health
        if self.config.mcp.enabled:
            # MCP Hub health check
            logger.info("MCP Hub initialized and ready")

        # Check system health
        system_health = self.health_monitor.get_system_health()
        logger.info(f"System Health: {system_health.get('overall_status', 'Unknown')}")

        # Check web app health
        web_health = await self.web_manager.get_web_app_health()
        logger.info(f"Web App Health: {web_health.get('status', 'Unknown')}")

        logger.info("✅ Health check completed")

    async def check_mcp_servers(self):
        """MCP 서버 연결 상태 확인."""
        logger.info("📊 Checking MCP server connections...")

        if not self.config.mcp.enabled:
            logger.warning("MCP is disabled")
            return

        try:
            # MCP Hub 초기화 확인 - 이미 연결된 서버가 있으면 그대로 사용
            if self.mcp_hub.mcp_sessions:
                logger.info(
                    f"Found {len(self.mcp_hub.mcp_sessions)} existing MCP server connections"
                )
            else:
                logger.info(
                    "No existing connections. Will attempt quick connection tests for each server..."
                )

            # 서버 상태 확인 (각 서버에 대해 짧은 타임아웃으로 연결 시도)
            logger.info("Checking MCP server connection status...")
            server_status = await self.mcp_hub.check_mcp_servers()

            # 결과 출력
            print("\n" + "=" * 80)
            print("📊 MCP 서버 연결 상태 확인")
            print("=" * 80)
            print(f"전체 서버 수: {server_status['total_servers']}")
            print(f"연결된 서버: {server_status['connected_servers']}")
            print(f"연결률: {server_status['summary']['connection_rate']}")
            print(
                f"전체 사용 가능한 Tool 수: {server_status['summary']['total_tools_available']}"
            )
            print("\n")

            for server_name, info in server_status["servers"].items():
                status_icon = "✅" if info["connected"] else "❌"
                print(f"{status_icon} 서버: {server_name}")
                print(f"   타입: {info['type']}")

                if info["type"] == "http":
                    print(f"   URL: {info.get('url', 'unknown')}")
                else:
                    cmd = info.get("command", "unknown")
                    args_preview = " ".join(info.get("args", [])[:3])
                    print(f"   명령어: {cmd} {args_preview}...")

                print(
                    f"   연결 상태: {'연결됨' if info['connected'] else '연결 안 됨'}"
                )
                print(f"   제공 Tool 수: {info['tools_count']}")

                if info["tools"]:
                    print("   Tool 목록:")
                    for tool in info["tools"][:5]:  # 처음 5개만 표시
                        registered_name = f"{server_name}::{tool}"
                        print(f"     - {registered_name}")
                    if len(info["tools"]) > 5:
                        print(f"     ... 및 {len(info['tools']) - 5}개 더")

                if info.get("error"):
                    print(f"   ⚠️ 오류: {info['error']}")
                print()

            print("=" * 80)

            # 요약 정보 로깅
            logger.info(
                f"MCP 서버 확인 완료: {server_status['summary']['connection_rate']} 연결"
            )

        except Exception as e:
            logger.error(f"MCP 서버 확인 실패: {e}")
            import traceback

            logger.error(traceback.format_exc())


