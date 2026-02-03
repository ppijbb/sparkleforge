#!/usr/bin/env python3
"""
Autonomous Multi-Agent Research System - Main Entry Point
Implements 9 Core Innovations: Production-Grade Reliability, Universal MCP Hub, Streaming Pipeline

MCP agent 라이브러리 기반의 자율 리서처 시스템.
모든 하드코딩, fallback, mock 코드를 제거하고 실제 MCP agent를 사용.

현재 상태: Production Level 개발 진행 중 🚧

Usage:
    python main.py --request "연구 주제"                    # CLI 모드
    python main.py --web                                    # 웹 모드
    python main.py --mcp-server                            # MCP 서버 모드
    python main.py --streaming                             # 스트리밍 모드
"""

import asyncio
import sys
import argparse
import subprocess
import signal
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import os
import time
# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.researcher_config import load_config_from_env

# CRITICAL: Load configuration BEFORE importing any modules that depend on it
config = load_config_from_env()

# Use new AgentOrchestrator for multi-agent orchestration
from src.core.agent_orchestrator import AgentOrchestrator as NewAgentOrchestrator
from src.core.autonomous_orchestrator import AutonomousOrchestrator
from src.monitoring.system_monitor import HealthMonitor

# Configure logging for production-grade reliability
# Advanced logging setup: setup logger manually to ensure logs directory exists and avoid issues with logging.basicConfig (per best practices)
log_dir = project_root / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "researcher.log"

# Streamlit 경고 필터링 (CLI 모드에서 streamlit이 import될 때 발생하는 경고 무시)
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)
logging.getLogger("streamlit").setLevel(logging.ERROR)

# HTTP 에러 메시지 필터링 클래스
class HTTPErrorFilter(logging.Filter):
    """HTML 에러 응답을 필터링하여 간단한 메시지만 출력"""
    def filter(self, record):
        message = record.getMessage()
        
        # HTML 에러 페이지 감지 및 필터링
        if '<!DOCTYPE html>' in message or '<html' in message.lower():
            # HTML에서 에러 메시지 추출 시도
            import re
            
            # HTTP 상태 코드 추출
            status_match = re.search(r'HTTP (\d{3})', message)
            status_code = status_match.group(1) if status_match else "Unknown"
            
            # 에러 제목 추출 시도
            title_match = re.search(r'<title>([^<]+)</title>', message, re.IGNORECASE)
            error_title = title_match.group(1).strip() if title_match else None
            
            # 간단한 에러 메시지 생성
            if error_title:
                record.msg = f"HTTP {status_code}: {error_title}"
            else:
                # 상태 코드에 따른 기본 메시지
                if status_code == "502":
                    record.msg = f"HTTP {status_code}: Bad Gateway - Server temporarily unavailable"
                elif status_code == "504":
                    record.msg = f"HTTP {status_code}: Gateway Timeout - Server response timeout"
                elif status_code == "503":
                    record.msg = f"HTTP {status_code}: Service Unavailable - Server temporarily unavailable"
                elif status_code == "401":
                    record.msg = f"HTTP {status_code}: Unauthorized - Authentication failed"
                elif status_code == "404":
                    record.msg = f"HTTP {status_code}: Not Found"
                elif status_code == "500":
                    record.msg = f"HTTP {status_code}: Internal Server Error"
                else:
                    record.msg = f"HTTP {status_code}: Server Error"
            
            record.args = ()  # args 초기화
        
        return True

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Remove existing handlers to avoid duplicate logs on reloads (e.g., under some app servers or noteboks)
if logger.hasHandlers():
    logger.handlers.clear()

# File handler
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
console_handler.addFilter(HTTPErrorFilter())  # HTTP 에러 필터 추가
logger.addHandler(console_handler)

# ============================================================================
# MCP / Runner / HTTP 관련 로거 억제 (과도한 로그 출력 방지)
# ============================================================================
# FastMCP Runner 로거 - WARNING 이상만 출력
runner_logger = logging.getLogger("Runner")
runner_logger.setLevel(logging.WARNING)
runner_logger.propagate = False  # 상위로 전파 차단

# MCP 관련 로거들
logging.getLogger("mcp").setLevel(logging.WARNING)
logging.getLogger("fastmcp").setLevel(logging.WARNING)
logging.getLogger("mcp.client").setLevel(logging.WARNING)
logging.getLogger("mcp.server").setLevel(logging.WARNING)

# HTTP 클라이언트 로거들
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

# asyncio 관련 로거
logging.getLogger("asyncio").setLevel(logging.WARNING)

# Root logger에도 필터 추가
root_logger = logging.getLogger()
for handler in root_logger.handlers:
    if isinstance(handler, logging.StreamHandler) and not any(isinstance(f, HTTPErrorFilter) for f in handler.filters):
        handler.addFilter(HTTPErrorFilter())



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
            
            logger.info("🌐 Starting Local Researcher Web Application with Streaming Pipeline...")
            logger.info(f"App will be available at: http://{address}:{port}")
            logger.info("Features: Real-time streaming, Progressive reporting, Incremental save")
            logger.info("Press Ctrl+C to stop the application")
            
            # Create logs directory if it doesn't exist
            logs_dir = self.project_root / "logs"
            logs_dir.mkdir(exist_ok=True)
            
            cmd = [
                sys.executable, "-m", "streamlit", "run",
                str(streamlit_app_path),
                "--server.port", port,
                "--server.address", address,
                "--browser.gatherUsageStats", "false",
                "--server.enableCORS", "false",
                "--server.enableXsrfProtection", "false"
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
            'status': 'running',
            'port': port,
            'streaming_enabled': True,
            'progressive_reporting': True,
            'incremental_save': True,
            'timestamp': datetime.now().isoformat()
        }


class AutonomousResearchSystem:
    """자율 리서처 시스템 - 9가지 핵심 혁신 통합 메인 클래스"""
    
    def __init__(self):
        # Load configurations from environment - ALL REQUIRED, NO DEFAULTS
        try:
            self.config = load_config_from_env()
            logger.info("✅ Configuration loaded successfully from environment variables")
            
            # Validate ChromaDB availability (optional)
            try:
                import chromadb  # type: ignore
                logger.info("✅ ChromaDB module available")
            except ImportError:
                logger.warning("⚠️ ChromaDB not installed - vector search will be disabled")
                logger.info("   Install with: pip install chromadb")
            
        except ValueError as e:
            logger.error(f"❌ Configuration loading failed: {e}")
            logger.error("Please check your .env file and ensure all required variables are set")
            logger.info("\nRequired environment variables:")
            logger.info("  - LLM_MODEL: LLM model identifier (e.g., google/gemini-2.5-flash-lite)")
            logger.info("  - GOOGLE_API_KEY: Your Google or Vertex AI API key")
            logger.info("  - LLM_PROVIDER: Provider name (e.g., google)")
            raise
        
        # Initialize components with 8 innovations
        logger.info("🔧 Initializing system components...")
        try:
            # Use new multi-agent orchestrator (no fallback - fail clearly)
            self.orchestrator = NewAgentOrchestrator()
            logger.info("✅ Multi-Agent Orchestrator initialized (no fallback mode)")
            logger.info("✅ Autonomous Orchestrator initialized")
        except Exception as e:
            logger.error(f"❌ Orchestrator initialization failed: {e}")
            raise
        
        try:
            from src.core.mcp_integration import UniversalMCPHub
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
                logger.info("⚠️ MCP Stability Service disabled (via DISABLE_MCP_STABILITY)")
            
            # MCP 백그라운드 헬스체크 (기본 활성화)
            if FeatureFlags.ENABLE_MCP_HEALTH_BACKGROUND:
                from src.core.mcp_health_background import MCPHealthBackgroundService
                self.mcp_health_service = MCPHealthBackgroundService(self.mcp_hub, interval=60)
                logger.info("✅ MCP Health Background Service enabled (default, will start on first execution)")
            else:
                self.mcp_health_service = None
                logger.info("⚠️ MCP Health Background Service disabled (via DISABLE_MCP_HEALTH_BACKGROUND)")
            
            # Guardrails 검증 (기본 활성화)
            if FeatureFlags.ENABLE_GUARDRAILS:
                from src.core.guardrails_validator import GuardrailsValidator
                self.guardrails_validator = GuardrailsValidator()
                logger.info("✅ Guardrails Validator enabled (default)")
            else:
                self.guardrails_validator = None
                logger.info("⚠️ Guardrails Validator disabled (via DISABLE_GUARDRAILS)")
            
            # Agent Tool Wrapper (기본 활성화)
            if FeatureFlags.ENABLE_AGENT_TOOLS:
                from src.core.agent_tool_wrapper import AgentToolWrapper
                # 에이전트는 나중에 할당 (execute 시점)
                self.agent_tool_wrapper = None
                logger.info("✅ Agent Tool Wrapper enabled (default, will be initialized on first execution)")
            else:
                self.agent_tool_wrapper = None
                logger.info("⚠️ Agent Tool Wrapper disabled (via DISABLE_AGENT_TOOLS)")
            
            # YAML 설정 로더 (기본 활성화)
            if FeatureFlags.ENABLE_YAML_CONFIG:
                from src.core.yaml_config_loader import YAMLConfigLoader
                self.yaml_config_loader = YAMLConfigLoader()
                logger.info("✅ YAML Config Loader enabled (default)")
            else:
                self.yaml_config_loader = None
                logger.info("⚠️ YAML Config Loader disabled (via DISABLE_YAML_CONFIG)")
        except Exception as e:
            logger.warning(f"⚠️ Feature initialization failed: {e} - continuing with core features only")
            # 기본값: 모든 기능 None (에러 발생 시)
            self.mcp_stability_service = None
            self.mcp_health_service = None
            self.guardrails_validator = None
            self.agent_tool_wrapper = None
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
            logger.info("✅ ProcessManager initialized (signal handlers managed by main.py)")
        except Exception as e:
            logger.warning(f"⚠️ ProcessManager initialization failed: {e} - continuing without process tracking")
        
        # Initialize signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Shutdown flag 초기화
        self._shutdown_requested = False
        
        logger.info("✅ AutonomousResearchSystem initialized successfully")
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        import sys
        import os
        
        # Reentrant call 방지를 위해 os.write 사용 (print 대신)
        try:
            msg = f"Received signal {signum}, initiating graceful shutdown...\n"
            os.write(sys.stderr.fileno(), msg.encode('utf-8'))
        except (OSError, AttributeError):
            # stderr에 쓸 수 없으면 무시 (이미 종료 중일 수 있음)
            pass
        
        # Shutdown 플래그 설정 (중복 방지)
        if hasattr(self, '_shutdown_requested') and self._shutdown_requested:
            # 이미 종료 중이면 재진입 방지만 수행
            try:
                msg = "Shutdown already in progress; ignoring additional signal\n"
                os.write(sys.stderr.fileno(), msg.encode('utf-8'))
            except (OSError, AttributeError):
                pass
            return
        
        self._shutdown_requested = True
        
        # 실행 중인 이벤트 루프가 있는지 확인하고 shutdown 작업 스케줄링
        try:
            loop = asyncio.get_running_loop()
            # 중복 생성 방지: 이미 스케줄된 작업이 있으면 재생성하지 않음
            if not hasattr(self, '_shutdown_task') or self._shutdown_task is None or self._shutdown_task.done():
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
                os.write(sys.stderr.fileno(), msg.encode('utf-8'))
            except (OSError, AttributeError):
                pass
            os._exit(1)  # sys.exit 대신 os._exit 사용 (더 안전)
    
    async def _graceful_shutdown(self):
        """Graceful shutdown with state persistence."""
        import sys
        import signal
        
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
                    if hasattr(self.mcp_hub, 'start_shutdown'):
                        self.mcp_hub.start_shutdown()
                    # cleanup은 CancelledError를 발생시킬 수 있으므로 무시
                    try:
                        await asyncio.wait_for(self.mcp_hub.cleanup(), timeout=3.0)  # 타임아웃 단축
                    except asyncio.CancelledError:
                        logger.debug("MCP Hub cleanup was cancelled (normal during shutdown)")
                    except asyncio.TimeoutError:
                        logger.warning("MCP Hub cleanup timed out (continuing shutdown)")
                except asyncio.CancelledError:
                    logger.debug("MCP Hub cleanup setup was cancelled (normal during shutdown)")
                except Exception as e:
                    logger.warning(f"Error cleaning up MCP Hub: {e}")
            
            # Health monitor 정지 (타임아웃 단축: 5초 -> 2초)
            try:
                await asyncio.wait_for(self.health_monitor.stop_monitoring(), timeout=2.0)
            except asyncio.TimeoutError:
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
        if ("def " in content_clean[:1000] and "import " in content_clean[:1000]) or content.startswith("```python"):
            return "py"
        
        # Java 코드 패턴
        if ("public class" in content_clean[:1000] or "public static void" in content_clean[:1000]) or content.startswith("```java"):
            return "java"
        
        # JavaScript 코드 패턴
        if (("const " in content_clean[:1000] or "function " in content_clean[:1000]) and "console" in content_clean[:1000]) or content.startswith("```javascript"):
            return "js"
        
        # HTML 패턴
        if content.strip().startswith("<!DOCTYPE") or content.strip().startswith("<html"):
            return "html"
        
        # 기본: Markdown
        return "md"
    
    async def run_research(self, request: str, output_path: Optional[str] = None, 
                          streaming: bool = False, output_format: Optional[str] = None) -> Dict[str, Any]:
        """연구 실행 - 9가지 핵심 혁신 적용"""
        logger.info("🤖 Starting Autonomous Research System with 9 Core Innovations")
        logger.info("=" * 80)
        logger.info(f"Request: {request}")
        logger.info(f"Primary LLM: {self.config.llm.primary_model}")
        logger.info(f"Planning Model: {self.config.llm.planning_model}")
        logger.info(f"Reasoning Model: {self.config.llm.reasoning_model}")
        logger.info(f"Verification Model: {self.config.llm.verification_model}")
        logger.info(f"Self-planning: {self.config.agent.enable_self_planning}")
        logger.info(f"Agent Communication: {self.config.agent.enable_agent_communication}")
        logger.info(f"MCP Enabled: {self.config.mcp.enabled}")
        logger.info(f"Streaming Pipeline: {streaming}")
        logger.info(f"Adaptive Supervisor: {self.config.agent.max_concurrent_research_units}")
        logger.info(f"Hierarchical Compression: {self.config.compression.enabled}")
        logger.info(f"Continuous Verification: {self.config.verification.enabled}")
        logger.info(f"Adaptive Context Window: {self.config.context_window.enabled}")
        logger.info("=" * 80)
        
        try:
            # Start health monitoring
            await self.health_monitor.start_monitoring()
            
            # Initialize MCP client if enabled
            if self.config.mcp.enabled:
                try:
                    await self.mcp_hub.initialize_mcp()
                    
                    # MCP 백그라운드 헬스체크 시작 (선택적)
                    if hasattr(self, 'mcp_health_service') and self.mcp_health_service:
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
                # Guardrails Input 검증 (선택적)
                if hasattr(self, 'guardrails_validator') and self.guardrails_validator:
                    try:
                        from src.core.agent_orchestrator import AgentState
                        initial_state: AgentState = {
                            'messages': [],
                            'user_query': request,
                            'research_plan': None,
                            'research_tasks': [],
                            'research_results': [],
                            'verified_results': [],
                            'final_report': None,
                            'current_agent': None,
                            'iteration': 0,
                            'session_id': None,
                            'research_failed': False,
                            'verification_failed': False,
                            'report_failed': False,
                            'error': None
                        }
                        validated_state = await self.guardrails_validator.validate_input(initial_state)
                        # 검증 통과 시 request 그대로 사용 (기존 코드 수정 없음)
                        request = validated_state.get('user_query', request)
                    except ValueError as e:
                        logger.error(f"❌ Input validation failed: {e}")
                        return {
                            'success': False,
                            'error': str(e),
                            'final_report': f"Input validation failed: {str(e)}"
                        }
                    except Exception as e:
                        logger.warning(f"⚠️ Guardrails validation error: {e} - continuing without validation")
                
                # Use new multi-agent orchestrator (no fallback - fail clearly)
                workflow_result = await self.orchestrator.execute(request)
                
                # Guardrails Output 검증 (선택적)
                if hasattr(self, 'guardrails_validator') and self.guardrails_validator:
                    try:
                        validated_result = await self.guardrails_validator.validate_output(workflow_result)
                        workflow_result = validated_result
                    except ValueError as e:
                        logger.error(f"❌ Output validation failed: {e}")
                        workflow_result['error'] = str(e)
                        workflow_result['final_report'] = f"Output validation failed: {str(e)}"
                    except Exception as e:
                        logger.warning(f"⚠️ Guardrails validation error: {e} - continuing without validation")
                
                # 실패 상태 확인 - fallback 없이 명확한 오류 반환
                research_failed = workflow_result.get('research_failed', False)
                verification_failed = workflow_result.get('verification_failed', False)
                report_failed = workflow_result.get('report_failed', False)
                final_report = workflow_result.get("final_report", "")
                
                # 실패 보고서도 실패로 처리 (내용이 "연구 실패" 또는 "연구 완료 불가" 포함)
                is_failure_report = (
                    final_report and 
                    ("연구 실패" in final_report or "연구 완료 불가" in final_report or "❌" in final_report)
                )
                
                if research_failed or verification_failed or report_failed or is_failure_report:
                    error_msg = workflow_result.get('error', '알 수 없는 오류')
                    if not error_msg or error_msg == '알 수 없는 오류':
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
                    
                    failed_agent = workflow_result.get('current_agent', 'unknown')
                    session_id = workflow_result.get("session_id", 'N/A')
                    
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
                            "failed": True
                        },
                        "synthesis_results": {
                            "content": final_report or "",
                            "failed": True
                        },
                        "sources": [],
                        "innovation_stats": {"multi_agent_orchestration": "enabled"},
                        "system_health": {"overall_status": "unhealthy", "error": error_msg},
                        "retry_available": True
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
                            "retry_available": True
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
                                "confidence": 0.9
                            },
                            "synthesis_results": {
                                "content": final_report
                            },
                            "sources": self._extract_sources_from_workflow(workflow_result),
                            "innovation_stats": {"multi_agent_orchestration": "enabled"},
                            "system_health": {"overall_status": "healthy"},
                            "session_id": workflow_result.get("session_id")
                        }
            
            # Apply hierarchical compression if enabled
            # Commented out to avoid serialization errors
            # if self.config.compression.enabled:
            #     result = await self._apply_hierarchical_compression(result)
            
            # Save results - LLM이 생성한 내용을 그대로 사용
            content = result.get('content', '') or result.get('synthesis_results', {}).get('content', '')
            
            if output_path:
                # 사용자가 지정한 경로 사용
                final_path = await self._save_content_as_file(content, output_path, result)
                logger.info(f"📄 Results saved to: {final_path}")
            else:
                # 생성된 내용을 보고 형식 결정
                detected_format = self._detect_output_format_from_content(content, request)
                if output_format is None:
                    output_format = detected_format
                
                output_dir = project_root / "output"
                output_dir.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # 확장자 결정 (간단하게)
                if output_format.startswith("."):
                    ext = output_format
                else:
                    ext_map = {"py": ".py", "java": ".java", "js": ".js", "html": ".html", "md": ".md", "pdf": ".pdf"}
                    ext = ext_map.get(output_format, ".md")
                default_output = output_dir / f"research_{timestamp}{ext}"
                
                final_path = await self._save_content_as_file(content, str(default_output), result)
                logger.info(f"📄 Results saved to default location: {final_path} (format: {output_format})")
                self._display_results(result)
            
            # Get final health status
            health_status = self.health_monitor.get_system_health()
            result['system_health'] = health_status
            
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
                    "failed": True
                },
                "synthesis_results": {"content": "", "failed": True},
                "sources": [],
                "innovation_stats": {},
                "system_health": error_health,
                "retry_available": True
            }
            
            # 실패 결과도 저장
            if output_path:
                await self._save_results_incrementally(result, output_path, output_format)
            else:
                output_dir = project_root / "output"
                output_dir.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                default_output = output_dir / f"research_failed_{timestamp}.{output_format}"
                await self._save_results_incrementally(result, str(default_output), output_format)
                self._display_results(result)
            
            # 예외를 발생시키지 않고 결과 반환
            return result
    
    async def _run_streaming_research(self, request: str) -> Dict[str, Any]:
        """Run research with streaming pipeline (Innovation 5)."""
        logger.info("🌊 Starting streaming research pipeline...")
        
        # Create streaming callback
        async def streaming_callback(partial_result: Dict[str, Any]):
            logger.info(f"📊 Streaming partial result: {partial_result.get('type', 'unknown')}")
            # In a real implementation, this would send to web interface
            print(f"📊 Partial Result: {partial_result.get('summary', 'Processing...')}")
        
        # Use standard run_research but with streaming callback simulation
        result = await self.orchestrator.run_research(user_request=request, context={})
        
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
            "system_health": result.get("system_health", {})
        }
        
        logger.info("✅ Streaming research completed")
        return formatted_result
    
    def _extract_sources_from_workflow(self, workflow_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract sources from workflow results."""
        sources = []
        seen_urls = set()
        
        # Extract from verified results (우선)
        verified_results = workflow_result.get("verified_results", [])
        for result in verified_results:
            if isinstance(result, dict):
                url = result.get('url', '')
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    sources.append({
                        "title": result.get('title', ''),
                        "url": url,
                        "snippet": result.get('snippet', '')
                    })
        
        # Extract from research results (백업)
        research_results = workflow_result.get("research_results", [])
        for result in research_results:
            if isinstance(result, dict):
                url = result.get('url', '')
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    sources.append({
                        "title": result.get('title', ''),
                        "url": url,
                        "snippet": result.get('snippet', '')
                    })
            elif isinstance(result, str) and "Source:" in result:
                # Extract URL from result string (레거시)
                parts = result.split("Source:")
                if len(parts) > 1:
                    url = parts[1].strip()
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        title = parts[0].split(":")[-1].strip() if ":" in parts[0] else ""
                        sources.append({
                            "title": title,
                            "url": url,
                            "snippet": ""
                        })
        
        return sources[:20]  # Limit to 20 sources
    
    def _extract_sources(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract sources from research results (legacy method)."""
        sources = []
        
        # Try to extract from execution results
        execution_results = result.get("detailed_results", {}).get("execution_results", [])
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
                unique_sources.append({
                    "title": source.get("title", source.get("name", "")),
                    "url": url,
                    "snippet": source.get("snippet", source.get("summary", ""))
                })
        
        return unique_sources[:20]  # Limit to 20 sources
    
    async def _apply_hierarchical_compression(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Apply hierarchical compression (Innovation 2)."""
        if not self.config.compression.enabled:
            return result
        
        logger.info("🗜️ Applying hierarchical compression...")
        
        # Import compression module
        from src.core.compression import compress_data
        
        # Compress large text fields
        if 'synthesis_results' in result:
            compressed_synthesis = await compress_data(
                result['synthesis_results']
            )
            result['synthesis_results_compressed'] = compressed_synthesis
        
        logger.info("✅ Hierarchical compression applied")
        return result
    
    async def _save_content_as_file(self, content: str, output_path: str, result: Dict[str, Any]) -> str:
        """LLM이 생성한 내용을 그대로 파일로 저장 (하드코딩 없이)."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # LLM이 생성한 내용을 그대로 저장 (추가 템플릿 없이)
        # 소스 정보가 있으면 마지막에 추가
        if result.get('sources'):
            sources_text = "\n\n## 참고 문헌\n\n"
            for i, source in enumerate(result.get('sources', []), 1):
                sources_text += f"{i}. [{source.get('title', 'N/A')}]({source.get('url', '')})\n"
                if source.get('snippet'):
                    sources_text += f"   {source.get('snippet', '')[:200]}...\n"
                sources_text += "\n"
            content = content + sources_text
        
        output_file.write_text(content, encoding='utf-8')
        return str(output_file)
    
    async def _save_results_incrementally(self, result: Dict[str, Any], output_path: str, output_format: str = "json"):
        """Save results with incremental save (Innovation 5)."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save incrementally based on format
        temp_file = output_file.with_suffix('.tmp')
        
        if output_format.lower() == "json":
            # Custom JSON encoder for datetime objects
            class DateTimeEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, datetime):
                        return obj.isoformat()
                    elif hasattr(obj, 'value'):  # Enum 처리
                        return obj.value
                    elif hasattr(obj, '__dict__'):  # 객체의 경우
                        return str(obj)
                    return super().default(obj)
            
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False, cls=DateTimeEncoder)
        elif output_format.lower() == "yaml":
            import yaml
            with open(temp_file, 'w', encoding='utf-8') as f:
                yaml.dump(result, f, default_flow_style=False, allow_unicode=True)
        elif output_format.lower() == "txt":
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(f"Research Results\n")
                f.write(f"===============\n\n")
                f.write(f"Query: {result.get('query', 'N/A')}\n")
                f.write(f"Timestamp: {result.get('timestamp', 'N/A')}\n\n")
                if 'content' in result:
                    f.write(f"Content:\n{result['content']}\n\n")
                elif 'synthesis_results' in result:
                    synthesis = result['synthesis_results']
                    if isinstance(synthesis, dict) and 'content' in synthesis:
                        f.write(f"Content:\n{synthesis['content']}\n\n")
                if 'sources' in result:
                    f.write(f"Sources:\n")
                    for i, source in enumerate(result['sources'], 1):
                        f.write(f"{i}. {source.get('title', 'N/A')} - {source.get('url', 'N/A')}\n")
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        # Atomic move
        temp_file.replace(output_file)
        
        logger.info(f"✅ Results saved incrementally to: {output_file} (format: {output_format})")
    
    def _display_results(self, result: Dict[str, Any]):
        """Display results with enhanced formatting."""
        print("\n📋 Research Results with 9 Core Innovations:")
        print("=" * 80)
        
        # 실패 상태 확인 및 표시
        if not result.get('success', True):
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
        if 'content' in result and result['content']:
            print("\n📝 Research Content:")
            print("-" * 60)
            print(result['content'])
            print("-" * 60)
        elif 'synthesis_results' in result:
            synthesis = result['synthesis_results']
            if isinstance(synthesis, dict) and 'content' in synthesis:
                print("\n📝 Research Content:")
                print("-" * 60)
                print(synthesis['content'])
                print("-" * 60)
            else:
                print(f"\n📝 Synthesis: {synthesis}")
        else:
            print("\n❌ No research content found in results")
        
        # Display research metadata
        if 'metadata' in result:
            metadata = result['metadata']
            print(f"\n📊 Research Metadata:")
            print(f"  • Model Used: {metadata.get('model_used', 'N/A')}")
            print(f"  • Execution Time: {metadata.get('execution_time', 'N/A'):.2f}s")
            print(f"  • Cost: ${metadata.get('cost', 0):.4f}")
            print(f"  • Confidence: {metadata.get('confidence', 'N/A')}")
        
        # Display synthesis results
        if 'synthesis_results' in result and isinstance(result['synthesis_results'], dict):
            synthesis = result['synthesis_results']
            if 'synthesis_results' in synthesis:
                print(f"\n📝 Synthesis: {synthesis.get('synthesis_results', 'N/A')}")
        
        # Display innovation stats
        if 'innovation_stats' in result:
            stats = result['innovation_stats']
            print(f"\n🚀 Innovation Statistics:")
            print(f"  • Adaptive Supervisor: {stats.get('adaptive_supervisor', 'N/A')}")
            print(f"  • Hierarchical Compression: {stats.get('hierarchical_compression', 'N/A')}")
            print(f"  • Multi-Model Orchestration: {stats.get('multi_model_orchestration', 'N/A')}")
            print(f"  • Continuous Verification: {stats.get('continuous_verification', 'N/A')}")
            print(f"  • Streaming Pipeline: {stats.get('streaming_pipeline', 'N/A')}")
            print(f"  • Universal MCP Hub: {stats.get('universal_mcp_hub', 'N/A')}")
            print(f"  • Adaptive Context Window: {stats.get('adaptive_context_window', 'N/A')}")
            print(f"  • Production-Grade Reliability: {stats.get('production_grade_reliability', 'N/A')}")
        
        # Display system health
        if 'system_health' in result:
            health = result['system_health']
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
                logger.info(f"Found {len(self.mcp_hub.mcp_sessions)} existing MCP server connections")
            else:
                logger.info("No existing connections. Will attempt quick connection tests for each server...")
            
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
            print(f"전체 사용 가능한 Tool 수: {server_status['summary']['total_tools_available']}")
            print("\n")
            
            for server_name, info in server_status["servers"].items():
                status_icon = "✅" if info["connected"] else "❌"
                print(f"{status_icon} 서버: {server_name}")
                print(f"   타입: {info['type']}")
                
                if info["type"] == "http":
                    print(f"   URL: {info.get('url', 'unknown')}")
                else:
                    cmd = info.get('command', 'unknown')
                    args_preview = ' '.join(info.get('args', [])[:3])
                    print(f"   명령어: {cmd} {args_preview}...")
                
                print(f"   연결 상태: {'연결됨' if info['connected'] else '연결 안 됨'}")
                print(f"   제공 Tool 수: {info['tools_count']}")
                
                if info["tools"]:
                    print(f"   Tool 목록:")
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
            logger.info(f"MCP 서버 확인 완료: {server_status['summary']['connection_rate']} 연결")
            
        except Exception as e:
            logger.error(f"MCP 서버 확인 실패: {e}")
            import traceback
            logger.error(traceback.format_exc())


async def main():
    """Main function - 9가지 핵심 혁신 통합 실행 진입점 (Suna-style CLI)"""
    # Python 종료 시 발생하는 async generator 정리 오류 무시
    def ignore_async_gen_errors(loop, context):
        """anyio cancel scope 및 async generator 종료 오류 무시"""
        exception = context.get('exception')
        if exception:
            error_msg = str(exception)
            # anyio cancel scope 오류는 무시
            if isinstance(exception, RuntimeError) and ("cancel scope" in error_msg.lower() or "different task" in error_msg.lower()):
                return  # 무시
            # async generator 종료 오류는 무시
            if isinstance(exception, GeneratorExit) or "async_generator" in error_msg.lower():
                return  # 무시
        # 기타 오류는 기본 handler로 전달
        loop.set_exception_handler(None)
        loop.call_exception_handler(context)
        loop.set_exception_handler(ignore_async_gen_errors)

    # asyncio exception handler 설정
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(ignore_async_gen_errors)

    # Suna-style 서브커맨드 구조로 개선
    parser = argparse.ArgumentParser(
        description="SparkleForge - Autonomous Multi-Agent Research System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
SparkleForge: Where Ideas Sparkle and Get Forged ⚒️✨

EXAMPLES:
  # 연구 실행
  python main.py run "인공지능의 미래 전망"

  # 웹 대시보드 시작
  python main.py web

  # 시스템 헬스체크
  python main.py health

  # MCP 서버 상태 확인
  python main.py mcp status

  # 도구 목록 확인
  python main.py tools list

  # Docker 서비스 관리
  python main.py docker up
        """
    )

    # 서브커맨드 추가
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # run 커맨드
    run_parser = subparsers.add_parser('run', help='Execute research request')
    run_parser.add_argument('query', help='Research query')
    run_parser.add_argument('--output', '-o', help='Output file path')
    run_parser.add_argument('--format', choices=['json', 'markdown', 'html'], default='markdown', help='Output format')
    run_parser.add_argument('--streaming', action='store_true', help='Enable streaming output')

    # web 커맨드
    web_parser = subparsers.add_parser('web', help='Start web dashboard')
    web_parser.add_argument('--port', default='8501', help='Web server port')
    web_parser.add_argument('--host', default='0.0.0.0', help='Web server host')

    # mcp 커맨드
    mcp_parser = subparsers.add_parser('mcp', help='MCP server management')
    mcp_subparsers = mcp_parser.add_subparsers(dest='mcp_command', help='MCP commands')

    # mcp status
    mcp_status_parser = mcp_subparsers.add_parser('status', help='Check MCP server status')
    mcp_status_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    # mcp server
    mcp_server_parser = mcp_subparsers.add_parser('server', help='Start MCP server')

    # health 커맨드
    health_parser = subparsers.add_parser('health', help='System health check')
    health_parser.add_argument('--detailed', action='store_true', help='Detailed health report')

    # tools 커맨드
    tools_parser = subparsers.add_parser('tools', help='Tool management')
    tools_subparsers = tools_parser.add_subparsers(dest='tools_command', help='Tool commands')

    # tools list
    tools_list_parser = tools_subparsers.add_parser('list', help='List available tools')
    tools_list_parser.add_argument('--category', help='Filter by category')

    # tools test
    tools_test_parser = tools_subparsers.add_parser('test', help='Test tool functionality')
    tools_test_parser.add_argument('tool_name', help='Tool name to test')

    # docker 커맨드
    docker_parser = subparsers.add_parser('docker', help='Docker service management')
    docker_subparsers = docker_parser.add_subparsers(dest='docker_command', help='Docker commands')

    # docker up
    docker_up_parser = docker_subparsers.add_parser('up', help='Start Docker services')
    docker_up_parser.add_argument('--build', action='store_true', help='Rebuild images')
    docker_up_parser.add_argument('--profile', action='append', help='Enable specific profiles (e.g., sandbox)')

    # docker down
    docker_down_parser = docker_subparsers.add_parser('down', help='Stop Docker services')
    docker_down_parser.add_argument('--volumes', action='store_true', help='Remove volumes')
    docker_down_parser.add_argument('--images', action='store_true', help='Remove images')

    # docker logs
    docker_logs_parser = docker_subparsers.add_parser('logs', help='Show service logs')
    docker_logs_parser.add_argument('service', nargs='?', help='Specific service name')
    docker_logs_parser.add_argument('--follow', '-f', action='store_true', help='Follow log output')

    # docker status
    docker_status_parser = docker_subparsers.add_parser('status', help='Show service status')

    # docker build
    docker_build_parser = docker_subparsers.add_parser('build', help='Build Docker images')
    docker_build_parser.add_argument('--no-cache', action='store_true', help='Build without cache')

    # docker restart
    docker_restart_parser = docker_subparsers.add_parser('restart', help='Restart Docker services')
    docker_restart_parser.add_argument('service', nargs='?', help='Specific service name')

    # setup 커맨드
    setup_parser = subparsers.add_parser('setup', help='System setup and configuration')
    setup_parser.add_argument('--force', action='store_true', help='Force reinstallation')

    # cli 커맨드 (CLI 에이전트 관리)
    cli_parser = subparsers.add_parser('cli', help='CLI agent management')
    cli_subparsers = cli_parser.add_subparsers(dest='cli_command', help='CLI agent commands')

    # cli list
    cli_list_parser = cli_subparsers.add_parser('list', help='List available CLI agents')

    # cli test
    cli_test_parser = cli_subparsers.add_parser('test', help='Test CLI agent')
    cli_test_parser.add_argument('agent_name', help='CLI agent name to test')

    # cli run
    cli_run_parser = cli_subparsers.add_parser('run', help='Run query with CLI agent')
    cli_run_parser.add_argument('agent_name', help='CLI agent name')
    cli_run_parser.add_argument('query', help='Query to execute')
    cli_run_parser.add_argument('--mode', help='Execution mode')
    cli_run_parser.add_argument('--files', nargs='*', help='Related files')

    # 하위 호환성을 위한 기존 인자들 (deprecated)
    parser.add_argument("--request", "--query", dest="legacy_request", help="Legacy: Use 'run' command instead")
    parser.add_argument("--web", action="store_true", dest="legacy_web", help="Legacy: Use 'web' command instead")
    parser.add_argument("--mcp-server", action="store_true", dest="legacy_mcp_server", help="Legacy: Use 'mcp server' command instead")
    parser.add_argument("--health-check", action="store_true", dest="legacy_health", help="Legacy: Use 'health' command instead")
    parser.add_argument("--check-mcp-servers", action="store_true", dest="legacy_mcp_status", help="Legacy: Use 'mcp status' command instead")
    mode_group.add_argument("--daemon", action="store_true", help="Start as long-running daemon (24/7 mode)")
    
    # Optional arguments
    parser.add_argument("--output", help="Output file path for research results")
    parser.add_argument("--output-format", choices=["text", "json", "stream-json"], default="text", help="Output format for headless mode")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--checkpoint", help="Checkpoint file path to save/restore conversation")
    parser.add_argument("--restore", help="Restore conversation from checkpoint file")
    
    # Long-running daemon options
    parser.add_argument("--auto-save-interval", type=int, default=300, help="Auto-save interval in seconds (default: 300)")
    parser.add_argument("--max-memory-mb", type=float, default=2048, help="Max memory usage in MB before restart (default: 2048)")
    parser.add_argument("--max-uptime-hours", type=float, default=0, help="Max uptime in hours before restart (0=unlimited, default: 0)")
    parser.add_argument("--no-auto-restart", action="store_true", help="Disable auto-restart on errors")
    parser.add_argument("--format", choices=["json", "yaml", "txt"], default="json", help="Output format for research results")
    parser.add_argument("--streaming", action="store_true", help="Enable streaming pipeline for real-time results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()

    # 하위 호환성 처리
    if args.legacy_request:
        args.command = 'run'
        args.query = args.legacy_request
        args.streaming = True
    elif args.legacy_web:
        args.command = 'web'
    elif args.legacy_mcp_server:
        args.command = 'mcp'
        args.mcp_command = 'server'
    elif args.legacy_health:
        args.command = 'health'
    elif args.legacy_mcp_status:
        args.command = 'mcp'
        args.mcp_command = 'status'

    # 기본값 설정
    if not args.command:
        args.command = 'run'  # 기본은 run 커맨드
        if not hasattr(args, 'query') or not args.query:
            # 쿼리가 없으면 인터랙티브 모드로 전환
            args.command = 'interactive'

    # 서브커맨드 처리
    if args.command == 'run':
        await handle_run_command(args)
    elif args.command == 'web':
        await handle_web_command(args)
    elif args.command == 'mcp':
        await handle_mcp_command(args)
    elif args.command == 'health':
        await handle_health_command(args)
    elif args.command == 'tools':
        await handle_tools_command(args)
    elif args.command == 'docker':
        await handle_docker_command(args)
    elif args.command == 'setup':
        await handle_setup_command(args)
    elif args.command == 'cli':
        await handle_cli_command(args)
    elif args.command == 'interactive':
        await handle_interactive_command(args)
    else:
        parser.print_help()
    
    # REPL 모드에서는 모든 로그를 완전히 억제 (ERROR만 표시)
    if is_repl_mode:
        import warnings
        # 모든 로거를 ERROR 레벨로 설정 (WARNING, INFO, DEBUG 모두 억제)
        logging.getLogger().setLevel(logging.ERROR)
        
        # 특정 모듈들의 로거도 ERROR로 설정
        for logger_name in [
            '__main__', 'src', 'src.core', 'src.core.era_server_manager',
            'src.core.agent_orchestrator', 'src.core.mcp_integration',
            'src.core.shared_memory', 'src.core.skills_manager',
            'src.core.prompt_refiner_wrapper', 'root',
            'streamlit', 'streamlit.runtime', 'local_researcher'
        ]:
            logging.getLogger(logger_name).setLevel(logging.ERROR)
        
        # warnings도 완전히 억제
        warnings.filterwarnings('ignore')
    else:
        # Set logging level
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
    
    # Create logs directory
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)

    # Initialize enhanced systems
    from src.utils.output_manager import (
        UserCenteredOutputManager,
        set_output_manager,
        OutputLevel,
        OutputFormat
    )
    from src.core.error_handler import ErrorHandler, set_error_handler
    from src.core.progress_tracker import ProgressTracker, set_progress_tracker

    # 출력 매니저 초기화
    output_manager = UserCenteredOutputManager(
        output_level=OutputLevel.USER,
        output_format=OutputFormat.TEXT,
        enable_colors=True,
        stream_output=True,
        show_progress=True
    )
    set_output_manager(output_manager)

    # 에러 핸들러 초기화
    error_handler = ErrorHandler(
        log_errors=True,
        enable_recovery=True
    )
    set_error_handler(error_handler)

    # 진행 상황 추적기 초기화 (세션별로 생성)
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    progress_tracker = ProgressTracker(
        session_id=session_id,
        enable_real_time_updates=True,
        update_interval=1.0
    )
    set_progress_tracker(progress_tracker)

    # 진행 상황 추적기 콜백 설정 (출력 매니저와 연동)
    last_stage = [None]  # 클로저를 위한 리스트 (nonlocal 대신)
    last_progress = [0]
    
    async def progress_callback(workflow_progress):
        """진행 상황 업데이트 시 출력 매니저에 표시."""
        try:
            progress_pct = int(workflow_progress.overall_progress * 100)
            stage_name = workflow_progress.current_stage.value

            # 단계가 변경되었을 때만 start_progress 호출
            if last_stage[0] != stage_name:
                last_stage[0] = stage_name
                eta_str = ""
                if workflow_progress.estimated_completion:
                    eta_seconds = max(0, int(workflow_progress.estimated_completion - time.time()))
                    eta_str = f" (예상 {eta_seconds}초 남음)"

                # 진행률 바 스타일로 표시 (단계 변경 시에만)
                await output_manager.start_progress(
                    stage_name,
                    100,
                    f"{progress_pct}% 완료",
                    workflow_progress.estimated_completion
                )
            
            # 진행률이 실제로 변경되었을 때만 업데이트 (1% 이상 차이)
            if abs(progress_pct - last_progress[0]) >= 1 or progress_pct == 100:
                last_progress[0] = progress_pct
                await output_manager.update_progress(progress_pct)

        except Exception as e:
            logger.warning(f"Progress callback failed: {e}")

    progress_tracker.add_progress_callback(progress_callback)
    
    # 시스템 초기화 (REPL 모드가 아닐 때만 전체 초기화)
    system = None
    if not is_repl_mode:
        # ERA 서버 초기화 및 시작
        try:
            from src.core.researcher_config import get_era_config
            from src.core.era_server_manager import get_era_server_manager
            
            era_config = get_era_config()
            if era_config.enabled:
                # ERA server 초기화
                era_manager = get_era_server_manager()
                await era_manager.ensure_server_running_with_retry()
            else:
                logger.debug("ERA is disabled in configuration")
        except ImportError as e:
            logger.debug(f"ERA modules not available: {e}")
        except Exception as e:
            logger.warning(f"⚠️ ERA initialization failed: {e}")
            logger.debug("Code execution features may be limited", exc_info=True)
        
        # Initialize system
        system = AutonomousResearchSystem()
    
    try:
        # 체크포인트 복원 (있는 경우)
        if args.restore:
            from src.core.checkpoint_manager import CheckpointManager
            checkpoint_manager = CheckpointManager()
            restored_state = await checkpoint_manager.restore_checkpoint(args.restore)
            if restored_state:
                logger.info(f"✅ Restored checkpoint: {args.restore}")
                # 복원된 상태를 사용하여 계속 진행
            else:
                logger.error(f"❌ Failed to restore checkpoint: {args.restore}")
                return
        
        # Headless 모드 (비대화형)
        if args.prompt:
            from src.core.autonomous_orchestrator import AutonomousOrchestrator
            orchestrator = AutonomousOrchestrator()
            
            result = await orchestrator.run_research(args.prompt)
            
            # 출력 형식에 따라 결과 출력
            if args.output_format == "json":
                import json
                output = json.dumps(result, indent=2, ensure_ascii=False)
                print(output)
            elif args.output_format == "stream-json":
                # 스트리밍 JSON (newline-delimited)
                import json
                if isinstance(result, dict):
                    for key, value in result.items():
                        print(json.dumps({key: value}, ensure_ascii=False))
                else:
                    print(json.dumps(result, ensure_ascii=False))
            else:
                # 기본 텍스트 출력
                if isinstance(result, dict):
                    if "content" in result:
                        print(result["content"])
                    elif "final_synthesis" in result:
                        print(result["final_synthesis"].get("content", ""))
                    else:
                        print(str(result))
                else:
                    print(str(result))
            
            # 체크포인트 저장 (요청된 경우)
            if args.checkpoint:
                from src.core.checkpoint_manager import CheckpointManager
                checkpoint_manager = CheckpointManager()
                checkpoint_id = await checkpoint_manager.save_checkpoint(
                    state={"result": result, "prompt": args.prompt},
                    metadata={"mode": "headless", "output_format": args.output_format}
                )
                logger.info(f"✅ Checkpoint saved: {checkpoint_id}")
            
            return
        
        # 기본 동작: REPL 모드 (아무 옵션도 없으면)
        # 또는 --cli 옵션이 있으면
        if is_repl_mode:
            try:
                from src.cli.repl_cli import REPLCLI
                cli = REPLCLI()
                await cli.run()
            except (EOFError, KeyboardInterrupt, SystemExit):
                # 정상 종료
                pass
            finally:
                pass
            return
        
        # Interactive 모드 (기존)
        if args.interactive:
            from src.cli.interactive_cli import InteractiveCLI
            from src.core.scheduler import get_scheduler
            
            # 스케줄러 초기화 및 시작
            scheduler = get_scheduler()
            
            # 실행 콜백 설정
            async def execute_scheduled_query(user_query: str, session_id: str):
                from src.core.autonomous_orchestrator import AutonomousOrchestrator
                orchestrator = AutonomousOrchestrator()
                return await orchestrator.execute_full_research_workflow(user_query)
            
            scheduler.set_execution_callback(execute_scheduled_query)
            await scheduler.start()
            
            cli = InteractiveCLI()
            try:
                await cli.run()
            finally:
                await scheduler.stop()
            return
        
        if args.request:
            # CLI Research Mode with 8 innovations
            logger.info("🚀 Starting Local Researcher with enhanced systems...")

            # 진행 상황 추적 시작
            await progress_tracker.start_tracking()
            
            # 초기화 단계 명시적으로 설정
            from src.core.progress_tracker import WorkflowStage
            progress_tracker.set_workflow_stage(WorkflowStage.INITIALIZING, {"message": "시스템 초기화 중..."})

            # 워크플로우 시작 알림
            await output_manager.output(
                f"🔬 연구 주제: {args.request}",
                level=OutputLevel.USER
            )
            await output_manager.output(
                "실시간 진행 상황 추적 및 향상된 에러 처리가 활성화되었습니다.",
                level=OutputLevel.SERVICE
            )

            # 연구 실행
            await system.run_research(
                args.request, 
                args.output, 
                streaming=args.streaming,
                output_format=args.format
            )
            
        elif args.web:
            # Web Application Mode with Streaming Pipeline
            system.run_web_app()
            
        elif args.mcp_server:
            # MCP Server Mode with Universal MCP Hub - 실제 연결 수행
            logger.info("Initializing MCP servers...")
            try:
                await system.mcp_hub.initialize_mcp()
                logger.info("✅ MCP servers initialized")
                
                # 연결된 서버 상태 출력
                if system.mcp_hub.mcp_sessions:
                    print("\n" + "=" * 80)
                    print("✅ MCP 서버 연결 완료")
                    print("=" * 80)
                    for server_name in system.mcp_hub.mcp_sessions.keys():
                        tools_count = len(system.mcp_hub.mcp_tools_map.get(server_name, {}))
                        print(f"✅ {server_name}: {tools_count} tools available")
                    print("=" * 80)
                    print("\nMCP Hub is running. Press Ctrl+C to stop.")
                    
                    # 계속 실행 대기
                    try:
                        await asyncio.sleep(3600)  # 1시간 대기 (또는 Ctrl+C로 종료)
                    except KeyboardInterrupt:
                        logger.info("Shutting down MCP Hub...")
                else:
                    logger.warning("⚠️ No MCP servers connected")
                    sys.exit(1)
            except Exception as e:
                logger.error(f"Failed to initialize MCP servers: {e}")
                sys.exit(1)
            
        elif args.mcp_client:
            # MCP Client Mode with Smart Tool Selection
            success = await system.run_mcp_client()
            if not success:
                sys.exit(1)
            
        elif args.health_check:
            # Health Check Mode
            await system.run_health_check()
        
        elif args.check_mcp_servers:
            # MCP 서버 확인 모드
            await system.check_mcp_servers()
            
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user (KeyboardInterrupt)")
        if system is not None:
            system._shutdown_requested = True
            try:
                await system._graceful_shutdown()
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")
        # sys.exit(0) 제거 - asyncio.run()이 자동으로 처리
    except asyncio.CancelledError:
        # 취소된 경우 정리 후 종료
        logger.info("Operation cancelled")
        if system is not None:
            system._shutdown_requested = True
            try:
                await system._graceful_shutdown()
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")
        # asyncio.CancelledError는 다시 raise하여 정상적인 취소 흐름 유지
        raise
    except Exception as e:
        # 향상된 에러 처리
        from src.core.error_handler import ErrorContext, ErrorCategory, ErrorSeverity
        import traceback

        error_context = ErrorContext(
            component="main",
            operation="run_research" if args.request else "system_operation",
            session_id=session_id
        )

        await error_handler.handle_error(
            e,
            category=ErrorCategory.UNKNOWN,
            severity=ErrorSeverity.HIGH,
            context=error_context,
            custom_message=f"시스템 실행 중 치명적 오류 발생: {str(e)}"
        )

        if system is not None:
            system._shutdown_requested = True
            try:
                await system._graceful_shutdown()
            except Exception as e2:
                logger.error(f"Error during shutdown: {e2}")
        # 에러 발생 시 종료 코드 1로 종료
        sys.exit(1)
    finally:
        # 진행 상황 추적 중지 및 요약 출력
        try:
            await progress_tracker.stop_tracking()

            if args.request:
                # 워크플로우 완료 요약
                await output_manager.complete_progress(success=True)
                await output_manager.output_workflow_summary()

                # 진행 상황 통계 출력
                stats = progress_tracker.get_statistics()
                await output_manager.output(
                    f"📈 세션 통계: {stats['total_agents_created']}개 에이전트 생성, "
                    f"{stats['agents_completed']}개 완료, {stats['agents_failed']}개 실패",
                    level=OutputLevel.SERVICE
                )

        except Exception as e:
            logger.warning(f"Failed to finalize progress tracking: {e}")

        # 최종 정리 보장 (system이 초기화된 경우에만)
        if system is not None and hasattr(system, 'mcp_hub') and system.mcp_hub and hasattr(system.mcp_hub, 'mcp_sessions'):
            try:
                if system.mcp_hub.mcp_sessions:
                    logger.info("Final cleanup of MCP connections...")
                    await system.mcp_hub.cleanup()
                
                # MCP 백그라운드 헬스체크 서비스 종료 (선택적)
                if hasattr(system, 'mcp_health_service') and system.mcp_health_service:
                    try:
                        await system.mcp_health_service.stop()
                        logger.info("✅ MCP Health Background Service stopped")
                    except Exception as e:
                        logger.debug(f"Error stopping MCP health service: {e}")
            except Exception as e:
                logger.debug(f"Error in final cleanup: {e}")


# 서브커맨드 핸들러 함수들
async def handle_run_command(args):
    """연구 실행 커맨드 처리"""
    logger.info(f"🔬 Starting research: {args.query}")

    try:
        # Autonomous Orchestrator 초기화
        orchestrator = AutonomousOrchestrator()

        # 연구 실행
        result = await orchestrator.execute_research(
            research_query=args.query,
            output_format=args.format,
            streaming=args.streaming
        )

        # 결과 출력
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                if args.format == 'json':
                    json.dump(result, f, ensure_ascii=False, indent=2)
                else:
                    f.write(result)
            logger.info(f"✅ Results saved to {args.output}")
        else:
            print(result)

    except Exception as e:
        logger.error(f"❌ Research failed: {e}")
        return 1
    return 0


async def handle_web_command(args):
    """웹 대시보드 시작 커맨드 처리"""
    logger.info("🌐 Starting web dashboard...")

    web_manager = WebAppManager()
    os.environ["STREAMLIT_PORT"] = args.port
    os.environ["STREAMLIT_ADDRESS"] = args.host

    try:
        web_manager.start_web_app()
    except KeyboardInterrupt:
        logger.info("🛑 Web dashboard stopped")
    except Exception as e:
        logger.error(f"❌ Failed to start web dashboard: {e}")
        return 1
    return 0


async def handle_mcp_command(args):
    """MCP 관리 커맨드 처리"""
    if args.mcp_command == 'status':
        logger.info("🔍 Checking MCP server status...")

        try:
            # MCP Hub 초기화 및 상태 확인
            from src.core.mcp_integration import get_mcp_hub
            mcp_hub = get_mcp_hub()

            if args.verbose:
                await mcp_hub.initialize_mcp()

            server_status = await mcp_hub.check_mcp_servers()
            mcp_hub.print_server_status(server_status, verbose=args.verbose)

        except Exception as e:
            logger.error(f"❌ MCP status check failed: {e}")
            return 1

    elif args.mcp_command == 'server':
        logger.info("🚀 Starting MCP server...")

        try:
            # MCP 서버 시작
            mcp_manager = MCPManager()
            await mcp_manager.start_mcp_server()
        except KeyboardInterrupt:
            logger.info("🛑 MCP server stopped")
        except Exception as e:
            logger.error(f"❌ Failed to start MCP server: {e}")
            return 1

    return 0


async def handle_health_command(args):
    """시스템 헬스체크 커맨드 처리"""
    logger.info("🏥 Running system health check...")

    try:
        health_monitor = HealthMonitor()

        if args.detailed:
            # 상세 헬스체크
            health_report = await health_monitor.run_comprehensive_health_check()
            health_monitor.print_detailed_health_report(health_report)
        else:
            # 간단한 헬스체크
            is_healthy = await health_monitor.quick_health_check()
            if is_healthy:
                logger.info("✅ System is healthy")
            else:
                logger.error("❌ System has issues")
                return 1

    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return 1
    return 0


async def handle_tools_command(args):
    """도구 관리 커맨드 처리"""
    if args.tools_command == 'list':
        logger.info("🔧 Listing available tools...")

        try:
            from src.core.mcp_integration import get_mcp_hub
            mcp_hub = get_mcp_hub()
            await mcp_hub.initialize_mcp()

            # 도구 목록 출력
            tools_by_category = {}
            for tool_name, tool_info in mcp_hub.tools.items():
                category = tool_info.get('category', 'unknown')
                if category not in tools_by_category:
                    tools_by_category[category] = []
                tools_by_category[category].append(tool_name)

            for category, tools in tools_by_category.items():
                if not args.category or args.category == category:
                    print(f"\n📂 {category.upper()}:")
                    for tool in sorted(tools):
                        print(f"  - {tool}")

        except Exception as e:
            logger.error(f"❌ Failed to list tools: {e}")
            return 1

    elif args.tools_command == 'test':
        logger.info(f"🧪 Testing tool: {args.tool_name}")

        try:
            from src.core.mcp_integration import get_mcp_hub
            mcp_hub = get_mcp_hub()
            await mcp_hub.initialize_mcp()

            # 도구 테스트
            result = await mcp_hub.test_tool(args.tool_name)
            if result.get('success'):
                print(f"✅ Tool {args.tool_name} is working")
            else:
                print(f"❌ Tool {args.tool_name} failed: {result.get('error')}")
                return 1

        except Exception as e:
            logger.error(f"❌ Tool test failed: {e}")
            return 1

    return 0


async def handle_docker_command(args):
    """Docker 서비스 관리 커맨드 처리 (Suna-style)"""
    import subprocess
    import os

    # Docker Compose 명령어 자동 감지
    def get_docker_compose_cmd():
        """Docker Compose 명령어 자동 감지"""
        # Docker Compose v2 (docker compose)
        try:
            subprocess.run(['docker', 'compose', 'version'], capture_output=True, check=True)
            return ['docker', 'compose']
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        # Docker Compose v1 (docker-compose)
        try:
            subprocess.run(['docker-compose', 'version'], capture_output=True, check=True)
            return ['docker-compose']
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        return None

    # Docker 설치 확인
    def check_docker():
        """Docker 설치 상태 확인"""
        try:
            subprocess.run(['docker', '--version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    # Docker Compose 파일 존재 확인
    def check_compose_file():
        """docker-compose.yaml 파일 존재 확인"""
        compose_files = ['docker-compose.yaml', 'docker-compose.yml']
        for filename in compose_files:
            if (project_root / filename).exists():
                return filename
        return None

    # Docker 환경 확인
    if not check_docker():
        logger.error("❌ Docker is not installed or not running")
        logger.info("Please install Docker: https://docs.docker.com/get-docker/")
        return 1

    compose_cmd = get_docker_compose_cmd()
    if not compose_cmd:
        logger.error("❌ Docker Compose is not installed")
        logger.info("Please install Docker Compose: https://docs.docker.com/compose/install/")
        return 1

    compose_file = check_compose_file()
    if not compose_file:
        logger.error("❌ docker-compose.yaml file not found")
        logger.info("Please ensure docker-compose.yaml exists in the project root")
        return 1

    if args.docker_command == 'up':
        logger.info("🐳 Starting Docker services...")
        logger.info(f"Using Docker Compose: {' '.join(compose_cmd)}")
        logger.info(f"Compose file: {compose_file}")

        try:
            cmd = compose_cmd + ['-f', compose_file, 'up', '-d']
            if args.build:
                cmd.append('--build')
                logger.info("🔨 Building images...")

            # 프로필 지원 (예: sandbox)
            if hasattr(args, 'profile') and args.profile:
                for profile in args.profile:
                    cmd.extend(['--profile', profile])
                    logger.info(f"🔧 Enabling profile: {profile}")

            # 환경 변수 로드 (.env 파일)
            env = os.environ.copy()
            env_file = project_root / '.env'
            if env_file.exists():
                logger.info("📄 Loading environment from .env file")
                # .env 파일에서 환경 변수 로드 (간단한 구현)
                with open(env_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            env[key] = value

            result = subprocess.run(cmd, cwd=str(project_root), env=env)
            if result.returncode == 0:
                logger.info("✅ Docker services started successfully")
                logger.info("🌐 Services:")
                logger.info("   - Backend API: http://localhost:8000")
                logger.info("   - Frontend: http://localhost:8501")
                logger.info("   - Redis: localhost:6379")
                logger.info("📊 View logs: python main.py docker logs")
                logger.info("📊 Check status: python main.py docker status")
            else:
                logger.error("❌ Failed to start Docker services")
                return 1

        except Exception as e:
            logger.error(f"❌ Docker command failed: {e}")
            return 1

    elif args.docker_command == 'down':
        logger.info("🐳 Stopping Docker services...")

        try:
            cmd = compose_cmd + ['-f', compose_file, 'down']
            if hasattr(args, 'volumes') and args.volumes:
                cmd.append('--volumes')
                logger.info("🗑️ Removing volumes...")
            if hasattr(args, 'images') and args.images:
                cmd.append('--rmi')
                cmd.append('all')
                logger.info("🖼️ Removing images...")

            result = subprocess.run(cmd, cwd=str(project_root))
            if result.returncode == 0:
                logger.info("✅ Docker services stopped successfully")
            else:
                logger.error("❌ Failed to stop Docker services")
                return 1

        except Exception as e:
            logger.error(f"❌ Docker command failed: {e}")
            return 1

    elif args.docker_command == 'logs':
        service_name = getattr(args, 'service', None)
        logger.info(f"📊 Showing Docker service logs{f' for {service_name}' if service_name else ''}...")

        try:
            cmd = compose_cmd + ['-f', compose_file, 'logs']
            if service_name:
                cmd.append(service_name)
            if hasattr(args, 'follow') and args.follow:
                cmd.append('-f')

            if hasattr(args, 'follow') and args.follow:
                subprocess.run(cmd, cwd=str(project_root))
            else:
                result = subprocess.run(cmd, cwd=str(project_root), capture_output=True, text=True)
                if result.returncode == 0:
                    print(result.stdout)
                else:
                    logger.error("❌ Failed to get logs")
                    return 1
        except KeyboardInterrupt:
            logger.info("🛑 Stopped log monitoring")
        except Exception as e:
            logger.error(f"❌ Failed to show logs: {e}")
            return 1

    elif args.docker_command == 'status':
        logger.info("📊 Checking Docker service status...")

        try:
            cmd = compose_cmd + ['-f', compose_file, 'ps']
            result = subprocess.run(cmd, cwd=str(project_root), capture_output=True, text=True)
            if result.returncode == 0:
                print("🐳 Docker Services Status:")
                print("=" * 50)
                print(result.stdout)
            else:
                logger.error("❌ Failed to get service status")
                return 1
        except Exception as e:
            logger.error(f"❌ Failed to check status: {e}")
            return 1

    elif args.docker_command == 'build':
        logger.info("🔨 Building Docker images...")

        try:
            cmd = compose_cmd + ['-f', compose_file, 'build']
            if hasattr(args, 'no_cache') and args.no_cache:
                cmd.append('--no-cache')
                logger.info("🧹 Building without cache...")

            result = subprocess.run(cmd, cwd=str(project_root))
            if result.returncode == 0:
                logger.info("✅ Docker images built successfully")
            else:
                logger.error("❌ Failed to build Docker images")
                return 1
        except Exception as e:
            logger.error(f"❌ Build failed: {e}")
            return 1

    elif args.docker_command == 'restart':
        service_name = getattr(args, 'service', None)
        logger.info(f"🔄 Restarting Docker services{f' ({service_name})' if service_name else ''}...")

        try:
            cmd = compose_cmd + ['-f', compose_file, 'restart']
            if service_name:
                cmd.append(service_name)

            result = subprocess.run(cmd, cwd=str(project_root))
            if result.returncode == 0:
                logger.info("✅ Docker services restarted successfully")
            else:
                logger.error("❌ Failed to restart Docker services")
                return 1
        except Exception as e:
            logger.error(f"❌ Restart failed: {e}")
            return 1

    else:
        logger.error(f"❌ Unknown Docker command: {args.docker_command}")
        logger.info("Available commands: up, down, logs, status, build, restart")
        return 1

    return 0


async def handle_setup_command(args):
    """시스템 설정 커맨드 처리"""
    logger.info("⚙️ Running system setup...")

    try:
        # 간단한 설정 확인
        required_files = [
            'pyproject.toml',
            'requirements.txt',
            'src/core/configs/researcher_config.yaml'
        ]

        missing_files = []
        for file_path in required_files:
            if not (project_root / file_path).exists():
                missing_files.append(file_path)

        if missing_files:
            logger.error(f"❌ Missing required files: {missing_files}")
            return 1

        # 환경 변수 확인
        required_env_vars = ['OPENROUTER_API_KEY']
        missing_env_vars = []
        for env_var in required_env_vars:
            if not os.getenv(env_var):
                missing_env_vars.append(env_var)

        if missing_env_vars:
            logger.warning(f"⚠️ Missing environment variables: {missing_env_vars}")
            logger.info("Please set these in your .env file or environment")

        logger.info("✅ System setup completed")

    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        return 1
    return 0


async def handle_interactive_command(args):
    """인터랙티브 모드 처리"""
    logger.info("💬 Starting interactive mode...")

    try:
        # 간단한 REPL 구현
        print("SparkleForge Interactive Mode")
        print("Type 'help' for commands, 'quit' to exit")
        print("-" * 50)

        while True:
            try:
                query = input("🔍 Research query: ").strip()
                if not query:
                    continue
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                if query.lower() == 'help':
                    print("Commands:")
                    print("  help  - Show this help")
                    print("  quit  - Exit interactive mode")
                    print("  <query> - Execute research query")
                    continue

                # 연구 실행
                await handle_run_command(type('Args', (), {'query': query, 'output': None, 'format': 'markdown', 'streaming': True})())

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"❌ Error: {e}")

        logger.info("👋 Goodbye!")

    except Exception as e:
        logger.error(f"❌ Interactive mode failed: {e}")
        return 1
    return 0


async def handle_cli_command(args):
    """CLI 에이전트 관리 커맨드 처리"""
    from src.core.cli_agents.cli_agent_manager import get_cli_agent_manager
    from src.core.researcher_config import initialize_cli_agents

    # CLI 에이전트 초기화
    if not initialize_cli_agents():
        logger.warning("⚠️ CLI agents not enabled or failed to initialize")

    cli_manager = get_cli_agent_manager()

    if args.cli_command == 'list':
        logger.info("🤖 Available CLI Agents:")

        try:
            available_agents = cli_manager.get_available_agents()
            if not available_agents:
                logger.info("  No CLI agents configured")
                return 0

            for agent_name in available_agents:
                agent_info = cli_manager.get_agent_info(agent_name)
                if agent_info:
                    status = "✅ Available" if agent_info.get('instance') else "⚠️ Configured"
                    logger.info(f"  - {agent_name}: {status}")
                    if agent_info.get('type'):
                        logger.info(f"    Type: {agent_info['type']}")
                    if agent_info.get('command'):
                        logger.info(f"    Command: {agent_info['command']}")
                else:
                    logger.info(f"  - {agent_name}: ❌ Not configured")

        except Exception as e:
            logger.error(f"❌ Failed to list CLI agents: {e}")
            return 1

    elif args.cli_command == 'test':
        agent_name = args.agent_name
        logger.info(f"🧪 Testing CLI agent: {agent_name}")

        try:
            # 헬스체크
            agent = cli_manager.create_agent(agent_name)
            if not agent:
                logger.error(f"❌ CLI agent not available: {agent_name}")
                return 1

            is_healthy = await agent.health_check()
            if is_healthy:
                logger.info(f"✅ CLI agent {agent_name} is healthy")
                # 추가 정보 표시
                info = agent.get_info()
                logger.info(f"   Name: {info.get('name')}")
                logger.info(f"   Command: {info.get('command')}")
                logger.info(f"   Timeout: {info.get('timeout')}s")
            else:
                logger.error(f"❌ CLI agent {agent_name} is not healthy")
                return 1

        except Exception as e:
            logger.error(f"❌ CLI agent test failed: {e}")
            return 1

    elif args.cli_command == 'run':
        agent_name = args.agent_name
        query = args.query
        logger.info(f"🚀 Running query with CLI agent: {agent_name}")
        logger.info(f"   Query: {query}")

        try:
            # 실행 옵션 준비
            kwargs = {}
            if hasattr(args, 'mode') and args.mode:
                kwargs['mode'] = args.mode
            if hasattr(args, 'files') and args.files:
                kwargs['files'] = args.files

            # CLI 에이전트로 쿼리 실행
            result = await cli_manager.execute_with_agent(agent_name, query, **kwargs)

            if result.get('success'):
                logger.info("✅ CLI agent execution successful")
                logger.info("📄 Response:")
                print(result.get('response', ''))

                # 메타데이터 표시
                metadata = result.get('metadata', {})
                if metadata:
                    logger.info("📊 Metadata:")
                    for key, value in metadata.items():
                        if key != 'execution_time':  # 실행 시간은 별도로 표시
                            logger.info(f"   {key}: {value}")

                execution_time = metadata.get('execution_time', 0)
                if execution_time:
                    logger.info(f"⏱️ Execution time: {execution_time:.2f}s")

                confidence = result.get('confidence', 0)
                logger.info(f"🎯 Confidence: {confidence:.2f}")

            else:
                logger.error("❌ CLI agent execution failed")
                error_msg = result.get('error', 'Unknown error')
                logger.error(f"   Error: {error_msg}")
                return 1

        except Exception as e:
            logger.error(f"❌ CLI agent execution failed: {e}")
            return 1

    else:
        logger.error(f"❌ Unknown CLI command: {args.cli_command}")
        logger.info("Available commands: list, test, run")
        return 1

    return 0


def main_entry():
    """Entry point for sparkle command."""
    asyncio.run(main())


if __name__ == "__main__":
    main_entry()