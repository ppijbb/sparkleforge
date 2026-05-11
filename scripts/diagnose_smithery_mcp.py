#!/usr/bin/env python3
"""Smithery MCP 서버 상세 진단 스크립트

각 서버의 연결 과정을 단계별로 분석하여 문제점을 정확히 파악합니다.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# MCP imports
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamablehttp_client
    from mcp.shared.exceptions import McpError

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logger.error("MCP package not available. Install with: pip install mcp")
    sys.exit(1)


class SmitheryDiagnostic:
    """Smithery MCP 서버 상세 진단"""

    def __init__(self):
        self.smithery_servers = {
            "fetch": {
                "command": "npx",
                "args": [
                    "-y",
                    "@smithery/cli@latest",
                    "run",
                    "@smithery-ai/fetch",
                    "--key",
                    os.getenv("SMITHERY_API_KEY", ""),
                    "--profile",
                    os.getenv("SMITHERY_PROFILE", ""),
                ],
            },
            "docfork": {
                "command": "npx",
                "args": [
                    "-y",
                    "@smithery/cli@latest",
                    "run",
                    "@docfork/mcp",
                    "--key",
                    os.getenv("SMITHERY_API_KEY", ""),
                    "--profile",
                    os.getenv("SMITHERY_PROFILE", ""),
                ],
            },
            "context7-mcp": {
                "command": "npx",
                "args": [
                    "-y",
                    "@smithery/cli@latest",
                    "run",
                    "@upstash/context7-mcp",
                    "--key",
                    os.getenv("SMITHERY_API_KEY", ""),
                    "--profile",
                    os.getenv("SMITHERY_PROFILE", ""),
                ],
            },
            "parallel-search": {
                "command": "npx",
                "args": [
                    "-y",
                    "@smithery/cli@latest",
                    "run",
                    "@parallel/search",
                    "--key",
                    os.getenv("SMITHERY_API_KEY", ""),
                    "--profile",
                    os.getenv("SMITHERY_PROFILE", ""),
                ],
            },
            "tavily-mcp": {
                "command": "npx",
                "args": [
                    "-y",
                    "@smithery/cli@latest",
                    "run",
                    "@Jeetanshu18/tavily-mcp",
                    "--key",
                    os.getenv("SMITHERY_API_KEY", ""),
                    "--profile",
                    os.getenv("SMITHERY_PROFILE", ""),
                ],
            },
            "WebSearch-MCP": {
                "command": "npx",
                "args": [
                    "-y",
                    "@smithery/cli@latest",
                    "run",
                    "@mnhlt/WebSearch-MCP",
                    "--key",
                    os.getenv("SMITHERY_API_KEY", ""),
                    "--profile",
                    os.getenv("SMITHERY_PROFILE", ""),
                ],
            },
            "semantic_scholar": {
                "type": "http",
                "httpUrl": "https://server.smithery.ai/@hamid-vakilzadeh/mcpsemanticscholar/mcp",
                "params": {
                    "api_key": os.getenv("SMITHERY_API_KEY", ""),
                    "profile": os.getenv("SMITHERY_PROFILE", ""),
                },
            },
        }

    def check_environment(self):
        """환경 변수 확인"""
        logger.info("=" * 80)
        logger.info("환경 변수 확인")
        logger.info("=" * 80)

        api_key = os.getenv("SMITHERY_API_KEY", "")
        profile = os.getenv("SMITHERY_PROFILE", "")

        if bool(api_key):
            logger.info(
                f"✅ SMITHERY_API_KEY: 설정됨 (길이: {len(api_key)}, 앞 10자: {api_key[:10]}...)"
            )
        else:
            logger.error("❌ SMITHERY_API_KEY: 설정되지 않음")

        if profile:
            logger.info(f"✅ SMITHERY_PROFILE: {profile}")
        else:
            logger.warning("⚠️ SMITHERY_PROFILE: 설정되지 않음 (일부 서버에 필요)")

        logger.info("")
        return bool(api_key)

    async def diagnose_stdio_server(
        self, server_name: str, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Stdio 서버 상세 진단"""
        result = {
            "server_name": server_name,
            "type": "stdio",
            "stages": {},
            "success": False,
            "error": None,
        }

        logger.info(f"\n{'=' * 80}")
        logger.info(f"진단: {server_name} (stdio)")
        logger.info(f"{'=' * 80}")

        # Stage 1: 환경 변수 치환 확인
        logger.info("[Stage 1] 환경 변수 치환 확인...")
        args = []
        for arg in config.get("args", []):
            if arg.startswith("${") and arg.endswith("}"):
                env_var = arg[2:-1]
                value = os.getenv(env_var, "")
                args.append(value)
                logger.info(f"  - {arg} -> {value[:10] if value else 'EMPTY'}...")
            else:
                args.append(arg)

        result["stages"]["env_substitution"] = {"success": True, "args": args}

        # Stage 2: API 키 확인
        logger.info("[Stage 2] API 키 확인...")
        if "--key" in args:
            key_idx = args.index("--key")
            if key_idx + 1 < len(args):
                api_key = args[key_idx + 1]
                if bool(api_key):
                    logger.info(f"  ✅ API 키 발견 (길이: {len(api_key)})")
                    result["stages"]["api_key_check"] = {
                        "success": True,
                        "key_length": len(api_key),
                    }
                else:
                    logger.error("  ❌ API 키가 비어있음")
                    result["stages"]["api_key_check"] = {
                        "success": False,
                        "error": "Empty API key",
                    }
                    result["error"] = "Empty API key"
                    return result
            else:
                logger.error("  ❌ --key 다음에 값이 없음")
                result["stages"]["api_key_check"] = {
                    "success": False,
                    "error": "Missing key value",
                }
                result["error"] = "Missing key value"
                return result

        # Stage 3: 서버 파라미터 생성
        logger.info("[Stage 3] 서버 파라미터 생성...")
        try:
            server_params = StdioServerParameters(command=config["command"], args=args)
            logger.info("  ✅ 파라미터 생성 성공")
            logger.info(f"    Command: {server_params.command}")
            logger.info(f"    Args: {server_params.args[:3]}... (총 {len(server_params.args)}개)")
            result["stages"]["params_creation"] = {"success": True}
        except Exception as e:
            logger.error(f"  ❌ 파라미터 생성 실패: {e}")
            result["stages"]["params_creation"] = {"success": False, "error": str(e)}
            result["error"] = str(e)
            return result

        # Stage 4: 연결 시도
        logger.info("[Stage 4] 연결 시도...")
        try:
            start_time = datetime.now()
            async with stdio_client(server_params) as (read, write):
                connection_time = (datetime.now() - start_time).total_seconds()
                logger.info(f"  ✅ stdio 클라이언트 생성 성공 ({connection_time:.2f}s)")
                result["stages"]["stdio_client"] = {
                    "success": True,
                    "time": connection_time,
                }

                # Stage 5: 세션 초기화
                logger.info("[Stage 5] 세션 초기화...")
                try:
                    async with ClientSession(read, write) as session:
                        init_start = datetime.now()
                        await session.initialize()
                        init_time = (datetime.now() - init_start).total_seconds()
                        logger.info(f"  ✅ 세션 초기화 성공 ({init_time:.2f}s)")
                        result["stages"]["session_init"] = {
                            "success": True,
                            "time": init_time,
                        }

                        # Stage 6: 도구 목록 조회
                        logger.info("[Stage 6] 도구 목록 조회...")
                        try:
                            tools_start = datetime.now()
                            tools_result = await asyncio.wait_for(
                                session.list_tools(), timeout=15.0
                            )
                            tools_time = (datetime.now() - tools_start).total_seconds()

                            tools = tools_result.tools if hasattr(tools_result, "tools") else []
                            logger.info(
                                f"  ✅ 도구 목록 조회 성공 ({tools_time:.2f}s, {len(tools)}개 도구)"
                            )
                            logger.info(f"    도구: {[t.name for t in tools]}")
                            result["stages"]["list_tools"] = {
                                "success": True,
                                "time": tools_time,
                                "tools_count": len(tools),
                                "tools": [t.name for t in tools],
                            }
                            result["success"] = True
                        except TimeoutError:
                            logger.error("  ❌ 도구 목록 조회 타임아웃 (15초)")
                            result["stages"]["list_tools"] = {
                                "success": False,
                                "error": "timeout",
                            }
                            result["error"] = "list_tools timeout"
                        except Exception as e:
                            logger.error(f"  ❌ 도구 목록 조회 실패: {e}")
                            result["stages"]["list_tools"] = {
                                "success": False,
                                "error": str(e),
                            }
                            result["error"] = str(e)
                except Exception as e:
                    logger.error(f"  ❌ 세션 초기화 실패: {e}")
                    result["stages"]["session_init"] = {
                        "success": False,
                        "error": str(e),
                    }
                    result["error"] = str(e)
        except Exception as e:
            logger.error(f"  ❌ stdio 클라이언트 생성 실패: {e}")
            result["stages"]["stdio_client"] = {"success": False, "error": str(e)}
            result["error"] = str(e)

        return result

    async def diagnose_http_server(
        self, server_name: str, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """HTTP 서버 상세 진단"""
        result = {
            "server_name": server_name,
            "type": "http",
            "stages": {},
            "success": False,
            "error": None,
        }

        logger.info(f"\n{'=' * 80}")
        logger.info(f"진단: {server_name} (http)")
        logger.info(f"{'=' * 80}")

        # Stage 1: URL 확인
        logger.info("[Stage 1] URL 확인...")
        http_url = config.get("httpUrl") or config.get("url")
        if not http_url:
            logger.error("  ❌ URL이 없음")
            result["stages"]["url_check"] = {"success": False, "error": "No URL"}
            result["error"] = "No URL"
            return result
        logger.info(f"  ✅ URL: {http_url}")
        result["stages"]["url_check"] = {"success": True, "url": http_url}

        # Stage 2: 파라미터 구성
        logger.info("[Stage 2] 파라미터 구성...")
        params = config.get("params", {})
        if params:
            from urllib.parse import urlencode

            url_params = {}
            for key, value in params.items():
                if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                    env_var = value[2:-1]
                    env_value = os.getenv(env_var, "")
                    url_params[key] = env_value
                    logger.info(
                        f"  - {key}: {env_var} -> {env_value[:10] if env_value else 'EMPTY'}..."
                    )
                else:
                    url_params[key] = value
                    logger.info(f"  - {key}: {value}")

            if url_params:
                final_url = f"{http_url}?{urlencode(url_params)}"
                logger.info(f"  ✅ 최종 URL: {final_url[:100]}...")
                result["stages"]["params"] = {
                    "success": True,
                    "params": url_params,
                    "final_url": final_url,
                }
            else:
                final_url = http_url
                result["stages"]["params"] = {
                    "success": True,
                    "params": {},
                    "final_url": final_url,
                }
        else:
            final_url = http_url
            result["stages"]["params"] = {
                "success": True,
                "params": {},
                "final_url": final_url,
            }

        # Stage 3: HTTP 연결
        logger.info("[Stage 3] HTTP 연결 시도...")
        try:
            start_time = datetime.now()
            async with streamablehttp_client(final_url) as (read, write):
                connection_time = (datetime.now() - start_time).total_seconds()
                logger.info(f"  ✅ HTTP 클라이언트 생성 성공 ({connection_time:.2f}s)")
                result["stages"]["http_client"] = {
                    "success": True,
                    "time": connection_time,
                }

                # Stage 4: 세션 초기화
                logger.info("[Stage 4] 세션 초기화...")
                try:
                    async with ClientSession(read, write) as session:
                        init_start = datetime.now()
                        await session.initialize()
                        init_time = (datetime.now() - init_start).total_seconds()
                        logger.info(f"  ✅ 세션 초기화 성공 ({init_time:.2f}s)")
                        result["stages"]["session_init"] = {
                            "success": True,
                            "time": init_time,
                        }

                        # Stage 5: 도구 목록 조회
                        logger.info("[Stage 5] 도구 목록 조회...")
                        try:
                            tools_start = datetime.now()
                            tools_result = await asyncio.wait_for(
                                session.list_tools(), timeout=15.0
                            )
                            tools_time = (datetime.now() - tools_start).total_seconds()

                            tools = tools_result.tools if hasattr(tools_result, "tools") else []
                            logger.info(
                                f"  ✅ 도구 목록 조회 성공 ({tools_time:.2f}s, {len(tools)}개 도구)"
                            )
                            logger.info(f"    도구: {[t.name for t in tools]}")
                            result["stages"]["list_tools"] = {
                                "success": True,
                                "time": tools_time,
                                "tools_count": len(tools),
                                "tools": [t.name for t in tools],
                            }
                            result["success"] = True
                        except TimeoutError:
                            logger.error("  ❌ 도구 목록 조회 타임아웃 (15초)")
                            result["stages"]["list_tools"] = {
                                "success": False,
                                "error": "timeout",
                            }
                            result["error"] = "list_tools timeout"
                        except Exception as e:
                            logger.error(f"  ❌ 도구 목록 조회 실패: {e}")
                            result["stages"]["list_tools"] = {
                                "success": False,
                                "error": str(e),
                            }
                            result["error"] = str(e)
                except McpError as e:
                    error_code = getattr(e.error, "code", None) if hasattr(e, "error") else None
                    logger.error(f"  ❌ 세션 초기화 MCP 에러: {e} (code: {error_code})")
                    result["stages"]["session_init"] = {
                        "success": False,
                        "error": str(e),
                        "code": error_code,
                    }
                    result["error"] = f"MCP Error: {e}"
                except Exception as e:
                    logger.error(f"  ❌ 세션 초기화 실패: {e}")
                    result["stages"]["session_init"] = {
                        "success": False,
                        "error": str(e),
                    }
                    result["error"] = str(e)
        except Exception as e:
            logger.error(f"  ❌ HTTP 클라이언트 생성 실패: {e}")
            result["stages"]["http_client"] = {"success": False, "error": str(e)}
            result["error"] = str(e)

        return result

    async def diagnose_all(self):
        """모든 서버 진단"""
        logger.info("=" * 80)
        logger.info("Smithery MCP 서버 상세 진단 시작")
        logger.info("=" * 80)

        # 환경 변수 확인
        if not self.check_environment():
            logger.error("SMITHERY_API_KEY가 설정되지 않았습니다. 진단을 중단합니다.")
            return

        results = {}

        for server_name, config in self.smithery_servers.items():
            try:
                if config.get("type") == "http":
                    result = await self.diagnose_http_server(server_name, config)
                else:
                    result = await self.diagnose_stdio_server(server_name, config)
                results[server_name] = result
            except Exception as e:
                logger.error(f"진단 중 예외 발생 ({server_name}): {e}")
                results[server_name] = {
                    "server_name": server_name,
                    "success": False,
                    "error": str(e),
                }

        # 결과 요약
        self.print_summary(results)

        return results

    def print_summary(self, results: Dict[str, Any]):
        """결과 요약 출력"""
        logger.info("\n" + "=" * 80)
        logger.info("진단 결과 요약")
        logger.info("=" * 80)

        successful = []
        failed = []

        for server_name, result in results.items():
            if result.get("success"):
                successful.append(server_name)
            else:
                failed.append(server_name)

        logger.info(f"✅ 성공: {len(successful)}/{len(results)}")
        logger.info(f"❌ 실패: {len(failed)}/{len(results)}")
        logger.info("")

        # 실패한 서버 상세 분석
        if failed:
            logger.info("❌ 실패한 서버 상세 분석:")
            for server_name in failed:
                result = results[server_name]
                logger.info(f"\n  [{server_name}]")
                logger.info(f"    최종 에러: {result.get('error', 'Unknown')}")

                stages = result.get("stages", {})
                for stage_name, stage_result in stages.items():
                    if isinstance(stage_result, dict):
                        if stage_result.get("success"):
                            logger.info(f"    ✅ {stage_name}: 성공")
                        else:
                            logger.info(
                                f"    ❌ {stage_name}: 실패 - {stage_result.get('error', 'Unknown')}"
                            )

        logger.info("\n" + "=" * 80)

    def generate_report(self, results: Dict[str, Any], output_file: str = None):
        """마크다운 리포트 생성"""
        from datetime import datetime

        if output_file is None:
            output_file = (
                f"smithery_mcp_diagnosis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            )

        successful = []
        failed = []

        for server_name, result in results.items():
            if result.get("success"):
                successful.append((server_name, result))
            else:
                failed.append((server_name, result))

        report = f"""# Smithery MCP 서버 진단 리포트

**생성 일시**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 📊 요약

- **전체 서버 수**: {len(results)}
- **✅ 성공**: {len(successful)} ({len(successful) / len(results) * 100:.1f}%)
- **❌ 실패**: {len(failed)} ({len(failed) / len(results) * 100:.1f}%)

## ✅ 성공한 서버

"""

        for server_name, result in successful:
            stages = result.get("stages", {})
            tools_info = stages.get("list_tools", {})
            tools_count = tools_info.get("tools_count", 0)
            tools = tools_info.get("tools", [])

            report += f"""### {server_name}

- **타입**: {result.get("type", "unknown")}
- **도구 수**: {tools_count}개
- **도구 목록**: {", ".join(tools) if tools else "N/A"}

**단계별 성공 여부**:
"""
            for stage_name, stage_result in stages.items():
                if isinstance(stage_result, dict):
                    status = "✅ 성공" if stage_result.get("success") else "❌ 실패"
                    time_info = (
                        f" ({stage_result.get('time', 0):.2f}s)" if stage_result.get("time") else ""
                    )
                    report += f"- {stage_name}: {status}{time_info}\n"

            report += "\n"

        if failed:
            report += """## ❌ 실패한 서버

"""
            for server_name, result in failed:
                report += f"""### {server_name}

- **타입**: {result.get("type", "unknown")}
- **최종 에러**: `{result.get("error", "Unknown")}`

**단계별 분석**:
"""
                stages = result.get("stages", {})
                for stage_name, stage_result in stages.items():
                    if isinstance(stage_result, dict):
                        status = "✅ 성공" if stage_result.get("success") else "❌ 실패"
                        error_info = (
                            f" - {stage_result.get('error', '')}"
                            if not stage_result.get("success") and stage_result.get("error")
                            else ""
                        )
                        time_info = (
                            f" ({stage_result.get('time', 0):.2f}s)"
                            if stage_result.get("time")
                            else ""
                        )
                        report += f"- {stage_name}: {status}{time_info}{error_info}\n"

                report += "\n"

        # 문제점 분석
        report += """## 🔍 문제점 분석

"""

        # 500 에러
        error_500_servers = [
            name
            for name, r in failed
            if "500" in str(r.get("error", ""))
            or "Failed to get user config" in str(r.get("error", ""))
        ]
        if error_500_servers:
            report += f"""### 1. Smithery 서버 500 에러 (Bundle 설정 조회 실패)

**영향 서버**: {", ".join(error_500_servers)}

**증상**: Bundle 다운로드는 성공했지만, 사용자 설정 조회 단계에서 Smithery 서버가 500 에러를 반환합니다.

**원인**: Smithery 서버 측 내부 오류로 인한 설정 조회 실패

**해결 방안**:
- Smithery 서버 상태 확인
- 일시적 장애일 가능성이 있으므로 재시도 권장
- Bundle 기반 서버의 경우 직접 실행 방식으로 전환 고려

"""

        # 401 에러
        error_401_servers = [
            name
            for name, r in failed
            if "401" in str(r.get("error", "")) or "invalid_token" in str(r.get("error", ""))
        ]
        if error_401_servers:
            report += f"""### 2. HTTP 401 인증 실패

**영향 서버**: {", ".join(error_401_servers)}

**증상**: 연결은 성공했으나 세션 초기화 또는 heartbeat 단계에서 401 에러 발생

**원인**: 
- 세션 유지 중 토큰 검증 실패
- 서버 측 세션 관리 문제 가능성

**해결 방안**:
- API 키 재확인
- 세션 재연결 로직 강화
- Heartbeat 실패 시 자동 재연결

"""

        # 520 에러
        error_520_servers = [name for name, r in failed if "520" in str(r.get("error", ""))]
        if error_520_servers:
            report += f"""### 3. HTTP 520 에러 (Cloudflare-Origin 서버 연결 문제)

**영향 서버**: {", ".join(error_520_servers)}

**증상**: Cloudflare는 정상이지만 Origin 서버(server.smithery.ai) 연결 실패

**원인**: Smithery origin 서버 장애 또는 과부하

**해결 방안**:
- Smithery 서버 상태 확인
- 일시적 장애일 가능성이 있으므로 재시도 권장
- 타임아웃 증가

"""

        # 권장 사항
        report += """## 💡 권장 사항

1. **즉시 조치**
   - Smithery 서버 상태 확인
   - 실패한 서버는 일시적으로 비활성화 고려

2. **단기 조치**
   - 재시도 로직 강화 (500/520 에러 시 자동 재시도)
   - 타임아웃 조정 (Bundle 다운로드 및 설정 조회 타임아웃 증가)

3. **중기 조치**
   - 서버 상태 모니터링 구현
   - 실패한 서버 자동 비활성화
   - 성공한 서버 우선 사용 로직 구현

## 📝 상세 결과 (JSON)

<details>
<summary>전체 진단 결과 JSON 보기</summary>

```json
{json.dumps(results, indent=2, ensure_ascii=False, default=str)}
```

</details>

---
*이 리포트는 자동으로 생성되었습니다.*
"""

        # 파일 저장
        report_path = Path(output_file)
        if not report_path.is_absolute():
            report_path = project_root / "reports" / report_path
            report_path.parent.mkdir(exist_ok=True)

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        logger.info(f"\n📄 리포트 저장됨: {report_path}")
        return str(report_path)


async def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description="Smithery MCP 서버 상세 진단")
    parser.add_argument("--server", type=str, help="특정 서버만 진단 (예: fetch, docfork)")

    args = parser.parse_args()

    diagnostic = SmitheryDiagnostic()

    if args.server:
        # 특정 서버만 진단
        if args.server not in diagnostic.smithery_servers:
            logger.error(f"서버 '{args.server}'를 찾을 수 없습니다")
            logger.info(f"사용 가능한 서버: {', '.join(diagnostic.smithery_servers.keys())}")
            return

        config = diagnostic.smithery_servers[args.server]
        if config.get("type") == "http":
            result = await diagnostic.diagnose_http_server(args.server, config)
        else:
            result = await diagnostic.diagnose_stdio_server(args.server, config)

        # 리포트 생성
        results_dict = {args.server: result}
        report_path = diagnostic.generate_report(results_dict)

        print("\n" + "=" * 80)
        print("진단 결과 (JSON):")
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        print(f"\n📄 상세 리포트: {report_path}")
    else:
        # 모든 서버 진단
        results = await diagnostic.diagnose_all()

        # 리포트 생성
        report_path = diagnostic.generate_report(results)

        print("\n" + "=" * 80)
        print("전체 진단 결과 (JSON):")
        print(json.dumps(results, indent=2, ensure_ascii=False, default=str))
        print(f"\n📄 상세 리포트: {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
