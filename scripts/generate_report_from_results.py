#!/usr/bin/env python3
"""
진단 결과를 바탕으로 리포트 생성
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# 실행 결과를 JSON으로 받아서 리포트 생성
if len(sys.argv) > 1:
    results_file = sys.argv[1]
    with open(results_file, 'r') as f:
        results = json.load(f)
else:
    # 기본 결과 (실제 실행 결과 기반)
    results = {
        "fetch": {
            "server_name": "fetch",
            "type": "stdio",
            "success": True,
            "stages": {
                "env_substitution": {"success": True},
                "api_key_check": {"success": True, "key_length": 36},
                "params_creation": {"success": True},
                "stdio_client": {"success": True, "time": 0.01},
                "session_init": {"success": True, "time": 2.42},
                "list_tools": {"success": True, "time": 1.03, "tools_count": 3, "tools": ["fetch_url", "extract_elements", "get_page_metadata"]}
            }
        },
        "docfork": {
            "server_name": "docfork",
            "type": "stdio",
            "success": True,
            "stages": {
                "env_substitution": {"success": True},
                "api_key_check": {"success": True, "key_length": 36},
                "params_creation": {"success": True},
                "stdio_client": {"success": True, "time": 0.00},
                "session_init": {"success": True, "time": 2.51},
                "list_tools": {"success": True, "time": 0.92, "tools_count": 2, "tools": ["docfork_search_docs", "docfork_read_url"]}
            }
        },
        "context7-mcp": {
            "server_name": "context7-mcp",
            "type": "stdio",
            "success": False,
            "error": "list_tools timeout",
            "stages": {
                "env_substitution": {"success": True},
                "api_key_check": {"success": True, "key_length": 36},
                "params_creation": {"success": True},
                "stdio_client": {"success": True, "time": 0.00},
                "session_init": {"success": True, "time": 3.52},
                "list_tools": {"success": False, "error": "timeout"}
            }
        },
        "parallel-search": {
            "server_name": "parallel-search",
            "type": "stdio",
            "success": False,
            "error": "unhandled errors in a TaskGroup (1 sub-exception)",
            "stages": {
                "env_substitution": {"success": True},
                "api_key_check": {"success": True, "key_length": 36},
                "params_creation": {"success": True},
                "stdio_client": {"success": True, "time": 0.00},
                "session_init": {"success": False, "error": "HTTP 401: invalid_token"}
            }
        },
        "tavily-mcp": {
            "server_name": "tavily-mcp",
            "type": "stdio",
            "success": False,
            "error": "unhandled errors in a TaskGroup (1 sub-exception)",
            "stages": {
                "env_substitution": {"success": True},
                "api_key_check": {"success": True, "key_length": 36},
                "params_creation": {"success": True},
                "stdio_client": {"success": True, "time": 0.01},
                "session_init": {"success": False, "error": "Failed to get user config: Config get request failed with status 500"}
            }
        },
        "WebSearch-MCP": {
            "server_name": "WebSearch-MCP",
            "type": "stdio",
            "success": False,
            "error": "unhandled errors in a TaskGroup (1 sub-exception)",
            "stages": {
                "env_substitution": {"success": True},
                "api_key_check": {"success": True, "key_length": 36},
                "params_creation": {"success": True},
                "stdio_client": {"success": True, "time": 0.01},
                "session_init": {"success": False, "error": "Failed to get user config: Config get request failed with status 500"}
            }
        },
        "semantic_scholar": {
            "server_name": "semantic_scholar",
            "type": "http",
            "success": False,
            "error": "unhandled errors in a TaskGroup (1 sub-exception)",
            "stages": {
                "url_check": {"success": True},
                "params": {"success": True},
                "http_client": {"success": False, "error": "unhandled errors in a TaskGroup"}
            }
        }
    }

project_root = Path(__file__).parent.parent
reports_dir = project_root / "reports"
reports_dir.mkdir(exist_ok=True)

successful = []
failed = []

for server_name, result in results.items():
    if result.get("success"):
        successful.append((server_name, result))
    else:
        failed.append((server_name, result))

report = f"""# Smithery MCP 서버 진단 리포트

**생성 일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 요약

- **전체 서버 수**: {len(results)}
- **✅ 성공**: {len(successful)} ({len(successful)/len(results)*100:.1f}%)
- **❌ 실패**: {len(failed)} ({len(failed)/len(results)*100:.1f}%)

## ✅ 성공한 서버

"""

for server_name, result in successful:
    stages = result.get("stages", {})
    tools_info = stages.get("list_tools", {})
    tools_count = tools_info.get("tools_count", 0)
    tools = tools_info.get("tools", [])
    
    report += f"""### {server_name}

- **타입**: {result.get('type', 'unknown')}
- **도구 수**: {tools_count}개
- **도구 목록**: {', '.join(tools) if tools else 'N/A'}

**단계별 성공 여부**:
"""
    for stage_name, stage_result in stages.items():
        if isinstance(stage_result, dict):
            status = "✅ 성공" if stage_result.get("success") else "❌ 실패"
            time_info = f" ({stage_result.get('time', 0):.2f}s)" if stage_result.get("time") else ""
            report += f"- {stage_name}: {status}{time_info}\n"
    
    report += "\n"

if failed:
    report += """## ❌ 실패한 서버

"""
    for server_name, result in failed:
        report += f"""### {server_name}

- **타입**: {result.get('type', 'unknown')}
- **최종 에러**: `{result.get('error', 'Unknown')}`

**단계별 분석**:
"""
        stages = result.get("stages", {})
        for stage_name, stage_result in stages.items():
            if isinstance(stage_result, dict):
                status = "✅ 성공" if stage_result.get("success") else "❌ 실패"
                error_info = f" - {stage_result.get('error', '')}" if not stage_result.get("success") and stage_result.get("error") else ""
                time_info = f" ({stage_result.get('time', 0):.2f}s)" if stage_result.get("time") else ""
                report += f"- {stage_name}: {status}{time_info}{error_info}\n"
        
        report += "\n"

# 문제점 분석
report += """## 🔍 문제점 분석

"""

# 500 에러
error_500_servers = [name for name, r in failed if "500" in str(r.get("error", "")) or "Failed to get user config" in str(r.get("error", ""))]
if error_500_servers:
    report += f"""### 1. Smithery 서버 500 에러 (Bundle 설정 조회 실패)

**영향 서버**: {', '.join(error_500_servers)}

**증상**: Bundle 다운로드는 성공했지만, 사용자 설정 조회 단계에서 Smithery 서버가 500 에러를 반환합니다.

**원인**: Smithery 서버 측 내부 오류로 인한 설정 조회 실패

**해결 방안**:
- Smithery 서버 상태 확인
- 일시적 장애일 가능성이 있으므로 재시도 권장
- Bundle 기반 서버의 경우 직접 실행 방식으로 전환 고려

"""

# 401 에러
error_401_servers = [name for name, r in failed if "401" in str(r.get("error", "")) or "invalid_token" in str(r.get("error", ""))]
if error_401_servers:
    report += f"""### 2. HTTP 401 인증 실패

**영향 서버**: {', '.join(error_401_servers)}

**증상**: 연결은 성공했으나 세션 초기화 또는 heartbeat 단계에서 401 에러 발생

**원인**: 
- 세션 유지 중 토큰 검증 실패
- 서버 측 세션 관리 문제 가능성

**해결 방안**:
- API 키 재확인
- 세션 재연결 로직 강화
- Heartbeat 실패 시 자동 재연결

"""

# 타임아웃
timeout_servers = [name for name, r in failed if "timeout" in str(r.get("error", "")).lower()]
if timeout_servers:
    report += f"""### 3. 타임아웃 에러

**영향 서버**: {', '.join(timeout_servers)}

**증상**: 도구 목록 조회 시 타임아웃 발생

**원인**: 서버 응답 지연 또는 네트워크 문제

**해결 방안**:
- 타임아웃 시간 증가 (현재 15초 → 30초 이상 권장)
- 재시도 로직 추가

"""

# 권장 사항
report += """## 💡 권장 사항

1. **즉시 조치**
   - Smithery 서버 상태 확인
   - 실패한 서버는 일시적으로 비활성화 고려
   - 성공한 서버(`fetch`, `docfork`) 우선 사용

2. **단기 조치**
   - 재시도 로직 강화 (500/520 에러 시 자동 재시도)
   - 타임아웃 조정 (Bundle 다운로드 및 설정 조회 타임아웃 증가)
   - Heartbeat 실패 시 자동 재연결

3. **중기 조치**
   - 서버 상태 모니터링 구현
   - 실패한 서버 자동 비활성화
   - 성공한 서버 우선 사용 로직 구현

## 📝 상세 결과 (JSON)

<details>
<summary>전체 진단 결과 JSON 보기</summary>

```json
""" + json.dumps(results, indent=2, ensure_ascii=False, default=str) + """
```

</details>

---
*이 리포트는 자동으로 생성되었습니다.*
"""

report_file = reports_dir / f"smithery_mcp_diagnosis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
with open(report_file, 'w', encoding='utf-8') as f:
    f.write(report)

print(f"📄 리포트 생성 완료: {report_file}")
print(f"   성공: {len(successful)}/{len(results)}, 실패: {len(failed)}/{len(results)}")

