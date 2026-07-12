# Anvil Plan — SparkleForge의 에이전틱 OS 로드맵

> 이 문서는 `README.md`가 오래전부터 가리키고 있었지만 실제로는 존재한 적이
> 없던 로드맵 문서다. Anvil 관련 기획은 지금까지 전부 커밋 메시지
> (`git log --grep anvil`)에만 흩어져 있었고, 어디에도 정리된 적이 없었다.
> 이 문서는 그 기획을 복구하고, 앞으로는 여기서 이어간다.

## 1. Anvil이란 무엇인가

Anvil은 "리눅스 같은 진짜 OS"가 아니라 **에이전트를 위한 OS 형태의 실행
계층**이다. SparkleForge라는 리서치 프로덕트 위에, 에이전트가 안전하게
관찰(observe)하고, 행동(actuate)하고, 통제받고(guard), 사람과
교신하는(surface) 커널급 규율을 부여하는 것이 목표다.

전통적인 OS 개념과 Anvil 모듈의 대응 관계:

| OS 개념 | Anvil 대응 | 위치 |
|---|---|---|
| 커널 관측/텔레메트리 (`/proc`, syslog) | Observe plane | `src/core/observe/` (`event_bus.py`, `system_collector.py`, `window_tracker.py`, `snapshot_api.py`) |
| 시스템 콜 / 실행 계층 (셸, 패키지 매니저, 파일시스템) | Actuate plane | `src/core/actuate/` (`os_control.py`, `shell_executor.py`, `package_manager.py`, `semantic_fs.py`, `iot_device.py`) |
| 권한/케이퍼빌리티, 샌드박스, 감사 로그 | Guard plane | `src/core/guard/` (`capability_manager.py`, `sandbox_executor.py`, `action_journal.py`, `anomaly_detector.py`) |
| 사용자 셸 / 대시보드 | Surface plane | `src/core/surface/` (`nl_shell.py`, `task_dashboard.py`, `explainability.py`) |
| 프로세스 스케줄러 | Orchestrator (LangGraph 상태 머신) | `src/core/orchestrator/` (`graph.py`, `execution.py`, `planning.py`) |
| 실행 모드 전환 (커널/사용자 모드에 대응) | `ModeController` (autonomous ↔ HITL) | `src/core/anvil/mode_controller.py` |
| 동적 디스패치 / 방법 탐색 | `MethodResolver` | `src/core/anvil/method_resolver.py` |
| 인텐트 검증 가드레일 | `IntentGuardrail` | `src/core/anvil/intent_guardrail.py` |
| 패키지 매니저 (스킬 설치/증류) | `SkillRepository` / `SkillDistiller` | `src/core/anvil/skill_repository.py`, `skill_distillation.py` |
| 인터럽트 / 사람 개입 지점 | `HITLCheckpointManager` | `src/core/anvil/hitl_checkpoint.py` |
| 멀티노드 연합, 원격 세션 | Federation / Session | `src/core/federation/protocol.py`, `src/core/session/` |
| 상시 구동 데몬 (자가 치유) | Nightwelding | `src/core/nightwelding/` — 이슈 재현 → 실패 테스트 작성 → 수정 → Draft PR (머지는 절대 안 함) |
| 플러그인/드라이버 탐색 | Plugin system | `src/core/plugin_system/` (`discovery.py`, `manifest.py`, `hooks.py`) |

## 2. Phase 히스토리 (실제 커밋 기준 복원)

| Phase | 주제 | 상태 | 주요 산출물 |
|---|---|---|---|
| M1 | FastMCP 지연 로딩 | ✅ | `src/core/mcp_integration/` 단계적 분리 |
| M2 | Anvil 워크플로우 코어 | ✅ | `src/core/anvil/engine.py`, `skill_repository.py` |
| M3 | 요청 분석 & 동적 체크리스트 | ✅ | `request_analyzer.py`, `dynamic_checklist_generator.py` |
| M4 | HITL 의도 검증 가드레일 | ✅ | `intent_guardrail.py`, `hitl_checkpoint.py` |
| M5 | 범용 문제 해결 / 동적 방법 탐색 | ✅ | `method_resolver.py`, `mode_controller.py` |
| A | 하드닝 | ✅ | 안정성 기반 작업 |
| B | Observe | ✅ | `src/core/observe/` (event bus, 텔레메트리) |
| C | Actuate — 실행 평면 | ✅ | `src/core/actuate/` |
| D | 자동화 진화 | ✅ | `src/core/automation/automation_engine.py` |
| E | 메모리 & 컨텍스트 | ✅ | 2-tier 상수 크기 메모리 |
| F | 라우팅 확장 | ✅ | |
| G | Guard — 보안/신뢰 | ✅ | `src/core/guard/` |
| H | Surface — 사용자 경계 | ✅ | `src/core/surface/` |
| Z | 연합 & 스케일 (Z-1 원격 세션, Z-2 coordinator mode, Z-3 IoT 어댑터) | ✅ | `src/core/federation/`, `src/core/session/` |
| Ψ | 크로스플랫폼 배포 패키징 | ✅ | |
| Ω | 증명대 & 자가개선 (Ω-1 시나리오 평가 하네스, Ω-2 auto-fix 파이프라인, Ω-4 비용/지연 인지 라우팅, Ω-5 모놀리스 분할) | ✅ | `src/core/nightwelding/`, eval harness |
| v | Agent Vivarium — 상시 운영 환경 | ✅ | |
| Σ | 구조적 무결성 & 자율성 (Σ-1 모놀리스 실제 분할, Σ-2 런타임 서브에이전트 위임, Σ-3 GuardPlane/TrustGate 하네스, Σ-4 auto-fix 검증 게이트) | ✅ | `src/core/orchestrator/` 분할, `agent_orchestrator.py` 위임 깊이 가드 — 마일스톤 #507 |
| Ξ | 프로세스 모델 — 세션별 리소스 쿼터 (Ξ-1 쿼터 스키마, Ξ-2 초과 자동 처리, Ξ-3 CLI 노출) | 🔲 계획됨 | 마일스톤 #542 (#543, #544, #545) |

**현재 위치**: Σ는 2026-07-12에 닫혔다(#507). 다만 Σ-1이 만든 모놀리스
분할 자체에 circular import / missing return / signature mismatch 결함이
남아 있었고(#539), 이걸 실제로 고치는 작업이 `fix/mcp-integration-tools-monolith-split`
브랜치에서 진행 중이다 — Σ가 이름 그대로 안정된 상태가 되려면 이 브랜치가
먼저 머지돼야 한다. Σ 이후로는 `planning: anvil phase X` 형식의 커밋이
끊기고, CLAUDE.md의 auto-fix 에이전트(Nightwelding)가 올리는 리액티브
버그 픽스만 이어졌다 — **의도적인 다음 phase 설계가 비어 있는 상태**였다는
것이 이 문서를 쓰게 된 계기다. 그 다음 phase로 §4의 후보 중 "프로세스
모델"을 Phase Ξ로 확정했다 (마일스톤 #542).

## 3. 알려진 결함 (이번에 확인됨)

- `README.md`가 가리키던 `docs/ANVIL_PLAN.md`는 git 히스토리 전체에서
  커밋된 적이 없었다 — 이 문서로 해소.
- `docs/ARCHITECTURE.md`는 SparkleForge/Anvil 리네임 이전 "Local Researcher"
  시절 문서가 그대로 남아 있어 현재 아키텍처와 무관했다 — 별도로 재작성.

## 4. 다음 phase 후보 (미결정 — 논의 대상)

Σ 다음 단계로 검토했던 방향들. "프로세스 모델"은 Phase Ξ로 확정되어 §2
표로 이동했다(마일스톤 #542) — 조회·제어(`SessionControl`, `session.py`
CLI)는 이미 있었고 세션별 리소스 쿼터만 비어 있어 그 부분만 스코프로
잡았다. 나머지는 아직 phase로 확정하지 않았고, 우선순위를 논의한 뒤 이
섹션을 갱신한다.

- **VFS 통합**: `storage/`, `output/`, `temp/`가 산발적으로 존재 —
  `semantic_fs.py`를 에이전트/스킬 공용 주소 공간으로 승격.
- **syscall 경계 공식화**: Σ-2에서 고친 위임 버그(#516), quarantine 결함
  (#519), credential delegation 결함(#312)이 공통적으로 가리키는 방향 —
  에이전트 간 호출을 `IntentGuardrail`을 반드시 통과하는 공식 API로 정리.
  이슈화할 때는 이미 closed된 과거 결함의 구체적 메커니즘을 다시 상세히
  서술하지 말고, 재발 방지라는 목적과 설계 방향 위주로 작성할 것.
- **스킬 마켓플레이스**: `SkillRepository`/`SkillDistiller`는 있지만 인스턴스
  간 스킬 공유/임포트-익스포트 개념은 없음.
- **Surface 통합**: `nl_shell.py`, `task_dashboard.py`, CLI, 웹 UI가 하나의
  일관된 "프로세스 가시성" 표면으로 통합되어 있지 않음.
