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
| M4 | HITL 의도 검증 가드레일 | ✅ | `intent_guardrail.py`, `hitl_checkpoint.py` — `verify_plan`(`src/core/orchestrator/verification.py`)의 AFTER_PLANNING 체크포인트에서 실배선(대화형 세션에서만 동작, `autopilot_mode`/비-TTY에서는 기존과 동일하게 자동 승인) |
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
| Ξ | 프로세스 모델 — 세션별 리소스 쿼터 (Ξ-1 쿼터 스키마, Ξ-2 초과 자동 처리, Ξ-3 CLI 노출) | ✅ | `SessionQuota`, `SessionControl.check_quotas`, `session quota` CLI — 마일스톤 #542 (#543, #544, #545) |
| Π | VFS 통합 — `storage/`/`output`/`temp` 공용 주소 공간 | 🔲 계획됨 | 마일스톤 #567 |
| Τ | syscall 경계 공식화 — 에이전트 간 호출 단일 진입점 | 🔲 계획됨 | 마일스톤 #568 |
| Λ | 스킬 마켓플레이스 — 인스턴스 간 스킬 공유 | 🔲 계획됨 | 마일스톤 #569 |
| Φ | Surface 통합 — CLI/웹 UI/자연어 셸 단일화 | 🔲 계획됨 | 마일스톤 #570 |

**현재 위치**: Σ는 2026-07-12에 닫혔고, Σ-1이 남긴 모놀리스 분할 잔여
결함(circular import / missing return / signature mismatch, #539)도
2026-07-13에 실제로 고쳐져 머지됐다(#565) — Σ가 이름 그대로 안정된
상태가 됐다. Ξ도 2026-07-13에 세 서브페이즈(#543/#544/#545) 전부 닫혀
완료됐다. Σ 이후 한동안 `planning: anvil phase X` 형식의 커밋이 끊기고
CLAUDE.md의 auto-fix 에이전트(Nightwelding)가 올리는 리액티브 버그
픽스만 이어졌던 공백은, §4에 미결정으로 남아있던 네 후보(VFS 통합/syscall
경계 공식화/스킬 마켓플레이스/Surface 통합)를 전부 Phase Π/Τ/Λ/Φ로
확정하며(마일스톤 #567~#570) 다시 채워졌다.

## 3. 알려진 결함 (이번에 확인됨)

- `README.md`가 가리키던 `docs/ANVIL_PLAN.md`는 git 히스토리 전체에서
  커밋된 적이 없었다 — 이 문서로 해소.
- `docs/ARCHITECTURE.md`는 SparkleForge/Anvil 리네임 이전 "Local Researcher"
  시절 문서가 그대로 남아 있어 현재 아키텍처와 무관했다 — 별도로 재작성.
- (2026-07-24, 이슈 #910) `VerificationNode.__init__`이 `self._mode_controller`를
  한 번도 초기화하지 않아, 사람이 붙은 HITL 체크포인트(APPROVE/REVISE/ABORT)
  경로가 실제로는 `AttributeError`로 매번 죽고 있었다 — `verify_plan`의
  광범위한 `except Exception` 재시도 루프가 이를 삼켜 "Verification failed
  after 3 attempts"라는 무관한 에러로만 드러났다. 이 경로를 커버하는
  `tests/test_verify_plan_hitl.py`가 CI `pytest` 잡에 아예 포함되어 있지
  않아서 아무도 못 봤다. `src/core/orchestrator/verification.py`에서 수정,
  해당 테스트 및 다른 plane 테스트들(`test_guard_plane.py`,
  `test_security_tools.py`, `test_iot_adapter.py`, `test_coordinator.py`,
  `test_session_*.py`)을 `pr-merge-gate.yml`의 `pytest` 잡에 추가.
- (2026-07-24, 이슈 #910) `AnomalyDetector.observe()`는 `GuardPlane.
  check_and_execute()`를 통해서만 호출되는데, 그 유일한 실제 호출자인
  `WorkerNode.handle_execute()`는 `tests/test_coordinator.py` 밖에서 한 번도
  인스턴스화되지 않는다 — 이 저장소가 실제로 배포하는 단일 노드 실행
  경로(CLI / `AgentHarness` / `AgentLoop`)에서는 `AnomalyDetector`가 전혀
  호출되지 않는다는 뜻. #715류 결함이지만 아직 고치지 않았고,
  `tests/test_os_plane_integrity.py::test_anomaly_detector_has_no_reachable_single_node_entrypoint`가
  이 상태를 명시적으로 고정해 앞으로의 변화(수정이든 방치든)를 추적한다.

이 §3 목록이 앞으로도 "수동 감사가 우연히 발견"에 의존하지 않도록,
`tests/test_os_plane_integrity.py`가 #715류 결함(컴포넌트는 있는데 실제
프로덕션 경로에서 한 번도 안 불림)을 CI에서 상시 검증한다 — README.md의
"Anvil: The Agentic OS Layer" 절 참고.

## 4. 다음 phase 후보 (모두 확정 — §2 참고)

Σ 다음 단계로 검토했던 방향들. 전부 phase로 확정되어 §2 표로 이동했다:
"프로세스 모델"은 Phase Ξ(마일스톤 #542, 완료), 나머지 네 방향은
2026-07-13에 한 번에 Phase Π/Τ/Λ/Φ로 확정됐다(마일스톤 #567~#570).
아래는 확정 당시의 스코프 기록이며, 각 마일스톤 이슈 본문에 더 구체적인
작업 항목·성공 기준이 있다.

- **VFS 통합** (Π, #567): `storage/`, `output/`, `temp/`가 산발적으로
  존재 — `semantic_fs.py`를 에이전트/스킬 공용 주소 공간으로 승격.
- **syscall 경계 공식화** (Τ, #568): Σ-2에서 고친 위임 버그(#516),
  quarantine 결함(#519), credential delegation 결함(#312)이 공통적으로
  가리키는 방향 — 에이전트 간 호출을 `IntentGuardrail`을 반드시 통과하는
  공식 API로 정리. 과거 closed 결함의 메커니즘 재서술이 아니라 재발
  방지 설계가 목적.
- **스킬 마켓플레이스** (Λ, #569): `SkillRepository`/`SkillDistiller`는
  있지만 인스턴스 간 스킬 공유/임포트-익스포트 개념은 없음.
- **Surface 통합** (Φ, #570): `nl_shell.py`, `task_dashboard.py`, CLI,
  웹 UI가 하나의 일관된 "프로세스 가시성" 표면으로 통합되어 있지 않음.

다음 §4 후보를 새로 발굴할 때까지, 실제 구현 착수는 §2의 우선순위 논의
결과에 따른다 (이 문서는 아직 Π/Τ/Λ/Φ 간 우선순위를 정하지 않았다).
