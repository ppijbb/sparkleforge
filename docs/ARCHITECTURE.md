# SparkleForge / Anvil 아키텍처

## 개요

SparkleForge는 멀티에이전트 리서치 프로덕트이고, **Anvil**은 그 위에 얹힌
에이전틱 OS 실행 계층이다. 두 개를 나눠서 읽는다:

- **SparkleForge**: 리서치 오케스트레이션 (검색 → 종합 → 검증 → 리포트).
  `src/core/orchestrator/`의 LangGraph 상태 머신이 프로세스 스케줄러 역할.
- **Anvil**: 에이전트가 관찰하고, 행동하고, 통제받고, 사람과 교신하는 규율을
  부여하는 하부 계층. "리눅스 같은 진짜 OS"가 아니라 OS 형태로 설계된
  실행/권한/관측 프레임워크다.

Anvil의 phase별 이력과 다음 단계 후보는 `docs/ANVIL_PLAN.md`를 참고. 이
문서는 phase 히스토리가 아니라 **현재 시점의 구조**를 설명한다.

## 시스템 구조

```
┌──────────────────────────────────────────────────────────────────┐
│  Surface plane — 사람과의 접점                                     │
│  nl_shell.py │ task_dashboard.py │ notification_channel.py        │
│  │ explainability.py │  (src/core/surface/)                      │
├──────────────────────────────────────────────────────────────────┤
│  Guard plane — 권한/안전                                          │
│  capability_manager.py │ sandbox_executor.py │ hitl_gate.py       │
│  │ credential_vault.py │ anomaly_detector.py │ action_journal.py  │
│  (src/core/guard/)                                                │
├──────────────────────────────────────────────────────────────────┤
│  Anvil core — 워크플로우 엔진 & 실행 모드                           │
│  AnvilWorkflowEngine │ RequestAnalyzer │ DynamicChecklistGenerator │
│  MethodResolver │ ModeController │ IntentGuardrail                │
│  HITLCheckpointManager │ SkillRepository │ SkillDistiller         │
│  (src/core/anvil/)                                                │
├──────────────────────────────────────────────────────────────────┤
│  Orchestrator — 리서치 프로세스 스케줄러 (LangGraph)                │
│  analysis → planning → execution → verification → synthesis      │
│  → compression  (src/core/orchestrator/)                          │
├──────────────────────────────────────────────────────────────────┤
│  Actuate plane — 실행/시스템콜                                     │
│  os_control.py │ shell_executor.py │ package_manager.py           │
│  │ semantic_fs.py │ iot_device.py │ firmware_assistant.py         │
│  (src/core/actuate/)                                              │
├──────────────────────────────────────────────────────────────────┤
│  Observe plane — 관측/텔레메트리                                  │
│  event_bus.py │ system_collector.py │ window_tracker.py           │
│  │ snapshot_api.py │ iot_telemetry_loop.py │ package_inventory.py │
│  (src/core/observe/)                                              │
├──────────────────────────────────────────────────────────────────┤
│  Federation / Session — 멀티노드, 원격 세션                        │
│  protocol.py (federation) │ coordinator.py, remote_session.py,    │
│  secure_envelope.py (session)                                     │
├──────────────────────────────────────────────────────────────────┤
│  Nightwelding — 상시 구동 자가 치유 데몬                            │
│  이슈 재현 → 실패 테스트 작성 → 수정 → Draft PR만 생성 (머지 안 함)  │
│  (src/core/nightwelding/, CLAUDE.md의 auto-fix 에이전트 실체)      │
└──────────────────────────────────────────────────────────────────┘
```

## 핵심 구성 요소

### Anvil core (`src/core/anvil/`)

- `AnvilWorkflowEngine` — 동적 DAG 태스크 스케줄링.
- `RequestAnalyzer` / `DynamicChecklistGenerator` — 요청을 분석해 실행 가능한
  체크리스트로 분해.
- `MethodResolver` — 특정 능력(capability)을 수행할 방법을 여러 전략으로
  탐색, 실패 시 `UNRESOLVED`로 보고.
- `ModeController` — `AUTONOMOUS` ↔ `HITL_COLLABORATIVE` 두 실행 모드를
  연속 실패/성공, 인텐트 이탈, 체크포인트 결정 신호에 따라 동적 전환.
- `IntentGuardrail` — 사용자 의도에서 벗어난 실행을 감지해 사람 확인 요청.
- `HITLCheckpointManager` — 단계별 사람 개입 지점(approve/revise/abort).
- `SkillRepository` / `SkillDistiller` — 실행 트레이스에서 재사용 가능한
  스킬을 증류해 저장 (패키지 매니저에 대응).

### Orchestrator (`src/core/orchestrator/`)

리서치 파이프라인 자체의 LangGraph 상태 머신. `analysis.py → planning.py →
execution.py → verification.py → synthesis.py → compression.py` 순서로
그래프가 구성되며, `incremental_executor.py`가 24x7 연속 리서치 모드에서
증분 실행을 담당한다. `agent_orchestrator.py`는 서브에이전트 위임
(delegation)을 다루며, 위임 깊이 가드가 중첩 위임 상황에서도 정확히
동작하도록 유지하는 것이 최근 안정화의 핵심이었다 (Σ-2 관련).

### Guard plane (`src/core/guard/`)

- `capability_manager.py` — 툴/에이전트 단위의 케이퍼빌리티 기반 권한 부여·회수.
- `sandbox_executor.py` — 코드 실행을 gVisor(`runsc`) 컨테이너로 격리, 네트워크/
  메모리/CPU/PID/권한/파일시스템 제한.
- `credential_vault.py`, `action_journal.py`, `anomaly_detector.py` — 자격
  증명 보관, 행동 감사 로그, 이상 탐지.

### Actuate / Observe planes

- Actuate는 셸 실행, OS 제어, 패키지 매니저, "semantic 파일시스템"
  (`semantic_fs.py`, watchdog 기반 파일 변경 감시), IoT 디바이스 제어를 포함.
- Observe는 이벤트 버스, 시스템 리소스 수집, 윈도우 추적, IoT 텔레메트리
  루프, 패키지 인벤토리 스냅샷을 담당 — Actuate가 무엇을 했는지 되짚어볼 수
  있는 근거를 제공한다.

### Federation / Session (`src/core/federation/`, `src/core/session/`)

멀티노드 연합(`protocol.py`), 원격 세션(`remote_session.py`), 세션
조정(`coordinator.py`), 세션 간 안전한 데이터 교환(`secure_envelope.py`)을
다룬다 (Anvil Phase Z 산출물).

### Nightwelding (`src/core/nightwelding/`)

GitHub 이슈를 받아 문제를 재현하는 실패 테스트를 먼저 작성하고, 그 테스트가
통과할 때까지 수정한 뒤 Draft PR만 연다 — 절대 머지하지 않는다. 이것이
`CLAUDE.md`에 정의된 "SparkleForge auto-fix agent"의 실제 구현체이며,
GitHub Actions에서 push/issue 이벤트에 반응해 실행된다
(`sparkleforge nightwelding run` / `run_nightwelding_sweep`).

## 리서치 데이터 흐름 (SparkleForge 본연의 기능)

```
User Input → Anvil RequestAnalyzer → DynamicChecklistGenerator
     ↓
Orchestrator (LangGraph: analysis → planning → execution → verification)
     ↓
research/tools (웹 검색, 학술 검색) → research/processors (콘텐츠 처리)
     ↓
synthesis → compression (2-tier 상수 크기 메모리) → 리포트 출력
```

## 보안 모델

- **격리**: 코드 실행은 gVisor 컨테이너 샌드박스에서 수행 (`src/core/guard/sandbox_executor.py`).
- **권한**: 케이퍼빌리티 기반 부여/회수 (`capability_manager.py`), 인텐트
  가드레일이 이탈 시 사람 확인을 강제.
- **감사**: 모든 행동이 `action_journal.py`에 기록되고 `anomaly_detector.py`가
  이상 패턴을 감시.
- **자동화 경계**: Nightwelding은 Draft PR만 생성하고 머지 권한이 없다 —
  최종 병합은 항상 사람이 결정한다 (`CLAUDE.md` 원칙).

## 다음 단계

Phase 히스토리, 완료된 마일스톤, 그리고 다음 phase 후보 논의는
`docs/ANVIL_PLAN.md`에서 관리한다. 이 문서(ARCHITECTURE.md)는 구조가 바뀔
때마다 갱신하고, phase 진행 상황은 ANVIL_PLAN.md에 남긴다.
