# SparkleForge

<p align="center">
  <img src="docs/banner.jpg" alt="SparkleForge" width="100%" />
</p>

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![FastMCP](https://img.shields.io/badge/MCP-FastMCP%203.4+-007acc.svg)](https://github.com/jlowin/fastmcp)
[![Live Telemetry](https://img.shields.io/badge/telemetry-live-2ea44f.svg)](https://sparkleforge.streamlit.app)

SparkleForge is an autonomous multi-agent research and engineering runtime built on **Anvil**, a capability-gated execution layer. It combines a LangGraph research and coworker pipeline, a multi-host Model Context Protocol (MCP) server with an embedded control board UI, an HTTP task API, and **Nightwelding**, a reproduce-first GitHub issue repair daemon tuned for low-cost and free model tiers (`gemini-3.5-flash-lite`, `gemini-3.1-flash-lite-preview`, `gpt-5-nano`, and OpenRouter `:free` models).

---

## Architecture Overview

SparkleForge separates domain orchestration from runtime governance across two layers:

```text
+----------------------------------------------------------------------+
| Surface Plane        CLI REPL, MCP App board (ui://), notifications  |
+----------------------------------------------------------------------+
| Guard Plane          Capability grants, gVisor (runsc) sandbox,      |
|                      HITL checkpoints, credential vault, audit log   |
+----------------------------------------------------------------------+
| Anvil Core           DAG workflow engine, mode controller,           |
|                      intent guardrail, skill distiller & repository  |
+----------------------------------------------------------------------+
| Research & Coworker  LangGraph state machine: analysis -> planning   |
| Orchestrator         -> execution -> verification -> synthesis       |
|                      -> two-tier memory compression                  |
+----------------------------------------------------------------------+
| Actuate & Observe    Shell/OS execution, semantic file watcher,      |
|                      event bus, resource snapshots, IoT telemetry    |
+----------------------------------------------------------------------+
| Nightwelding         Reproduce-first repair worker (failing test     |
|                      -> implementation -> draft PR, never merges)    |
+----------------------------------------------------------------------+
```

### Core Subsystems

- **Two-Tier Bounded Memory**: Long-running research sessions maintain an evolving hierarchical summary alongside a fixed-capacity recent fact log instead of appending raw transcripts indefinitely.
- **Reactive Background Scheduling**: External and asynchronous jobs suspend LLM invocations until completion events arrive, avoiding idle polling token burn.
- **Capability-Gated Execution (Anvil)**: Tool and OS actions pass through `src/core/guard/` capability checks, an append-only action journal, and human-in-the-loop (HITL) approval gates when intent or risk thresholds require review.
- **Reproduce-First Issue Repair (Nightwelding)**: Automated repair runs write a failing reproduction test before modifying target code and open draft pull requests for human review rather than merging autonomously.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for module-level boundaries and [`docs/ANVIL_PLAN.md`](docs/ANVIL_PLAN.md) for milestone history.

---

## Installation

### Prerequisites

- Python `3.11` or newer
- [`uv`](https://docs.astral.sh/uv/) package manager
- At least one LLM API key (`GOOGLE_API_KEY`, `OPENROUTER_API_KEY`, or `NVIDIA_API_KEY`)

### Quick Setup (Python Environment Only)

```bash
git clone https://github.com/ppijbb/sparkleforge.git
cd sparkleforge

uv sync
cp env.example .env
```

Edit `.env` to configure your provider and model tier:

```dotenv
LLM_PROVIDER=google
LLM_MODEL=google/gemini-3.5-flash-lite
GOOGLE_API_KEY=your_api_key_here
```

### System & Sandbox Setup (`install.sh`)

```bash
./install.sh
```

- **Linux**: Installs Docker, downloads and verifies the gVisor `runsc` runtime, registers `runsc` with Docker, and validates container isolation (`--network none`, read-only root filesystem, CPU/memory/PID limits). Run `uv sync` afterward if not already initialized.
- **macOS**: Verifies `uv`, runs `uv sync`, and creates `.env` from `env.example`. Linux-specific `runsc` installation is skipped; Docker Desktop remains optional.

---

## CLI Usage

`pyproject.toml` registers both `sparkleforge` and `sparkle` console scripts (`uv run sparkleforge` works without manually activating `.venv`).

```bash
# Start the interactive session
uv run sparkleforge

# Run a single research request
uv run sparkleforge run "Compare KV-cache quantization methods published in 2026"

# Run a repository or engineering task in coworker mode
uv run sparkleforge work "Audit timeout handling across src/core/mcp_servers"

# Resume the most recent session or a specific session ID
uv run sparkleforge run --continue "Summarize latency trade-offs in a table"
uv run sparkleforge run --session session_20261127_120000 "Export citations"

# Non-interactive headless execution with structured output
uv run sparkleforge --prompt "Survey WebAssembly component model runtimes" \
  --output-format json \
  --output report.json
```

### Runtime & Governance Commands

```bash
uv run sparkleforge health        # Check environment, provider, and runtime health
uv run sparkleforge mcp status    # Inspect configured MCP tool servers
uv run sparkleforge tools list    # List available research and system tools
uv run sparkleforge session       # Inspect saved sessions
uv run sparkleforge actions       # List pending HITL actions (approve / deny)
uv run sparkleforge nightwelding  # Run or inspect reproduce-first issue repair
```

---

## Python SDK

Embed SparkleForge directly inside an existing Python process via [`src/sdk.py`](src/sdk.py) without spawning CLI subprocesses:

```python
import asyncio
from src.sdk import get_job_status, get_report, run, submit_job


async def main() -> None:
    # Blocking in-process execution
    result = await run("Evaluate SQLite WAL contention under concurrent readers")

    # Non-blocking background job execution
    job_id = await submit_job("Compare Rust async runtimes for embedded Linux")
    status = await get_job_status(job_id)
    if status.status == "completed":
        report = await get_report(job_id)


asyncio.run(main())
```

Background jobs persist locally to `~/.sparkleforge/sdk_jobs.json` (configurable via `SPARKLEFORGE_SDK_JOB_STORE`, 30-day TTL on terminal jobs) so completed reports survive single-worker process restarts, and mirror automatically to Supabase when credentials are configured.

---

## MCP Server, Control Board UI, and IDE Plugins

While internal modules under `src/core/mcp_servers/` provide tools to SparkleForge's own agents, `src/core/mcp_servers/sparkleforge_server.py` exposes SparkleForge itself as an outward-facing MCP server (`scripts/run_mcp_server.sh`). External coding agents and IDEs can delegate background research, inspect Nightwelding queues, query local skills, and open the interactive Swarm control board.

### Install as a Plugin

Repository root manifests support Claude Code (`.claude-plugin/`), Antigravity (`plugin.json` and `mcp_config.json`), Codex (`.codex-plugin/plugin.json`), and Gemini CLI (`gemini-extension.json`), paired with the `skills/sparkleforge/SKILL.md` workflow guide.

```bash
# Claude Code
/plugin install sparkleforge --marketplace https://github.com/ppijbb/sparkleforge.git
```

### Attach via `.mcp.json`

```json
{
  "mcpServers": {
    "sparkleforge": {
      "command": "bash",
      "args": ["scripts/run_mcp_server.sh"]
    }
  }
}
```

### Exposed MCP Tools and UI Resources

| Name | Type | Description |
| :--- | :--- | :--- |
| `ui://sparkleforge/board.html` | MCP App Resource | Interactive Swarm roster, task inbox, and Nightwelding status board (`text/html;profile=mcp-app`). |
| `open_board` | Tool | Opens the control board UI (including Antigravity side-pane and global entrypoints) and returns a snapshot summary. |
| `run_task(prompt)` | Tool | Runs a one-shot request in-process and returns the structured JSON result. |
| `start_research(query)` | Tool | Submits a non-blocking background job and returns `{"job_id": "..."}` immediately. |
| `get_report(job_id)` | Tool | Polls job state (`pending`, `running`, `completed`, `failed`) and returns the finished report payload. |
| `cancel_task(job_id)` | Tool | Cancels an active task or background job. |
| `nightwelding_status()` | Tool | Lists local Nightwelding queue items from `~/.sparkleforge/nightwelding/queue.json`. |
| `search_skills(query)` / `get_skill(skill_id)` | Tool | Searches and retrieves skill bundles from `~/.sparkleforge/skill_marketplace` (override via `SPARKLEFORGE_SKILL_MARKETPLACE_DIR`). |

---

## HTTP Task & Status API

[`src/web/status_api.py`](src/web/status_api.py) provides a bearer-authenticated Starlette service for submitting tasks and polling status from remote clients without exposing provider or database credentials.

```bash
STATUS_API_TOKEN=replace-with-strong-token \
  uv run uvicorn src.web.status_api:app --port 8502
```

Every route requires `Authorization: Bearer <STATUS_API_TOKEN>`. If `STATUS_API_TOKEN` is unset, the service fails closed with `503 Service Unavailable`.

| Endpoint | Backing Store | Description |
| :--- | :--- | :--- |
| `POST /tasks` | Local `sdk_jobs.json` (+ Supabase mirror) | Submits `{"prompt": "..."}` and returns `202 Accepted` with `job_id`. Run `uvicorn` with a single worker in local mode. |
| `GET /tasks/{job_id}/status` | Local `sdk_jobs.json` (+ Supabase mirror) | Returns current execution state and submission timestamp. |
| `GET /tasks/{job_id}/report` | Local `sdk_jobs.json` (+ Supabase mirror) | Returns the completed report (`409` while running, `500` on failure). |
| `GET /jobs/{job_id}/status` | Supabase `forge_jobs` | Reads hosted job state when Supabase is configured. |
| `GET /reports/{report_id}` | Supabase `reports` | Reads hosted report rows when Supabase is configured. |

Public hosted telemetry from Supabase-backed runs is published at [sparkleforge.streamlit.app](https://sparkleforge.streamlit.app) (`src/web/live_dashboard.py`).

---

## Verification and Benchmarks

SparkleForge separates structural governance checks from empirical capability benchmarks so neither can mask regressions in the other:

- **Continuous OS-Plane Integrity**: [`tests/test_os_plane_integrity.py`](tests/test_os_plane_integrity.py) exercises capability grants, action journaling, task dashboard hooks, and session quotas through production constructors on every pull request in [`.github/workflows/pr-merge-gate.yml`](.github/workflows/pr-merge-gate.yml).
- **Scenario Evaluation (`24.0%` latest adjusted pass rate)**: [`docs/BENCHMARK_REPORT.md`](docs/BENCHMARK_REPORT.md) is verified in CI by `python scripts/generate_benchmark_report.py --check` directly from [`tests/benchmark/baselines/scenario_history.jsonl`](tests/benchmark/baselines/scenario_history.jsonl) across five scenarios (`system_cleanup`, `scheduled_summary`, `semantic_search`, `security_scan`, `build_env`).
- **SWE-bench Lite (`0 / 6` on current sampled subset)**: [`docs/SWEBENCH_REPORT.md`](docs/SWEBENCH_REPORT.md) records weekly runs ([`.github/workflows/swebench-weekly.yml`](.github/workflows/swebench-weekly.yml)) of the production `fix-issue` entrypoint scored by the unmodified upstream `swebench.harness.run_evaluation` CLI. Empty-patch and evaluation failures are recorded as-is while low-cost fault-localization and test-time scaling harnesses are being hardened.
- **Operational Telemetry**: Reactive scheduling eliminates idle LLM polling during asynchronous waits (measured up to `92%` token reduction on background-heavy runs). Initial self-hosted CI repair runs recorded `141.08 min` mean time to merge (`2 / 3` PRs on an `N=3` sample); automated cross-run error remediation over `agent_error_contexts` is schema-defined but not yet instrumented in `src/`.

---

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): System planes, orchestrator flow, and security boundaries
- [`docs/ANVIL_PLAN.md`](docs/ANVIL_PLAN.md): Anvil execution layer phase history and roadmap
- [`docs/BENCHMARK_REPORT.md`](docs/BENCHMARK_REPORT.md): Reproducible scenario scores, momentum checks, and operational metrics
- [`docs/SWEBENCH_REPORT.md`](docs/SWEBENCH_REPORT.md): Weekly upstream SWE-bench Lite evaluation log
- [`CLAUDE.md`](CLAUDE.md) & [`SPARKLEFORGE.md`](SPARKLEFORGE.md): Repository conventions, development rules, and pull request checks

---

## Contributing

Contributions are welcome. Review [`CLAUDE.md`](CLAUDE.md) and [`.github/workflows/pr-merge-gate.yml`](.github/workflows/pr-merge-gate.yml) before opening a pull request. Commit subjects must follow lowercase conventional commit format (`feat(scope): summary`, validated by [`scripts/validate_commit_messages.py`](scripts/validate_commit_messages.py)).

## Acknowledgments

SparkleForge draws architectural ideas from the open-source agent and research ecosystem, including **auto-deep-researcher-24x7** (continuous execution and bounded memory), **LangGraph & Open Deep Research** (cyclic state-machine workflows), **LightMem** (hierarchical context compression), and **OpenManus** (modular multi-agent tool orchestration).

## License

Released under the [MIT License](https://opensource.org/licenses/MIT) (`pyproject.toml`).
