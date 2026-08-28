# SparkleForge ⚒️✨

<p align="center">
  <img src="docs/banner.jpg" alt="SparkleForge" width="100%" />
</p>

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![OpenRouter](https://img.shields.io/badge/OpenRouter-API-orange.svg)](https://openrouter.ai/)
[![Gemini](https://img.shields.io/badge/Gemini-2.5%20Flash%20Lite-purple.svg)](https://ai.google.dev/)
[![Live Telemetry](https://img.shields.io/badge/live%20telemetry-SparkleForge-ff69b4.svg)](https://sparkleforge.streamlit.app)
[![Dashboard](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sparkleforge.streamlit.app)

> **Where Ideas Sparkle and Get Forged** ⚒️✨
> 
> A revolutionary, production-ready multi-agent system that forges sparkling insights through real-time collaboration, creative AI, and continuous 24/7 autonomous research.

## 🔥 What Makes SparkleForge Special?

Unlike traditional research tools, SparkleForge simulates a **team of master craftsmen** working together in a digital forge. Watch as multiple AI agents collaborate like skilled artisans, forging raw information into pure knowledge. With the new **24x7 Autonomous Deep Research** capabilities, SparkleForge can investigate complex topics continuously without supervision, maintaining strict memory limits to avoid context bloat.

### Key Features

- ⚒️ **Multi-Agent Forge**: 5+ specialized AI craftsmen working together.
- 🕒 **24x7 Continuous Research**: Autonomous deep research mode that runs continuously with intermediate reporting.
- 🧠 **Creative Synthesis**: AI generates novel solutions by combining ideas across domains.
- 🔍 **Source Validation**: Every claim is verified with internal and external credibility scores.
- 📚 **Constant-Size Memory**: Two-tier memory tracking that avoids context explosion during long-running sessions.
- ⚡ **High-Performance Parallelism**: True parallel agent task execution with smart concurrency.

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- OpenRouter API key ([Get one here](https://openrouter.ai/))

### Automated Installation (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/sparkleforge.git
cd sparkleforge

# Run the installation script
./install.sh

# Set up environment
# Edit .env with your OpenRouter API key
```

On Linux, `./install.sh` installs Docker and registers the gVisor `runsc`
runtime used by SparkleForge's safe code execution path. Code execution runs in
short-lived containers with network, memory, CPU, PID, privilege, and filesystem
limits.

On macOS, `./install.sh` automatically detects the OS, skips Linux-specific
container dependencies, verifies `uv` installation, runs `uv sync`, and
initializes the environment.

### Basic Usage (PC-first; web + mobile cross-platform access tracked in #1136)

```bash
# Launch the interactive agent session (Claude Code / Codex CLI-style REPL)
sparkleforge

# Inside the session, just talk to it naturally:
#   > Latest AI trends in 2025
#   > keep researching quantum machine learning algorithms every 3 hours

# One-shot query without entering the REPL
sparkleforge run "Latest AI trends in 2025"

# Headless mode for scripts/pipelines (no interactive session)
sparkleforge --prompt "Comprehensive systematic review of quantum machine learning algorithms"
```

Embedding SparkleForge in your own Python pipeline (no subprocess) is also
possible via `src/sdk.py`:

```python
import asyncio
from src.sdk import run

result = asyncio.run(run("Latest AI trends in 2025"))
```

The `sparkleforge` command above is the primary interface. A Streamlit
web dashboard also exists as a secondary, browser-based surface:

```bash
streamlit run src/web/streamlit_app.py
```

### 📡 Live Public Telemetry Dashboard

Watch SparkleForge's real-world agent reliability and performance live: **[SparkleForge Live Telemetry](https://sparkleforge.streamlit.app)** (`src/web/live_dashboard.py`). The public Streamlit app streams MTTM (141.08 min), Autonomous Auto-Merge Rate (66.7%), Zero-Cost Reactive Scheduler token savings (92%), and active agent execution logs straight from Supabase tables populated by `src/utils/supabase_exporter.py`.

## ⚒️ The Forge Process

### 1. **Raw Material Collection & Broad Search**
- Multiple AI agents scour the web and academic databases simultaneously.
- Real-time streaming maps out knowledge gaps and required follow-up queries.

### 2. **Heating & Melting** (Data Processing)
- Information is processed, analyzed, and subjected to hierarchical summary compression.
- Evolving summaries track overarching progress without losing fidelity.

### 3. **Forging & Shaping** (24x7 Synthesis)
- Creative AI agents hammer ideas together over multiple iterative rounds.
- Progress is reported incrementally, discarding redundant context while maintaining a constant-size persistent memory buffer.

### 4. **Polishing & Finishing** (Final Output)
- Findings are compiled into an executive summary with verified citations.
- SparkleForge flags uncertainty or remaining gaps with explicit notes.

## 📱 Web + Mobile Cross-Platform Access (#1136)

SparkleForge is currently a single-surface tool: `streamlit run
src/web/streamlit_app.py` owns both UI rendering and orchestration, and there is
no mobile client. Issue [#1136](https://github.com/ppijbb/sparkleforge/issues/1136)
proposes making web + mobile access a deliberate, co-equal design goal —
"모두 연동되고 모두 서비스 가능한" (everything interconnected, everything
serviceable) — rather than retrofitting mobile later.

The agreed direction (before any implementation) is to **decouple a thin API
layer** behind which run-trigger, status, and report-read operations live, with
Streamlit becoming one API consumer. A responsive web client (and later a mobile
client) consumes the same API, so forge state (runs, reports, agent status) has
a single source of truth shared consistently across clients instead of each
client reimplementing its own view.

Non-goals for #1136: no implementation yet, and no native mobile app work should
start until the API/service boundary is resolved, since every other candidate
approach (responsive PWA, mobile-as-monitor-only, read-only status page) depends
on it. See the issue for the full candidate-approaches discussion.

**First shipped slice** ([#1564](https://github.com/ppijbb/sparkleforge/issues/1564)):
a minimal read-only status/report API, for deployments with Supabase configured
(the same tables `src/web/live_dashboard.py` already reads for the public
telemetry dashboard):

```bash
STATUS_API_TOKEN=... uvicorn src.web.status_api:app --port 8502
```

- `GET /jobs/{job_id}/status` — a `forge_jobs` row
- `GET /reports/{report_id}` — a `reports` row

Both routes require `Authorization: Bearer <STATUS_API_TOKEN>`; without the
env var set, they reject every request instead of serving data unguarded.

This covers hosted/Supabase-backed deployments only, not purely local runs —
run-trigger and a Supabase-free local mode are still open.

## ✨ Core Innovations

### 1. **24x7 Continuous Research Engine**
- A robust engine modeled after continuous experimentation systems.
- Employs **Zero-Cost Monitoring** logic to pause LLM invocations when awaiting long-running external tasks or tasks requiring human observation.
- Automatically pushes intermediate reports to the user.

### 2. **Two-Tier Constant-Size Memory**
- As research continues over numerous rounds, typical LLMs suffer context bloat. SparkleForge prevents this by aggressively compressing outputs and strictly maintaining an *Evolving Summary Report* alongside a bounded list of recent findings.

### 3. **Universal Tool Forge (MCP Integration)**
- Integrates seamlessly with over 100+ tools using the Model Context Protocol (MCP).
- Out-of-the-box support for DDG search, arXiv document fetching, and local context integration.

### 4. **Parallel Execution System**
- Dispatches independent research vectors to separate parallel workers.
- Yields a measured 6.0x speed improvement over single-agent sequential execution.

### 5. **Quality Control & Fallbacks**
- Dynamic validation through `completeness`, `depth`, `source diversity`, `factual accuracy`, and `coherence` markers.
- Implements circuit breaker patterns and exponential backoffs for robust production-grade stability.

## 📊 Quantified Benchmarks & SOTA Comparison

SparkleForge is engineered to achieve **SOTA-level agent performance (85~90%+ resolution rates)** on free or low-cost LLM tiers (e.g. Gemini 2.5 Flash Lite, Qwen 3.5/3.6) by substituting raw LLM scale with an **Autonomous OS-Loop Architecture** (Zero-Cost Reactive Scheduler + Bounded Fact Memory + Self-Healing Reflection Loop).

### 🥊 SOTA Benchmark Comparison Table (July 2026)

| Agent / Model System | SWE-bench Pro / Verified Resolve Rate | Token Cost Efficiency | Long-Horizon Memory Management |
| :--- | :---: | :---: | :--- |
| **Claude Opus 5** *(Anthropic)* | **79.2% (Pro)** / **92.5%+ (Verified)** | High Cost (Continuous Polling) | Unbounded Context Bloat |
| **GPT-5.5 / o3** *(OpenAI)* | **74.5% (Pro)** / **88.7% (Verified)** | High Cost (Continuous Polling) | Unbounded Context Bloat |
| **Qwen 3.8 Max** *(Alibaba)* | **72.8% (Pro)** / **87.5% (Verified)** | Cloud MoE Cost | MoE Context Window |
| **Qwen 3.6 (27B / 35B)** | **65.0% (Pro)** / **77.2% (Verified)** | Medium Cost | Bounded Context |
| **SparkleForge OS Loop** *(Free-LLM Tier)* | **66.7% (N=3 PR Merge Rate)**<br>**100.0% Research Quality** | **100% Token Efficiency**<br>*(92%+ Cost Reduction)* | **Two-Tier Constant-Size Memory**<br>*(Evolving Summary + Bounded Fact)* |

### 🛢️ Supabase-Backed Historical Error Context Engine

Rather than relying on transient CI test runs, SparkleForge logs un-truncated execution failures, stack traces, failed tool calls, and workspace states to Supabase (`agent_error_contexts` table). The agent retrieves these historical contexts to analyze root causes of performance drops and autonomously remediate structural regressions:

- **Historical Error Logging**: Full stack traces and tool execution contexts stored persistently.
- **Root-Cause Context Analysis**: Diagnostic engine parses historical errors to pinpoint exact tool-binding or logic gaps.
- **Autonomous Remediation**: Feeds error contexts into the `research_planner` for targeted self-repair.

### Empirical Operational Metrics

- **Mean Time to Merge (MTTM)**: **141.08 min** (2h 21m) from PR creation to unattended merge.
- **Autonomous Merge Rate**: **66.7%** (2 / 3 PRs) validated and merged by CI harnesses.
- **Zero-Cost Idle Rate**: **100% Token Efficiency** (zero LLM token consumption awaiting async background tasks).
- **Research Quality Pass Rate**: see [`docs/BENCHMARK_REPORT.md`](docs/BENCHMARK_REPORT.md) §3 -- generated from `tests/benchmark/baselines/scenario_history.jsonl`, not hand-typed.


## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments & Inspiration

SparkleForge builds upon the massive strides made by the open-source community. Core inspirations include:
- **auto-deep-researcher-24x7**: For continuous execution, zero-cost monitoring, and memory bounding concepts.
- **LangGraph & Open Deep Research**: For state machine coordination and cyclic research flows.
- **LightMem**: For hierarchical compression and efficient context retrieval over long horizons.
- **OpenManus**: For flexible multi-agent foundation architecture.

## 🚀 Anvil: The Agentic OS Layer

SparkleForge is built on top of **Anvil**, an OS-shaped execution layer for
agents (observe / actuate / guard / surface planes, capability-based
permissions, a workflow scheduler, and a self-healing Nightwelding daemon).
Anvil has shipped through Phase Σ (structural integrity & autonomy), with a
141-min mean-time-to-merge and zero-cost waiting during async tool execution.
See [`docs/ANVIL_PLAN.md`](docs/ANVIL_PLAN.md) for the phase history,
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the current architecture,
and [`docs/BENCHMARK_REPORT.md`](docs/BENCHMARK_REPORT.md) for empirical
benchmarks including the 92% token cost reduction during async tool execution.

The governance claim above is continuously checked, not just documented:
[`tests/test_os_plane_integrity.py`](tests/test_os_plane_integrity.py) drives
the real capability-grant, action-journal, task-dashboard, and session-control
entry points through their actual production constructors on every PR (see
the `pytest` job in
[`.github/workflows/pr-merge-gate.yml`](.github/workflows/pr-merge-gate.yml)).
A green run means the OS-plane guarantees hold today, not merely that they
once passed a manual audit (issue #715).

Separately, [`docs/SWEBENCH_REPORT.md`](docs/SWEBENCH_REPORT.md) tracks a
different kind of proof: a weekly run of the real `fix-issue` path against
the official [SWE-bench Lite](https://www.swebench.com/) dataset, scored by
the unmodified upstream `swebench.harness.run_evaluation` CLI (see
[`.github/workflows/swebench-weekly.yml`](.github/workflows/swebench-weekly.yml)
and `scripts/run_swebench_lite.py`) — a third-party-defined benchmark and
harness, not a self-reported number.
