---
name: CLI-Anything Builder
description: Build agent-native CLI harnesses for any GUI application using the CLI-Anything 7-phase methodology. Use when the user or task requires controlling software that has no existing CLI.
version: "1.0.0"
triggers:
  - "build a CLI for"
  - "make .* agent-controllable"
  - "CLI-Anything"
  - "cli-anything"
  - "harness for"
  - "control .* via command line"
---

# CLI-Anything Builder Skill

## Overview

This skill guides the agent to build production-ready, stateful CLI interfaces for GUI applications using the CLI-Anything methodology. When a user or research task requires controlling an application (e.g. GIMP, Blender, LibreOffice, or any software with a codebase) that does not yet have an agent-usable CLI, use this skill to generate and install a full harness, then control it via the `run_cli_anything` MCP tool.

## Capabilities

- Analyze a target software codebase (local path or GitHub repo) and design a CLI that mirrors GUI functionality
- Implement the complete 7-phase pipeline: analysis, design, implementation, test planning, test implementation, documentation, installation
- Produce installable `cli-anything-<software>` packages with REPL mode and `--json` output for agents
- After building, use `cli_anything_run` to invoke the new CLI with structured output

## Instructions

1. **Read the methodology first.** Before doing any implementation, read the authoritative CLI-Anything methodology document:
   - Path: `reference/CLI-Anything/cli-anything-plugin/HARNESS.md` (relative to the mcp_agent workspace) or the same file under the SparkleForge project if the reference is available at `../reference/CLI-Anything/` or as configured).
   - Follow HARNESS.md strictly. Do not improvise; every phase (analysis, design, implementation, test plan, test implementation, test documentation, publish/install) must follow the spec.

2. **Phase 0 – Source acquisition.** If the user provides a GitHub URL, clone the repository to a local directory. Derive the software name from the directory (e.g. `gimp`, `blender`).

3. **Phases 1–2 – Analysis and design.** Analyze the codebase: identify the backend engine, map GUI actions to APIs, identify the data model and existing CLI tools. Design command groups, state model, and output format (human + JSON via `--json`).

4. **Phase 3 – Implementation.** Create the directory structure under `agent-harness/cli_anything/<software>/` (core, utils, tests). Implement core modules, Click CLI with REPL support, and a `utils/<software>_backend.py` that invokes the **real** software (no reimplementation). Use the real application for rendering/export.

5. **Phases 4–6 – Tests.** Write `TEST.md` (plan first), then implement unit tests and E2E tests that call the real software. Append test results to `TEST.md`.

6. **Phase 7 – Publish.** Add `setup.py` with `find_namespace_packages(include=["cli_anything.*"])`, package name `cli-anything-<software>`. Run `pip install -e .` in the agent-harness directory and verify `which cli-anything-<software>`.

7. **After installation.** Tell the user (or the orchestrator) that the new CLI is available. The agent can then use the **run_cli_anything** MCP tool (or **cli_anything_run** in tools_config) with:
   - `software`: the derived name (e.g. `gimp`, `blender`)
   - `command`: the subcommand and arguments as one string (e.g. `project new -o out.json`)
   - `use_json`: true for machine-readable output

Use **list_cli_anything_tools** to see all installed harnesses in PATH.

## Usage

This skill is invoked when:

- The user asks to build a CLI for a specific application or to make software agent-controllable
- A research or automation task requires controlling a GUI application that has no existing CLI
- The planner or executor identifies that a CLI harness is needed for a target software

## Dependencies

- Access to `reference/CLI-Anything` (HARNESS.md and plugin structure)
- Filesystem and code execution (to create files and run `pip install`)
- After build: MCP tool `run_cli_anything` / `cli_anything_run` for execution

## Resources

- Methodology: `reference/CLI-Anything/cli-anything-plugin/HARNESS.md`
- Example harnesses: `reference/CLI-Anything/gimp/agent-harness`, `reference/CLI-Anything/blender/agent-harness`, etc.
- ReplSkin: `reference/CLI-Anything/cli-anything-plugin/repl_skin.py` (copy into each harness `utils/repl_skin.py`)

## Critical Rules (from HARNESS.md)

- **Use the real software** for rendering and export; do not reimplement functionality in Python.
- **No graceful degradation** in tests: if the backend is missing, tests must fail with clear install instructions.
- **Verify outputs** (magic bytes, format checks); do not assume success from exit code alone.
- **Filter translation**: when mapping effects to a renderer (e.g. MLT to ffmpeg), handle duplicate filters, stream ordering, and parameter scaling as described in HARNESS.md.
