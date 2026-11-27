---
name: sparkleforge
description: Drive SparkleForge research jobs, Nightwelding queue inspection, and local skill lookup through the bundled sparkleforge MCP server. Activate when the user asks to delegate deep research to SparkleForge, inspect Nightwelding status, or reuse published SparkleForge skills.
---

# SparkleForge Plugin Guide

This plugin connects coding agents (Claude Code, Antigravity, Codex, Gemini CLI) to the local SparkleForge engine (`src.core.mcp_servers.sparkleforge_server`) over stdio MCP.

## Available MCP Tools

1. **`start_research(query)` + `get_report(job_id)` (Preferred for multi-step work)**
   - Starts a non-blocking SparkleForge job and persists job state locally so long runs do not hit MCP request timeouts.
   - Poll `get_report(job_id)` until `status` is `"completed"` or `"failed"`.
2. **`run_task(prompt)` (Short one-shot requests only)**
   - Runs `src.sdk.run(prompt)` synchronously in-process. Avoid for deep research or multi-minute tasks.
3. **`nightwelding_status()`**
   - Reads `~/.sparkleforge/nightwelding/queue.json` and returns tracked Nightwelding items (`issue_number`, `status`, `branch`, `pr_url`, `failure_reason`).
4. **`search_skills(query)` / `get_skill(skill_id)`**
   - Searches and retrieves reusable manifests from `~/.sparkleforge/skill_marketplace`.

## Operating Guidelines

- Keep development and background runs on low-cost models (`gemini-3.1-flash-lite-preview`, `gpt-5-nano`, or configured `:free` endpoints) unless explicitly instructed otherwise.
- When targeting an external repository, respect that repository's native language and test runner; never add Python `tests/test_*.py` files to non-Python repositories or documentation-only changes.
