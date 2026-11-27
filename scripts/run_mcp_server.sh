#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SPARKLEFORGE_HOME:-${CLAUDE_PLUGIN_ROOT:-${PLUGIN_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}}}"

if [ -f "${REPO_ROOT}/.env" ]; then
  # Export only necessary env vars explicitly to avoid leaking all secrets via set -a
  export OPENROUTER_API_KEY ANTHROPIC_API_KEY GITHUB_TOKEN GOOGLE_API_KEY GEMINI_API_KEY NVIDIA_API_KEY SPARKLE_LLM_API_KEY
  set -o allexport
  . "${REPO_ROOT}/.env" >/dev/null 2>&1 || true
  set +o allexport
fi

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export SPARKLE_LLM_MODEL_PATH="${SPARKLE_LLM_MODEL_PATH:-gemini-3.1-flash-lite-preview}"
export SPARKLEFORGE_QUIET_BOOT="1"

pick_python() {
  local candidate
  for candidate in \
    "${SPARKLEFORGE_PYTHON:-}" \
    "${REPO_ROOT}/.venv/bin/python" \
    "${REPO_ROOT}/venv/bin/python" \
    "$(command -v python3 2>/dev/null || true)" \
    "$(command -v python 2>/dev/null || true)"; do
    if [ -n "${candidate}" ] && [ -x "${candidate}" ]; then
      if "${candidate}" -c "import mcp" >/dev/null 2>&1; then
        printf '%s\n' "${candidate}"
        return 0
      fi
    fi
  done
  return 1
}

if PYTHON_BIN="$(pick_python)"; then
  exec "${PYTHON_BIN}" -m src.core.mcp_servers.sparkleforge_server
fi

if [ -f "${REPO_ROOT}/pyproject.toml" ] && command -v uv >/dev/null 2>&1; then
  if uv run --quiet --project "${REPO_ROOT}" python -c "import mcp" >/dev/null 2>&1; then
    exec uv run --quiet --project "${REPO_ROOT}" python -m src.core.mcp_servers.sparkleforge_server
  fi
fi

echo "Error: could not find a Python interpreter with 'mcp' installed for SparkleForge MCP server." >&2
exit 1
