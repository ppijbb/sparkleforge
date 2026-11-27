#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SPARKLEFORGE_HOME:-${CLAUDE_PLUGIN_ROOT:-${PLUGIN_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}}}"

if [ -f "${REPO_ROOT}/.env" ]; then
  while IFS='=' read -r key value; do
    # Skip comments and empty lines
    case "$key" in
      '#'* | '') continue ;;
    es:
    # Remove leading/trailing whitespace or quotes from key/value if needed
    key=$(echo "$key" | tr -d '[:space:]')
    value=$(echo "$value" | sed -E 's/^[[:space:]]*["\x27]?//; s/["\x27]?[[:space:]]*$//')
    if [ -n "$key" ]; then
      export "$key"="$value"
    fi
  done < <(grep -v '^#' "${REPO_ROOT}/.env" | grep '=')
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
      if "${candidate}" -c "import mcp, fastmcp" >/dev/null 2>&1; then
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
  if uv run --quiet --project "${REPO_ROOT}" python -c "import mcp, fastmcp" >/dev/null 2>&1; then
    exec uv run --quiet --project "${REPO_ROOT}" python -m src.core.mcp_servers.sparkleforge_server
  fi
fi

echo "Error: could not find a Python interpreter with 'mcp' installed for SparkleForge MCP server." >&2
exit 1
