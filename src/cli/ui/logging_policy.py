"""Shared logger-suppression policy for the REPL's "quiet mode" and for
actively-watching-an-agent-work console output.

Both main.py's REPL entry path and REPLCLI's constructor need the exact
same suppression applied; keeping it in one place instead of two hardcoded
copies is what stops it from silently drifting out of sync (checkpoint
manager / session storage / session control / context loader all leaked
raw init chatter above the REPL banner because neither of the old copies
had been updated when those modules added their own __init__ logging).
"""

from __future__ import annotations

import logging
import re
import warnings

# Environment variables set before importing heavy ML/LLM dependencies unless
# --verbose is requested.
QUIET_BOOT_ENV_VARS: dict[str, str] = {
    "TQDM_DISABLE": "1",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "TRANSFORMERS_VERBOSITY": "error",
    "TOKENIZERS_PARALLELISM": "false",
    "SPARKLEFORGE_QUIET_BOOT": "1",
}

# Infra/init loggers whose own status text isn't meant for the chat surface --
# real chat output goes through src.cli.* or is printed directly via rich,
# never through these. A new "src.core.*" module that logs at __init__ time
# should be added here rather than silently leaking through.
QUIET_LOGGER_NAMES = (
    "__main__",
    "src.core.agent_orchestrator",
    "src.core.mcp_integration",
    "src.core.shared_memory",
    "src.core.skills_manager",
    "src.core.prompt_refiner_wrapper",
    "src.core.checkpoint_manager",
    "src.core.session_storage",
    "src.core.session_control",
    "src.core.context_loader",
    "src.core.scheduler",
    "src.core.cli_agents.cli_agent_manager",
    "streamlit",
    "streamlit.runtime",
    "local_researcher",
)

# Loggers whose startup/init/fallback chatter should stay at WARNING+ and off
# the console in non-verbose runs while still reaching log files.
STARTUP_QUIET_LOGGER_NAMES = (
    "src.core.llm_manager",
    "src.core.llm_manager.model_registry",
    "src.core.llm_manager.orchestrator",
    "src.core.llm_manager.cascade",
    "src.core.llm_manager.providers",
    "src.core.llm_manager.routing",
    "src.core.skill_tree",
    "src.core.memory_embeddings",
    "src.core.task_router",
    "google_genai",
    "google_genai._api_client",
    "google.genai",
    "google.generativeai",
    "transformers",
    "huggingface_hub",
    "sentence_transformers",
    "flashrank",
    "chromadb",
    "tqdm",
    "httpx",
    "httpcore",
    "urllib3",
    "langfuse",
    "py.warnings",
)


class StartupNoiseFilter(logging.Filter):
    """Console filter that routes noisy startup/fallback warnings to log files only.

    Suppresses known non-fatal startup and provider-fallback warnings from
    console handlers when quiet boot is active, while keeping ERROR+ records
    and user-facing ``src.cli.*`` output visible on the terminal and preserving
    all records in file handlers.
    """

    NOISY_LOGGER_PREFIXES: tuple[str, ...] = (
        "google_genai",
        "google.genai",
        "google.generativeai",
        "transformers",
        "huggingface_hub",
        "sentence_transformers",
        "flashrank",
        "chromadb",
        "tqdm",
        "py.warnings",
    )

    NOISY_MESSAGE_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"Both GOOGLE_API_KEY and GEMINI_API_KEY are set", re.IGNORECASE),
        re.compile(r"FlashRank Ranker unavailable", re.IGNORECASE),
        re.compile(r"Embedding provider unavailable", re.IGNORECASE),
        re.compile(r"Failed to build skill vectors, fallback to BM25", re.IGNORECASE),
        re.compile(r"sentence_transformers is not installed", re.IGNORECASE),
        re.compile(r"Gemini API key not found\. Embeddings might fail", re.IGNORECASE),
        re.compile(r"[A-Z0-9_]+_API_KEY not found\s*-\s*.*unavailable", re.IGNORECASE),
        re.compile(
            r"(?:OpenRouter|Groq|Gemini|OpenAI/GPT|NVIDIA NIM|Cerebras) models not loaded",
            re.IGNORECASE,
        ),
        re.compile(
            r"Failed to load (?:OpenRouter|Groq|Gemini|OpenAI|NVIDIA|Cerebras) models",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:OpenRouter|Groq|Gemini|OpenAI/GPT|NVIDIA NIM|Cerebras) disabled\s*-",
            re.IGNORECASE,
        ),
        re.compile(r"Cascade execution failed:.*falling back", re.IGNORECASE),
        re.compile(r"(?:trying|Trying) fallback(?: model)?", re.IGNORECASE),
        re.compile(
            r"Fallback model .* (?:failed|rate limited|rejected|not available)",
            re.IGNORECASE,
        ),
        re.compile(r"marked as rate-limited|All providers are rate-limited", re.IGNORECASE),
        re.compile(r"rate-limited \(429\), will use next provider", re.IGNORECASE),
        re.compile(
            r"returned HTTP (?:400|402|404|429) in OpenRouter, trying fallback",
            re.IGNORECASE,
        ),
        re.compile(r"Attempting to use replacement model", re.IGNORECASE),
        re.compile(r"TaskRouter: (?:attempt|agent assign attempt) \d+/\d+", re.IGNORECASE),
        re.compile(r"\[GeminiCache\] google-genai SDK unavailable", re.IGNORECASE),
        re.compile(r"library not installed\. Install with:", re.IGNORECASE),
    )

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.ERROR:
            return True
        if record.name.startswith("src.cli"):
            return True
        if record.name.startswith(self.NOISY_LOGGER_PREFIXES):
            return False
        message = record.getMessage()
        return not any(pattern.search(message) for pattern in self.NOISY_MESSAGE_PATTERNS)


def _set_min_warning_level(logger_names: tuple[str, ...]) -> None:
    """Ensure listed loggers do not emit INFO/DEBUG below WARNING."""
    for logger_name in logger_names:
        lg = logging.getLogger(logger_name)
        if lg.level == logging.NOTSET or lg.level < logging.WARNING:
            lg.setLevel(logging.WARNING)


def find_console_handlers() -> list[logging.Handler]:
    """Return console handlers on the root logger, excluding file and pytest capture handlers."""
    try:
        from src.cli.ui.console_log_handler import RichConsoleHandler
    except Exception:
        RichConsoleHandler = None  # type: ignore[assignment]

    handlers: list[logging.Handler] = []
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler):
            continue
        if type(handler).__name__ == "LogCaptureHandler":
            continue
        if RichConsoleHandler is not None and isinstance(handler, RichConsoleHandler):
            handlers.append(handler)
        elif isinstance(handler, logging.StreamHandler):
            handlers.append(handler)
    return handlers


def attach_startup_noise_filters() -> None:
    """Attach StartupNoiseFilter idempotently to all root console handlers."""
    for handler in find_console_handlers():
        if not any(isinstance(flt, StartupNoiseFilter) for flt in handler.filters):
            handler.addFilter(StartupNoiseFilter())


def remove_startup_noise_filters() -> None:
    """Remove StartupNoiseFilter from all root handlers (e.g. for --verbose)."""
    for handler in logging.getLogger().handlers:
        for flt in list(handler.filters):
            if isinstance(flt, StartupNoiseFilter):
                handler.removeFilter(flt)


def apply_repl_quiet_mode() -> None:
    """Suppress infra/init logger chatter so only chat-facing output reaches the console."""
    for logger_name in QUIET_LOGGER_NAMES:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    _set_min_warning_level(STARTUP_QUIET_LOGGER_NAMES)
    attach_startup_noise_filters()


def apply_quiet_boot_logging(suppress_warnings: bool = True) -> None:
    """Apply full quiet-boot logging policy for non-verbose CLI execution.

    Suppresses noisy INFO chatter, routes startup/fallback WARNING lines off
    console handlers (preserving them in log files), and filters noisy
    third-party startup warnings.
    """
    apply_repl_quiet_mode()
    if suppress_warnings:
        warnings.filterwarnings(
            "ignore", category=FutureWarning, module=r"google\.generativeai.*"
        )
        warnings.filterwarnings(
            "ignore", message=r".*urllib3.*or.*chardet.*doesn't match.*"
        )


class ChatModeFilter(logging.Filter):
    """Console-only allowlist for while an agent turn is actively running.

    QUIET_LOGGER_NAMES is a blocklist and it keeps drifting: every internal
    module that starts logging at INFO (capability_manager's "Granted
    capability ...", mode_controller's "Execution mode switched: ...",
    llm_manager's "Executing with NVIDIA NIM model: ...", agent_loop's
    "Iteration N/30", ...) leaks to the console as raw internal-state text
    until someone notices and adds it to the list one name at a time. None
    of that is what a user watching an agent work wants to see -- it's
    orchestration plumbing, not "here's what the agent just did". The
    curated version of that (a spinner caption, a per-tool-call trace line)
    already exists elsewhere; this just stops the raw duplicate from also
    reaching the console.

    Flipped to an allowlist instead: only src.cli.* (actual chat-facing
    output) and WARNING+ (real problems) get through. Everything else still
    goes to the log file -- this only changes what's visible on screen.
    Scoped to attach/detach around the specific span where an agent is
    actively executing (`stage_status()`), not applied globally, so
    diagnostic commands like `health`/`tools` that print non-src.cli INFO
    lines as their actual payload are unaffected.
    """

    ALLOWED_PREFIXES = ("src.cli",)

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        return record.name.startswith(self.ALLOWED_PREFIXES)


def find_console_handler():
    """Return the RichConsoleHandler on the root logger, if any."""
    from src.cli.ui.console_log_handler import RichConsoleHandler
    try:
        from _pytest.logging import LogCaptureHandler
    except ImportError:
        try:
            from pytest import LogCaptureHandler
        except ImportError:
            LogCaptureHandler = None

    for handler in logging.getLogger().handlers:
        if isinstance(handler, RichConsoleHandler):
            return handler
        if LogCaptureHandler and isinstance(handler, LogCaptureHandler):
            return handler
        if hasattr(handler, "capture") and handler.__class__.__name__ == "LogCaptureHandler":
            return handler
    return None
