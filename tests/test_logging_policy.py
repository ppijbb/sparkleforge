import logging

from src.cli.ui.logging_policy import QUIET_LOGGER_NAMES, apply_repl_quiet_mode


def test_cli_agent_manager_is_quieted():
    assert "src.core.cli_agents.cli_agent_manager" in QUIET_LOGGER_NAMES


def test_apply_quiet_mode_suppresses_registration_noise():
    noisy = logging.getLogger("src.core.cli_agents.cli_agent_manager")
    noisy.setLevel(logging.NOTSET)
    apply_repl_quiet_mode()
    assert not noisy.isEnabledFor(logging.INFO)
