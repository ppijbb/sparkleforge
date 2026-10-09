import logging

from src.core.health_monitor import HealthMonitor
logger = logging.getLogger(__name__)


async def run_command(args, config):
    """`run <query>` -- research/work is no longer a CLI-level choice.

    Always goes through AgentHarness's classify/TaskRouter (LLM) node via
    the coworker path (force_coworker=False), so the agent decides per
    request instead of a static flag.
    """
    from src.cli.commands.work import work_command_from_query

    HealthMonitor.suppress_noisy_init = True
    return await work_command_from_query(args, force_coworker=False)
