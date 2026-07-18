"""ResearchAgent - package facade.

``src/agents/research_agent.py`` used to be a single 2,983-line module
(issue #582, mirroring the Sigma-1 split of ``mcp_integration.py``). It's now
this package, split by responsibility:

- ``agent.py``: the core ``ResearchAgent`` class -- init, LLM plumbing, and
  task lifecycle. Composes the mixins below via multiple inheritance.
- ``search_providers.py``: web/academic search backends (Tavily, Exa, Brave,
  Serper, DuckDuckGo, etc.) as ``SearchProvidersMixin``.
- ``data_collection.py``: data collection/processing pipeline as
  ``DataCollectionMixin``.
- ``task_pipelines.py``: analysis/synthesis/validation/general-research task
  pipelines as ``TaskPipelinesMixin``.
- ``quality_metrics.py``: quality/confidence/completeness scoring helpers as
  ``QualityMetricsMixin``.
- ``browser.py``: browser-driven navigation/search/interactive research as
  ``BrowserAutomationMixin``.

``ResearchAgent`` is the only name that was importable from
``src.agents.research_agent`` before the split (confirmed via a repo-wide
grep of import sites: ``src/dataflow/operators/research_operator.py`` and
``src/core/orchestrator/delegation.py``), and it's re-exported here unchanged
so ``from src.agents.research_agent import ResearchAgent`` keeps working for
all existing call sites without modification.
"""

from src.agents.research_agent.agent import ResearchAgent

__all__ = ["ResearchAgent"]
