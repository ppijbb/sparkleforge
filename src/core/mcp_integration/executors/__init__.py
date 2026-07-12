"""Per-category tool-execution dispatchers, extracted from the monolithic
``src/core/mcp_integration/tools.py`` (Anvil Phase Sigma, issue #507/#524).

Each module here owns exactly one ``ToolCategory``'s ``_execute_*_tool``
implementation (and its LangChain-facing sync wrapper, where one exists).
``tools.py`` re-exports the names external callers use; ``hub.py`` imports
directly from these modules.
"""
