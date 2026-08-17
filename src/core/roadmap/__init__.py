"""Daily-roadmap judgment logic (target selection, fallback content, Anvil doc sync).

Moved out of sparkleforge-daily-roadmap.yml's python heredocs -- these are
selection/formatting decisions, not mechanical data extraction, so they
belong here alongside the rest of the CLI-driven CI logic. GitHub context
collection (gh/jq calls) and git/commit/push/PR mechanics stay in the
workflow; the latter now calls `sparkleforge ci publish`.
"""
