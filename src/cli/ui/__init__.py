"""Shared terminal UX layer for SparkleForge's CLI.

Every subcommand and REPL command should render user-facing output
through this package (backed by ``rich``) instead of ad hoc ANSI codes,
raw ``print()``, or piping status text through ``logging``.
"""
