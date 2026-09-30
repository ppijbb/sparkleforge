"""Console-script entrypoint shim for the ``sparkleforge``/``sparkle`` commands.

Guards against a real crash (issue #1757): the actual entrypoint lives at
``src.cli.entry:main_entry``, but ``src`` is a generic top-level package name.
If the invoking shell's ``PYTHONPATH`` happens to expose an unrelated
project that also ships a package named ``src``, plain path-based resolution
can pick up that other ``src`` instead of this repo's, crashing on an
unrelated missing dependency. This module is named uniquely (unlikely to
collide with anything on PYTHONPATH) and puts this repo's own root at the
front of ``sys.path`` before importing ``src``, so resolution is always this
project's regardless of PYTHONPATH ordering.
"""

from __future__ import annotations

import os
import sys


def _prioritize_repo_root() -> None:
    """Put this repo's own root at sys.path[0] so `src` always resolves here."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root in sys.path:
        sys.path.remove(project_root)
    sys.path.insert(0, project_root)


def main_entry() -> int:
    _prioritize_repo_root()
    # If src/ does not exist (e.g. non-editable standard pip install where
    # sparkleforge_bootstrap.py is located inside site-packages), skip
    # prepending a non-existent or incorrect src path.
    if not os.path.isdir(src_path):
        return

    from src.cli.entry import main_entry as _main_entry

    return _main_entry()
