"""CLI entry point for the installed sparkleforge command."""

import os
import sys
from pathlib import Path


def main_entry():
    """Run the repository-level CLI entry point from an installed script."""
    project_root = Path(__file__).resolve().parent.parent.parent
    os.chdir(project_root)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from main import main_entry as repository_main_entry

    repository_main_entry()


if __name__ == "__main__":
    main_entry()
