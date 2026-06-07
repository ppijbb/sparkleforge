"""CLI entry point for the installed sparkleforge command."""

import os
import sys
import asyncio
from pathlib import Path
from src.core.env_configurator import verify_environment


async def main_entry():
    """Run the repository-level CLI entry point from an installed script."""
    if len(sys.argv) == 1 or "--help" in sys.argv or "-h" in sys.argv:
        print("SparkleForge CLI")
        print("Usage: sparkleforge [options] --request <query>")
        return

    try:
        await asyncio.to_thread(verify_environment)
    except Exception as e:
        print(f"Security environment validation failed: {e}")
        sys.exit(1)

    project_root = Path(__file__).resolve().parent.parent.parent
    os.chdir(project_root)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from main import main_entry as repository_main_entry

    repository_main_entry()


if __name__ == "__main__":
    asyncio.run(main_entry())
