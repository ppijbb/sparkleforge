"""Read-only resource discovery helpers."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any


PROJECT_MARKERS = {".git", "pyproject.toml", "package.json", "Cargo.toml", "go.mod"}


def index_path_executables(limit: int = 500) -> list[dict[str, str]]:
    """Index executable files visible through PATH."""
    executables: list[dict[str, str]] = []
    seen: set[str] = set()
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        path = Path(directory)
        if not path.is_dir():
            continue
        try:
            children = sorted(path.iterdir())
        except OSError:
            continue
        for child in children:
            if len(executables) >= limit:
                return executables
            if child.name in seen or not child.is_file() or not os.access(child, os.X_OK):
                continue
            seen.add(child.name)
            executables.append({"name": child.name, "path": str(child)})
    return executables


def find_executables(names: list[str]) -> dict[str, str | None]:
    """Resolve executable names using PATH."""
    return {name: shutil.which(name) for name in names}


def standard_directories(home: Path | None = None) -> dict[str, str | None]:
    """Return common user directories when present."""
    root = home or Path.home()
    candidates = {
        "home": root,
        "desktop": root / "Desktop",
        "documents": root / "Documents",
        "downloads": root / "Downloads",
        "workspace": root / "workspace",
    }
    return {name: str(path) if path.exists() else None for name, path in candidates.items()}


def find_project_directories(
    roots: list[str | Path] | None = None,
    *,
    max_depth: int = 3,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Find likely project directories by common marker files."""
    search_roots = [Path.cwd(), Path.home() / "workspace"] if roots is None else [Path(p) for p in roots]
    projects: list[dict[str, Any]] = []

    def walk(path: Path, depth: int) -> None:
        if len(projects) >= limit or depth > max_depth or not path.is_dir():
            return
        try:
            names = {child.name for child in path.iterdir()}
        except OSError:
            return
        markers = sorted(PROJECT_MARKERS & names)
        if markers:
            projects.append({"path": str(path), "markers": markers})
            return
        for child in sorted(path.iterdir(), key=lambda p: p.name):
            if child.name.startswith(".") and child.name != ".github":
                continue
            walk(child, depth + 1)

    for root in search_roots:
        walk(root.expanduser(), 0)
    return projects


def locate_resources(
    roots: list[str | Path],
    *,
    extensions: list[str] | None = None,
    modified_within_days: int | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """Find files by extension and optional modification age."""
    suffixes = {ext if ext.startswith(".") else f".{ext}" for ext in extensions or []}
    now = None
    if modified_within_days is not None:
        import time

        now = time.time()
    matches: list[dict[str, Any]] = []
    for root in [Path(p).expanduser() for p in roots]:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if len(matches) >= limit:
                return matches
            if not path.is_file():
                continue
            if suffixes and path.suffix not in suffixes:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            if now is not None and now - stat.st_mtime > modified_within_days * 86400:
                continue
            matches.append(
                {
                    "path": str(path),
                    "size_bytes": stat.st_size,
                    "modified_at": stat.st_mtime,
                }
            )
    return matches
