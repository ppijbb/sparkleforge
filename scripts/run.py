#!/usr/bin/env python3
"""Cross-platform run/setup entry point for SparkleForge.

Mirrors scripts/run.sh so Windows users can run without WSL/Git Bash:
    python scripts/run.py            # setup + run
    python scripts/run.py --setup-only
    python scripts/run.py research "topic"
"""

import os
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / "venv"


def _info(msg):
    print(f"[INFO] {msg}")


def _success(msg):
    print(f"[SUCCESS] {msg}")


def _warn(msg):
    print(f"[WARNING] {msg}")


def _error(msg):
    print(f"[ERROR] {msg}")


def _python_bin():
    if VENV.exists():
        if os.name == "nt":
            return VENV / "Scripts" / "python.exe"
        return VENV / "bin" / "python"
    return sys.executable


def check_python_version():
    if sys.version_info < (3, 11):
        _error(f"Python 3.11 or higher is required. Current: {sys.version.split()[0]}")
        return False
    _success(f"Python version: {sys.version.split()[0]}")
    return True


def setup_venv():
    if not VENV.exists():
        _info("Creating Python virtual environment...")
        venv.create(VENV, with_pip=True)
        _success("Virtual environment created")
    else:
        _warn("Virtual environment already exists")
    _info("Upgrading pip...")
    subprocess.check_call([str(_python_bin()), "-m", "pip", "install", "--upgrade", "pip"])


def install_dependencies():
    _info("Installing Python dependencies...")
    reqs = ROOT / "requirements.txt"
    if reqs.exists():
        subprocess.check_call([str(_python_bin()), "-m", "pip", "install", "-r", str(reqs)])
        _success("Python dependencies installed")
    else:
        _warn("requirements.txt not found, skipping Python dependencies")

    npm = os.environ.get("NPM_CMD") or ("npm.cmd" if os.name == "nt" else "npm")
    if subprocess.run([npm, "--version"], capture_output=True).returncode == 0:
        _info("Installing Node.js dependencies...")
        subprocess.check_call([npm, "install"], cwd=str(ROOT))
        _success("Node.js dependencies installed")
    else:
        _warn("npm not found, skipping Node.js dependencies")


def create_directories():
    _info("Creating necessary directories...")
    for d in ("outputs", "logs", "data", "templates"):
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    _success("Directories created")


ENV_TEMPLATE = """# Local Researcher Environment Variables (v2.0 - 8대 혁신)
# Copy this file to .env and configure your API keys

# OpenRouter Configuration (필수)
OPENROUTER_API_KEY=your_openrouter_api_key_here

# LLM Configuration (Gemini 2.5 Flash Lite 우선)
LLM_PROVIDER=openrouter
LLM_MODEL=google/gemini-3.1-flash-lite-preview
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=4000

# Multi-Model Orchestration (Gemini 2.5 Flash Lite + Pro)
LLM_FALLBACK_MODEL=google/gemini-2.5-pro
"""
