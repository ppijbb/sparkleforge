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

# Multi-Model Orchestration (Gemini 2.5 Flash Lite 우선)
MULTI_MODEL_ENABLED=true
MULTI_MODEL_PRIMARY=google/gemini-3.1-flash-lite-preview
MULTI_MODEL_FALLBACK=openai/gpt-4o-mini

# Search Configuration
TAVILY_API_KEY=your_tavily_api_key_here
EXA_API_KEY=your_exa_api_key_here

# Google Configuration
GOOGLE_API_KEY=your_google_api_key_here

# MCP Configuration
MCP_BUILDER_ENABLED=true
MCP_ALLOWED_SERVERS=local-search,search,web-fetch,arxiv

# Storage Configuration
STORAGE_PATH=./data
CHROMADB_ENABLED=false

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=./logs/sparkleforge.log
"""


def create_env_file():
    env_file = ROOT / ".env"
    if env_file.exists():
        _warn(".env file already exists, skipping creation")
        return
    _info("Creating .env file from template...")
    env_file.write_text(ENV_TEMPLATE, encoding="utf-8")
    _success(".env file created")


def run_sparkleforge(args):
    _info("Starting SparkleForge...")
    cmd = [str(_python_bin()), "-m", "src.cli.entry"] + args
    subprocess.check_call(cmd, cwd=str(ROOT))


def main():
    if not check_python_version():
        sys.exit(1)

    args = sys.argv[1:]
    setup_only = "--setup-only" in args
    if setup_only:
        args.remove("--setup-only")

    setup_venv()
    install_dependencies()
    create_directories()
    create_env_file()

    if setup_only:
        _success("Setup complete. Edit .env with your API keys, then run without --setup-only.")
        return

    run_sparkleforge(args)


if __name__ == "__main__":
    main()
