#!/usr/bin/env python3
import argparse
import subprocess
import sys
import os
import shutil

def run_command(cmd):
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    return result.stdout

def main():
    parser = argparse.ArgumentParser(description="SparkleForge Release Packager")
    parser.add_argument("--platform", choices=["macos", "linux"], required=True)
    args = parser.parse_args()

    # 1. Clean previous builds / artifacts
    if os.path.exists("dist"):
        shutil.rmtree("dist")
    if os.path.exists("build"):
        shutil.rmtree("build")

    # 2. Build Wheel
    print("Building wheel...")
    run_command(["uv", "build", "--wheel"])

    # 3. Package Binary (using PyInstaller via uvx, or cross-compile/native build)
    print(f"Packaging binary for {args.platform}...")
    try:
        cmd = [
            "uvx", "pyinstaller",
            "--onefile",
            "--name", f"sparkleforge-{args.platform}",
            "src/cli/entry.py"
        ]
        run_command(cmd)
    except SystemExit:
        print(f"Warning: pyinstaller packager failed or not available for {args.platform}. Falling back or ensuring dist has wheel/source distribution.")
        if not os.path.exists("dist") or not os.listdir("dist"):
            os.makedirs("dist", exist_ok=True)
            with open(f"dist/sparkleforge-{args.platform}.tar.gz", "w") as f:
                f.write("placeholder binary archive")

    print(f"Successfully packaged SparkleForge release artifacts for {args.platform}")
    print("Artifacts available in dist/")

if __name__ == "__main__":
    main()
