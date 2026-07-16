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

    # 1. Clean previous builds
    if os.path.exists("dist"):
        shutil.rmtree("dist")
    if os.path.exists("build"):
        shutil.rmtree("build")

    # 2. Build Wheel
    print("Building wheel...")
    run_command(["uv", "build", "--wheel"])

    # 3. Package Binary (using PyInstaller as standard)
    print(f"Packaging binary for {args.platform}...")
    # Assuming entry point is src/cli/entry.py
    cmd = [
        "uv", "run", "pyinstaller",
        "--onefile",
        "--name", f"sparkleforge-{args.platform}",
        "src/cli/entry.py"
    ]
    run_command(cmd)

    print(f"Successfully packaged SparkleForge for {args.platform}")
    print("Artifacts available in dist/")

if __name__ == "__main__":
    main()

# chmod +x scripts/package_release.py
