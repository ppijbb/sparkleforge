#!/usr/bin/env python3
"""
SparkleForge Bootstrap Shim
"""
import sys
import os

def main_entry():
    project_root = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(project_root, "src")
    if os.path.isdir(src_path) and src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    from src.cli.entry import main_entry as cli_main
    return cli_main()

if __name__ == "__main__":
    sys.exit(main_entry())
