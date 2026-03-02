"""CLI entry point for sparkleforge command.

설치 후 어디서든 `sparkleforge` 실행 시 프로젝트 루트를 path/cwd에 넣고 main을 실행.
"""

import os
import sys
from pathlib import Path


def main_entry():
    # 프로젝트 루트 = src/cli/entry.py 기준으로 2단계 상위
    project_root = Path(__file__).resolve().parent.parent.parent
    os.chdir(project_root)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    # main 모듈은 프로젝트 루트의 main.py
    import main
    main.main_entry()
