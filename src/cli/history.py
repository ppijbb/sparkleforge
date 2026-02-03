"""
히스토리 관리

prompt_toolkit의 FileHistory를 사용하여 히스토리 관리.
추가 기능: 히스토리 검색, 필터링 등.
"""

from pathlib import Path
from prompt_toolkit.history import FileHistory
from typing import Optional


class SparkleForgeHistory:
    """SparkleForge 히스토리 관리."""
    
    def __init__(self, history_file: Optional[Path] = None):
        """초기화."""
        if history_file is None:
            history_file = Path.home() / ".sparkleforge" / "history"
        
        history_file.parent.mkdir(parents=True, exist_ok=True)
        self.history_file = history_file
        self.file_history = FileHistory(str(history_file))
    
    def get_file_history(self) -> FileHistory:
        """FileHistory 인스턴스 반환."""
        return self.file_history
    
    def clear_history(self):
        """히스토리 파일 삭제."""
        if self.history_file.exists():
            self.history_file.unlink()
            # 새로운 FileHistory 인스턴스 생성
            self.file_history = FileHistory(str(self.history_file))
    
    def get_history_file_path(self) -> Path:
        """히스토리 파일 경로 반환."""
        return self.history_file
