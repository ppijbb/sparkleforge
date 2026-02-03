"""
Checkpoint Manager (완전 자동형 SparkleForge)

대화 상태 저장, 파일 스냅샷 저장, 체크포인트 복원, 체크포인트 목록 관리 기능 제공.
gemini-cli의 checkpointing 기능을 참고하여 구현.
"""

import asyncio
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class Checkpoint:
    """체크포인트 정보."""
    checkpoint_id: str
    timestamp: str
    state: Dict[str, Any]
    file_snapshots: Dict[str, str]  # file_path -> content_hash
    metadata: Dict[str, Any]


class CheckpointManager:
    """체크포인팅 시스템."""
    
    def __init__(self, checkpoint_dir: Optional[Path] = None):
        """
        초기화.
        
        Args:
            checkpoint_dir: 체크포인트 저장 디렉토리 (None이면 기본 디렉토리)
        """
        if checkpoint_dir is None:
            checkpoint_dir = Path.home() / ".sparkleforge" / "checkpoints"
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.snapshots_dir = self.checkpoint_dir / "snapshots"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"CheckpointManager initialized: {self.checkpoint_dir}")
    
    async def save_checkpoint(
        self,
        state: Dict[str, Any],
        file_snapshots: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        체크포인트 저장.
        
        Args:
            state: 상태 딕셔너리
            file_snapshots: 파일 스냅샷 (file_path -> content)
            metadata: 추가 메타데이터
        
        Returns:
            체크포인트 ID
        """
        checkpoint_id = f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(str(state).encode()).hexdigest()[:8]}"
        timestamp = datetime.now().isoformat()
        
        # 파일 스냅샷 저장
        saved_snapshots = {}
        if file_snapshots:
            for file_path, content in file_snapshots.items():
                snapshot_hash = hashlib.md5(content.encode()).hexdigest()
                snapshot_path = self.snapshots_dir / f"{checkpoint_id}_{snapshot_hash}.snapshot"
                snapshot_path.write_text(content, encoding='utf-8')
                saved_snapshots[file_path] = snapshot_hash
        
        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            timestamp=timestamp,
            state=state,
            file_snapshots=saved_snapshots,
            metadata=metadata or {}
        )
        
        # 체크포인트 파일 저장
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.json"
        checkpoint_file.write_text(
            json.dumps(asdict(checkpoint), indent=2, ensure_ascii=False),
            encoding='utf-8'
        )
        
        logger.info(f"Checkpoint saved: {checkpoint_id}")
        return checkpoint_id
    
    async def load_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """
        체크포인트 로드.
        
        Args:
            checkpoint_id: 체크포인트 ID
        
        Returns:
            Checkpoint 객체 또는 None
        """
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.json"
        
        if not checkpoint_file.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_id}")
            return None
        
        try:
            data = json.loads(checkpoint_file.read_text(encoding='utf-8'))
            checkpoint = Checkpoint(**data)
            logger.info(f"Checkpoint loaded: {checkpoint_id}")
            return checkpoint
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
            return None
    
    async def restore_checkpoint(
        self,
        checkpoint_id: str,
        restore_files: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        체크포인트 복원.
        
        Args:
            checkpoint_id: 체크포인트 ID
            restore_files: 파일 스냅샷 복원 여부
        
        Returns:
            복원된 상태 딕셔너리 또는 None
        """
        checkpoint = await self.load_checkpoint(checkpoint_id)
        if not checkpoint:
            return None
        
        # 파일 스냅샷 복원
        if restore_files and checkpoint.file_snapshots:
            for file_path, snapshot_hash in checkpoint.file_snapshots.items():
                snapshot_path = self.snapshots_dir / f"{checkpoint_id}_{snapshot_hash}.snapshot"
                if snapshot_path.exists():
                    content = snapshot_path.read_text(encoding='utf-8')
                    target_path = Path(file_path)
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    target_path.write_text(content, encoding='utf-8')
                    logger.info(f"Restored file: {file_path}")
        
        logger.info(f"Checkpoint restored: {checkpoint_id}")
        return checkpoint.state
    
    async def list_checkpoints(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        체크포인트 목록 조회.
        
        Args:
            limit: 최대 반환 개수
        
        Returns:
            체크포인트 정보 리스트
        """
        checkpoints = []
        
        for checkpoint_file in sorted(
            self.checkpoint_dir.glob("checkpoint_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )[:limit]:
            try:
                data = json.loads(checkpoint_file.read_text(encoding='utf-8'))
                checkpoints.append({
                    'checkpoint_id': data['checkpoint_id'],
                    'timestamp': data['timestamp'],
                    'metadata': data.get('metadata', {})
                })
            except Exception as e:
                logger.warning(f"Failed to read checkpoint {checkpoint_file}: {e}")
        
        return checkpoints
    
    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """
        체크포인트 삭제.
        
        Args:
            checkpoint_id: 체크포인트 ID
        
        Returns:
            삭제 성공 여부
        """
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.json"
        
        if not checkpoint_file.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_id}")
            return False
        
        try:
            # 체크포인트 파일 삭제
            checkpoint_file.unlink()
            
            # 관련 스냅샷 파일 삭제
            for snapshot_file in self.snapshots_dir.glob(f"{checkpoint_id}_*.snapshot"):
                snapshot_file.unlink()
            
            logger.info(f"Checkpoint deleted: {checkpoint_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete checkpoint {checkpoint_id}: {e}")
            return False
    
    async def cleanup_old_checkpoints(self, days: int = 30) -> int:
        """
        오래된 체크포인트 정리.
        
        Args:
            days: 보관 기간 (일)
        
        Returns:
            삭제된 체크포인트 개수
        """
        from datetime import timedelta
        
        cutoff_time = datetime.now() - timedelta(days=days)
        deleted_count = 0
        
        for checkpoint_file in self.checkpoint_dir.glob("checkpoint_*.json"):
            try:
                file_time = datetime.fromtimestamp(checkpoint_file.stat().st_mtime)
                if file_time < cutoff_time:
                    checkpoint_id = checkpoint_file.stem
                    if await self.delete_checkpoint(checkpoint_id):
                        deleted_count += 1
            except Exception as e:
                logger.warning(f"Failed to process checkpoint {checkpoint_file}: {e}")
        
        logger.info(f"Cleaned up {deleted_count} old checkpoints")
        return deleted_count

