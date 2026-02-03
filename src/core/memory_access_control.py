"""
Memory Access Control

백서 요구사항: 사용자/테넌트 레벨 격리, ACL 기반 접근 제어, 사용자 데이터 삭제 옵션
"""

import logging
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class AccessLevel(Enum):
    """접근 레벨."""
    NONE = "none"  # 접근 불가
    READ = "read"  # 읽기만
    WRITE = "write"  # 읽기/쓰기
    ADMIN = "admin"  # 모든 권한 (삭제 포함)


@dataclass
class AccessControlEntry:
    """접근 제어 항목."""
    user_id: str
    memory_id: Optional[str] = None  # None이면 모든 메모리
    access_level: AccessLevel = AccessLevel.READ
    granted_by: Optional[str] = None  # 권한 부여자
    granted_at: Optional[str] = None


@dataclass
class UserMemoryPolicy:
    """사용자 메모리 정책."""
    user_id: str
    enable_memory_generation: bool = True  # 메모리 생성 허용 여부
    enable_memory_sharing: bool = False  # 다른 사용자와 공유 허용 여부
    retention_days: Optional[int] = None  # 보관 기간 (None이면 무제한)
    auto_delete: bool = False  # 자동 삭제 여부


class MemoryAccessControl:
    """
    메모리 접근 제어 시스템.
    
    백서 요구사항:
    - 사용자/테넌트 레벨 격리
    - ACL 기반 접근 제어
    - 사용자 데이터 삭제 옵션
    """
    
    def __init__(self):
        """초기화."""
        # ACL: memory_id -> user_id -> AccessLevel
        self.acls: Dict[str, Dict[str, AccessLevel]] = {}
        
        # 사용자 정책
        self.user_policies: Dict[str, UserMemoryPolicy] = {}
        
        # 기본 정책: 사용자는 자신의 메모리에만 접근
        self.default_policy = UserMemoryPolicy(
            user_id="default",
            enable_memory_generation=True,
            enable_memory_sharing=False
        )
        
        logger.info("MemoryAccessControl initialized")
    
    def check_access(
        self,
        user_id: str,
        memory_id: str,
        required_level: AccessLevel = AccessLevel.READ
    ) -> bool:
        """
        접근 권한 확인.
        
        Args:
            user_id: 사용자 ID
            memory_id: 메모리 ID
            required_level: 필요한 접근 레벨
            
        Returns:
            접근 허용 여부
        """
        # 메모리 소유자 확인 (메모리 ID에서 추출 또는 별도 저장소에서)
        # 간단한 구현: 메모리 ID에 user_id가 포함되어 있다고 가정
        if user_id in memory_id or memory_id.startswith(f"mem_{user_id}_"):
            # 소유자는 모든 권한
            return True
        
        # ACL 확인
        if memory_id in self.acls:
            user_access = self.acls[memory_id].get(user_id, AccessLevel.NONE)
            
            # 접근 레벨 비교
            level_hierarchy = {
                AccessLevel.NONE: 0,
                AccessLevel.READ: 1,
                AccessLevel.WRITE: 2,
                AccessLevel.ADMIN: 3
            }
            
            return level_hierarchy.get(user_access, 0) >= level_hierarchy.get(required_level, 0)
        
        # 기본 정책: 다른 사용자의 메모리는 접근 불가
        return False
    
    def grant_access(
        self,
        memory_id: str,
        user_id: str,
        access_level: AccessLevel,
        granted_by: str
    ) -> bool:
        """
        접근 권한 부여.
        
        Args:
            memory_id: 메모리 ID
            user_id: 사용자 ID
            access_level: 접근 레벨
            granted_by: 권한 부여자
            
        Returns:
            성공 여부
        """
        try:
            if memory_id not in self.acls:
                self.acls[memory_id] = {}
            
            self.acls[memory_id][user_id] = access_level
            
            logger.info(f"Access granted: user {user_id} -> memory {memory_id} ({access_level.value})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to grant access: {e}")
            return False
    
    def revoke_access(self, memory_id: str, user_id: str) -> bool:
        """접근 권한 취소."""
        try:
            if memory_id in self.acls:
                self.acls[memory_id].pop(user_id, None)
                logger.info(f"Access revoked: user {user_id} -> memory {memory_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to revoke access: {e}")
            return False
    
    def set_user_policy(self, policy: UserMemoryPolicy) -> bool:
        """사용자 정책 설정."""
        try:
            self.user_policies[policy.user_id] = policy
            logger.info(f"Policy set for user {policy.user_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to set user policy: {e}")
            return False
    
    def get_user_policy(self, user_id: str) -> UserMemoryPolicy:
        """사용자 정책 조회."""
        return self.user_policies.get(user_id, self.default_policy)
    
    def can_generate_memory(self, user_id: str) -> bool:
        """메모리 생성 허용 여부."""
        policy = self.get_user_policy(user_id)
        return policy.enable_memory_generation
    
    def can_share_memory(self, user_id: str) -> bool:
        """메모리 공유 허용 여부."""
        policy = self.get_user_policy(user_id)
        return policy.enable_memory_sharing
    
    def delete_user_memories(self, user_id: str) -> Dict[str, Any]:
        """
        사용자 메모리 전체 삭제.
        
        백서 요구사항: 사용자 데이터 삭제 옵션
        
        Args:
            user_id: 사용자 ID
            
        Returns:
            삭제 결과
        """
        try:
            deleted_count = 0
            
            # ACL에서 사용자 관련 항목 제거
            for memory_id, acl in list(self.acls.items()):
                if user_id in acl:
                    del acl[user_id]
                    deleted_count += 1
                    if not acl:  # ACL이 비면 전체 제거
                        del self.acls[memory_id]
            
            # 실제 메모리 삭제는 메모리 저장소에서 수행 (여기서는 ACL만 관리)
            logger.info(f"Deleted access for user {user_id}: {deleted_count} entries")
            
            return {
                "user_id": user_id,
                "deleted_acl_entries": deleted_count,
                "message": "User memories marked for deletion. Actual deletion should be performed by memory storage."
            }
            
        except Exception as e:
            logger.error(f"Failed to delete user memories: {e}")
            return {
                "user_id": user_id,
                "error": str(e)
            }
    
    def isolate_user_memories(self, user_id: str) -> bool:
        """
        사용자 메모리 격리 확인.
        
        백서 요구사항: 사용자/테넌트 레벨 격리
        
        Args:
            user_id: 사용자 ID
            
        Returns:
            격리 상태
        """
        # 사용자는 자신의 메모리만 접근 가능한지 확인
        policy = self.get_user_policy(user_id)
        return not policy.enable_memory_sharing  # 공유 비활성화 = 격리 활성화


# 전역 인스턴스
_memory_access_control: Optional[MemoryAccessControl] = None


def get_memory_access_control() -> MemoryAccessControl:
    """전역 메모리 접근 제어 인스턴스 반환."""
    global _memory_access_control
    if _memory_access_control is None:
        _memory_access_control = MemoryAccessControl()
    return _memory_access_control

