"""
Git 워크플로우 자동화 모듈

claude-code의 commit-commands 플러그인 패턴을 참고하여 구현.
Git 상태 확인, 커밋, 푸시, PR 생성 등을 자동화.
"""

import asyncio
import subprocess
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import re

logger = logging.getLogger(__name__)


class GitWorkflow:
    """Git 워크플로우 자동화 클래스."""
    
    def __init__(self, repo_path: Optional[Path] = None):
        """
        GitWorkflow 초기화.
        
        Args:
            repo_path: Git 저장소 경로 (None이면 현재 디렉토리)
        """
        self.repo_path = repo_path or Path.cwd()
        self.repo_path = self.repo_path.resolve()
        
        # Git 저장소 확인
        if not self._is_git_repo():
            raise ValueError(f"Not a git repository: {self.repo_path}")
    
    def _is_git_repo(self) -> bool:
        """Git 저장소인지 확인."""
        git_dir = self.repo_path / ".git"
        return git_dir.exists() and git_dir.is_dir()
    
    async def _run_git_command(self, *args: str, check: bool = True) -> Dict[str, Any]:
        """
        Git 명령 실행.
        
        Args:
            *args: Git 명령 인자
            check: 실패 시 예외 발생 여부
        
        Returns:
            실행 결과 (stdout, stderr, returncode)
        """
        try:
            process = await asyncio.create_subprocess_exec(
                "git",
                *args,
                cwd=str(self.repo_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            result = {
                "stdout": stdout.decode("utf-8", errors="replace").strip(),
                "stderr": stderr.decode("utf-8", errors="replace").strip(),
                "returncode": process.returncode
            }
            
            if check and result["returncode"] != 0:
                raise subprocess.CalledProcessError(
                    result["returncode"],
                    ["git"] + list(args),
                    result["stdout"],
                    result["stderr"]
                )
            
            return result
        except Exception as e:
            logger.error(f"Git command failed: {' '.join(args)} - {e}")
            raise
    
    async def git_status(self) -> Dict[str, Any]:
        """
        Git 상태 확인.
        
        Returns:
            Git 상태 정보 (branch, changes, etc.)
        """
        try:
            # 현재 브랜치
            branch_result = await self._run_git_command("branch", "--show-current")
            current_branch = branch_result["stdout"]
            
            # 상태 정보
            status_result = await self._run_git_command("status", "--porcelain")
            status_lines = status_result["stdout"].split("\n") if status_result["stdout"] else []
            
            # 변경된 파일 분류
            staged = []
            unstaged = []
            untracked = []
            
            for line in status_lines:
                if not line.strip():
                    continue
                status = line[:2]
                file_path = line[3:]
                
                if status[0] == "?":
                    untracked.append(file_path)
                elif status[0] != " ":
                    staged.append(file_path)
                if status[1] != " ":
                    unstaged.append(file_path)
            
            # 원격 브랜치 정보
            remote_result = await self._run_git_command("remote", "get-url", "origin", check=False)
            remote_url = remote_result["stdout"] if remote_result["returncode"] == 0 else None
            
            return {
                "current_branch": current_branch,
                "staged_files": staged,
                "unstaged_files": unstaged,
                "untracked_files": untracked,
                "has_changes": len(staged) > 0 or len(unstaged) > 0 or len(untracked) > 0,
                "remote_url": remote_url
            }
        except Exception as e:
            logger.error(f"Failed to get git status: {e}")
            return {
                "error": str(e),
                "current_branch": None,
                "staged_files": [],
                "unstaged_files": [],
                "untracked_files": [],
                "has_changes": False
            }
    
    async def _generate_commit_message(self, staged_files: List[str], unstaged_files: List[str]) -> str:
        """
        LLM을 사용하여 커밋 메시지 생성.
        
        Args:
            staged_files: 스테이징된 파일 목록
            unstaged_files: 스테이징되지 않은 파일 목록
        
        Returns:
            생성된 커밋 메시지
        """
        try:
            from src.core.mcp_integration import get_mcp_hub
            mcp_hub = get_mcp_hub()
            
            # 최근 커밋 메시지 가져오기 (스타일 학습용)
            recent_commits_result = await self._run_git_command("log", "--oneline", "-10", check=False)
            recent_commits = recent_commits_result["stdout"].split("\n")[:5] if recent_commits_result["stdout"] else []
            
            # 변경사항 diff 가져오기
            diff_result = await self._run_git_command("diff", "--cached", check=False)
            staged_diff = diff_result["stdout"] if diff_result["stdout"] else ""
            
            unstaged_diff_result = await self._run_git_command("diff", check=False)
            unstaged_diff = unstaged_diff_result["stdout"] if unstaged_diff_result["stdout"] else ""
            
            # LLM에 커밋 메시지 생성 요청
            prompt = f"""Generate a commit message based on the following changes.

Recent commit messages (for style reference):
{chr(10).join(recent_commits[:3])}

Staged changes:
{staged_diff[:2000] if staged_diff else "No staged changes"}

Unstaged changes (for context):
{unstaged_diff[:1000] if unstaged_diff else "No unstaged changes"}

Files changed:
- Staged: {', '.join(staged_files[:10])}
- Unstaged: {', '.join(unstaged_files[:10])}

Generate a concise commit message following conventional commit format (e.g., "feat: add feature", "fix: fix bug", "refactor: refactor code").
Return only the commit message, no additional text."""

            response = await mcp_hub.call_llm_async(
                model="gemini-2.5-flash-lite",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            commit_message = response.get("content", "").strip()
            # 여러 줄인 경우 첫 줄만 사용
            if "\n" in commit_message:
                commit_message = commit_message.split("\n")[0].strip()
            
            # 빈 메시지인 경우 기본 메시지
            if not commit_message:
                commit_message = f"Update {len(staged_files)} files"
            
            return commit_message
        except Exception as e:
            logger.warning(f"Failed to generate commit message with LLM: {e}")
            # LLM 실패 시 기본 메시지
            all_files = staged_files + unstaged_files
            if all_files:
                return f"Update {', '.join(all_files[:3])}"
            return "Update files"
    
    async def git_commit(self, message: Optional[str] = None, auto_stage: bool = True) -> Dict[str, Any]:
        """
        Git 커밋 생성.
        
        Args:
            message: 커밋 메시지 (None이면 자동 생성)
            auto_stage: 자동으로 파일 스테이징 여부
        
        Returns:
            커밋 결과
        """
        try:
            status = await self.git_status()
            
            if not status["has_changes"]:
                return {
                    "success": False,
                    "error": "No changes to commit",
                    "status": status
                }
            
            # 자동 스테이징
            if auto_stage:
                # .env, credentials.json 등 시크릿 파일 제외
                files_to_stage = []
                for file in status["unstaged_files"] + status["untracked_files"]:
                    if not any(secret in file.lower() for secret in [".env", "credentials.json", "secrets"]):
                        files_to_stage.append(file)
                
                if files_to_stage:
                    await self._run_git_command("add", *files_to_stage)
                    # 상태 다시 확인
                    status = await self.git_status()
            
            # 스테이징된 파일이 없으면 실패
            if not status["staged_files"]:
                return {
                    "success": False,
                    "error": "No staged files to commit",
                    "status": status
                }
            
            # 커밋 메시지 생성
            if not message:
                message = await self._generate_commit_message(
                    status["staged_files"],
                    status["unstaged_files"]
                )
            
            # 커밋 실행
            commit_result = await self._run_git_command("commit", "-m", message)
            
            # 커밋 해시 가져오기
            hash_result = await self._run_git_command("rev-parse", "HEAD")
            commit_hash = hash_result["stdout"][:7]
            
            return {
                "success": True,
                "commit_hash": commit_hash,
                "commit_message": message,
                "files_committed": status["staged_files"],
                "output": commit_result["stdout"]
            }
        except Exception as e:
            logger.error(f"Failed to commit: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def git_push(self, branch: Optional[str] = None, force: bool = False) -> Dict[str, Any]:
        """
        Git 브랜치 푸시.
        
        Args:
            branch: 푸시할 브랜치 (None이면 현재 브랜치)
            force: 강제 푸시 여부
        
        Returns:
            푸시 결과
        """
        try:
            if not branch:
                status = await self.git_status()
                branch = status["current_branch"]
            
            if not branch:
                return {
                    "success": False,
                    "error": "No branch specified"
                }
            
            # 원격 저장소 확인
            remote_result = await self._run_git_command("remote", "get-url", "origin", check=False)
            if remote_result["returncode"] != 0:
                return {
                    "success": False,
                    "error": "No remote 'origin' configured"
                }
            
            # 푸시 실행
            push_args = ["push", "origin", branch]
            if force:
                push_args.append("--force")
            
            push_result = await self._run_git_command(*push_args)
            
            return {
                "success": True,
                "branch": branch,
                "output": push_result["stdout"]
            }
        except Exception as e:
            logger.error(f"Failed to push: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def git_create_pr(self, title: str, body: str, base: str = "main") -> Dict[str, Any]:
        """
        Pull Request 생성 (GitHub CLI 사용).
        
        Args:
            title: PR 제목
            body: PR 본문
            base: 베이스 브랜치 (기본값: main)
        
        Returns:
            PR 생성 결과
        """
        try:
            status = await self.git_status()
            current_branch = status["current_branch"]
            
            if not current_branch:
                return {
                    "success": False,
                    "error": "No current branch"
                }
            
            # GitHub CLI 확인
            try:
                gh_check = await asyncio.create_subprocess_exec(
                    "gh", "--version",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await gh_check.communicate()
                if gh_check.returncode != 0:
                    raise FileNotFoundError("gh command not found")
            except FileNotFoundError:
                return {
                    "success": False,
                    "error": "GitHub CLI (gh) not installed. Install from https://cli.github.com/"
                }
            
            # PR 생성
            process = await asyncio.create_subprocess_exec(
                "gh", "pr", "create",
                "--title", title,
                "--body", body,
                "--base", base,
                cwd=str(self.repo_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                return {
                    "success": False,
                    "error": stderr.decode("utf-8", errors="replace").strip(),
                    "output": stdout.decode("utf-8", errors="replace").strip()
                }
            
            pr_url = stdout.decode("utf-8", errors="replace").strip()
            
            return {
                "success": True,
                "pr_url": pr_url,
                "branch": current_branch,
                "base": base,
                "title": title
            }
        except Exception as e:
            logger.error(f"Failed to create PR: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def git_commit_push_pr(
        self,
        commit_message: Optional[str] = None,
        pr_title: Optional[str] = None,
        pr_body: Optional[str] = None,
        base: str = "main"
    ) -> Dict[str, Any]:
        """
        전체 워크플로우: 커밋 + 푸시 + PR 생성.
        
        Args:
            commit_message: 커밋 메시지 (None이면 자동 생성)
            pr_title: PR 제목 (None이면 커밋 메시지 사용)
            pr_body: PR 본문 (None이면 자동 생성)
            base: 베이스 브랜치
        
        Returns:
            전체 워크플로우 결과
        """
        results = {}
        
        # 1. 커밋
        commit_result = await self.git_commit(message=commit_message)
        results["commit"] = commit_result
        
        if not commit_result.get("success"):
            return {
                "success": False,
                "error": "Commit failed",
                "results": results
            }
        
        # 2. 푸시
        push_result = await self.git_push()
        results["push"] = push_result
        
        if not push_result.get("success"):
            return {
                "success": False,
                "error": "Push failed",
                "results": results
            }
        
        # 3. PR 생성
        if not pr_title:
            pr_title = commit_result.get("commit_message", "Update")
        
        if not pr_body:
            # PR 본문 자동 생성
            status = await self.git_status()
            pr_body = f"""## Summary
{commit_result.get('commit_message', 'Update')}

## Changes
- Committed {len(commit_result.get('files_committed', []))} files
- Branch: {status.get('current_branch', 'unknown')}

## Test Plan
- [ ] Verify changes work as expected
- [ ] Check for any breaking changes
- [ ] Review code quality"""
        
        pr_result = await self.git_create_pr(
            title=pr_title,
            body=pr_body,
            base=base
        )
        results["pr"] = pr_result
        
        return {
            "success": pr_result.get("success", False),
            "results": results
        }
