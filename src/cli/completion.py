"""
자동완성 시스템

prompt_toolkit 기반 자동완성 제공자.
"""

from typing import Iterable, Optional
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document


class SparkleForgeCompleter(Completer):
    """SparkleForge 명령어 자동완성."""
    
    def __init__(self, cli_instance):
        """초기화."""
        self.cli = cli_instance
        self.commands = {
            'research': [],
            'session': ['list', 'show', 'pause', 'resume', 'cancel', 'delete', 'search', 'stats', 'tasks'],
            'context': ['show', 'reload'],
            'checkpoint': ['save', 'list', 'restore', 'delete'],
            'schedule': ['list', 'add', 'remove', 'enable', 'disable'],
            'config': ['show', 'set', 'get'],
            'help': [],
            'exit': [],
            'quit': [],
            'clear': [],
        }
    
    def get_completions(self, document: Document, complete_event) -> Iterable[Completion]:
        """자동완성 제공."""
        text = document.text_before_cursor
        words = text.split()
        
        if not words:
            # 첫 번째 명령어 자동완성
            for cmd in self.commands.keys():
                yield Completion(cmd, start_position=0, display=cmd)
            return
        
        first_word = words[0].lower()
        
        if len(words) == 1:
            # 첫 번째 명령어 자동완성
            for cmd in self.commands.keys():
                if cmd.startswith(first_word):
                    yield Completion(cmd, start_position=-len(first_word), display=cmd)
        
        elif first_word in self.commands:
            # 서브 명령어 자동완성
            subcommands = self.commands[first_word]
            if subcommands:
                if len(words) == 2:
                    current = words[1].lower()
                    for subcmd in subcommands:
                        if subcmd.startswith(current):
                            yield Completion(subcmd, start_position=-len(current), display=subcmd)
                elif len(words) == 3 and first_word == 'session' and words[1] in ['show', 'pause', 'resume', 'cancel', 'delete', 'tasks']:
                    # 세션 ID 자동완성
                    session_ids = self._get_session_ids()
                    current = words[2].lower()
                    for session_id in session_ids:
                        if session_id.lower().startswith(current):
                            yield Completion(session_id, start_position=-len(current), display=session_id)
                elif len(words) == 3 and first_word == 'checkpoint' and words[1] in ['restore', 'delete']:
                    # 체크포인트 ID 자동완성
                    checkpoint_ids = self._get_checkpoint_ids()
                    current = words[2].lower()
                    for checkpoint_id in checkpoint_ids:
                        if checkpoint_id.lower().startswith(current):
                            yield Completion(checkpoint_id, start_position=-len(current), display=checkpoint_id)
                elif len(words) == 3 and first_word == 'schedule' and words[1] in ['remove', 'enable', 'disable']:
                    # 스케줄 ID 자동완성
                    schedule_ids = self._get_schedule_ids()
                    current = words[2].lower()
                    for schedule_id in schedule_ids:
                        if schedule_id.lower().startswith(current):
                            yield Completion(schedule_id, start_position=-len(current), display=schedule_id)
    
    def _get_session_ids(self) -> list:
        """세션 ID 목록 반환 (자동완성용)."""
        if not self.cli.session_control:
            return []
        
        try:
            # 동기적으로 세션 목록 가져오기 (비동기는 나중에 처리)
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 이미 실행 중인 루프가 있으면 빈 리스트 반환
                return []
            else:
                sessions = loop.run_until_complete(
                    self.cli.session_control.search_sessions(limit=50)
                )
                return [s.session_id for s in sessions] if sessions else []
        except Exception:
            return []
    
    def _get_checkpoint_ids(self) -> list:
        """체크포인트 ID 목록 반환 (자동완성용)."""
        if not self.cli.checkpoint_manager:
            return []
        
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return []
            else:
                checkpoints = loop.run_until_complete(
                    self.cli.checkpoint_manager.list_checkpoints()
                )
                return [cp.get("checkpoint_id") for cp in checkpoints if cp.get("checkpoint_id")] if checkpoints else []
        except Exception:
            return []
    
    def _get_schedule_ids(self) -> list:
        """스케줄 ID 목록 반환 (자동완성용)."""
        try:
            from src.core.scheduler import get_scheduler
            import asyncio
            
            scheduler = get_scheduler()
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return []
            else:
                schedules = loop.run_until_complete(scheduler.list_schedules())
                return [s.get("schedule_id") for s in schedules if s.get("schedule_id")] if schedules else []
        except Exception:
            return []
