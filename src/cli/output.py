"""
출력 유틸리티

rich 기반 컬러 출력, 테이블, 진행 바, 패널 등.
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.text import Text
from typing import Optional, List, Dict, Any


class SparkleForgeOutput:
    """SparkleForge 출력 유틸리티."""
    
    def __init__(self, console: Optional[Console] = None):
        """초기화."""
        self.console = console or Console()
    
    def print_success(self, message: str):
        """성공 메시지 출력."""
        self.console.print(f"[green]✅ {message}[/green]")
    
    def print_error(self, message: str):
        """에러 메시지 출력."""
        self.console.print(f"[red]❌ {message}[/red]")
    
    def print_warning(self, message: str):
        """경고 메시지 출력."""
        self.console.print(f"[yellow]⚠️  {message}[/yellow]")
    
    def print_info(self, message: str):
        """정보 메시지 출력."""
        self.console.print(f"[cyan]ℹ️  {message}[/cyan]")
    
    def print_table(self, title: str, headers: List[str], rows: List[List[str]], **kwargs):
        """테이블 출력."""
        table = Table(title=title, show_header=True, header_style="bold cyan", **kwargs)
        
        for header in headers:
            table.add_column(header)
        
        for row in rows:
            table.add_row(*row)
        
        self.console.print(table)
    
    def print_panel(self, content: str, title: str = "", border_style: str = "cyan"):
        """패널 출력."""
        panel = Panel(content, title=title, border_style=border_style)
        self.console.print(panel)
    
    def create_progress(self, description: str = "Processing...", total: Optional[float] = None):
        """진행 바 생성."""
        columns = [
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
        ]
        
        if total is not None:
            columns.extend([
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
            ])
        
        return Progress(*columns, console=self.console)
    
    def print_banner(self, title: str, subtitle: Optional[str] = None):
        """배너 출력."""
        text = Text(title, style="bold cyan")
        if subtitle:
            text.append(f"\n[dim]{subtitle}[/dim]")
        
        panel = Panel(text, border_style="cyan", padding=(1, 2))
        self.console.print(panel)
