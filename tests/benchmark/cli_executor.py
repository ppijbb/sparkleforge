"""
SparkleForge CLI Executor

Handles subprocess execution of SparkleForge CLI for benchmark testing.
Provides realistic testing by calling the actual CLI rather than importing modules directly.
"""

import subprocess
import json
import time
import os
import tempfile
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import shutil

logger = logging.getLogger(__name__)


@dataclass
class CLIResult:
    """Result of CLI execution."""
    success: bool
    execution_time: float
    return_code: int
    stdout: str
    stderr: str
    output_file: Optional[str] = None
    parsed_output: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class CLIExecutor:
    """Executes SparkleForge CLI commands and parses results."""
    
    def __init__(self, project_root: str, timeout: int = 300):
        self.project_root = Path(project_root)
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
        
        # Ensure we're in the right directory
        if not (self.project_root / "main.py").exists():
            raise ValueError(f"main.py not found in {self.project_root}")
    
    def execute_research(self, query: str, output_dir: Optional[str] = None) -> CLIResult:
        """Execute a research query via CLI - NO FALLBACK ALLOWED."""
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="sparkleforge_benchmark_")
        
        output_path = Path(output_dir) / f"research_result_{int(time.time())}.json"
        
        # Execute actual CLI - failure is failure, no fallback
        result = self._try_actual_cli_execution(query, output_path)
        if not result.success:
            self.logger.error(f"CLI execution failed: {result.error_message}")
            # Return failed result instead of raising exception
            return result
        
        return result
    
    def _try_actual_cli_execution(self, query: str, output_path: Path) -> CLIResult:
        """Try to execute actual CLI command."""
        # Prepare CLI command
        cmd = [
            "python3", "main.py",
            "--request", query,
            "--output", str(output_path),
            "--format", "json"
        ]
        
        self.logger.info(f"Attempting actual CLI execution: {' '.join(cmd)}")
        
        start_time = time.time()
        
        try:
            # Execute command
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            execution_time = time.time() - start_time
            
            # Parse output if file was created
            parsed_output = None
            if output_path.exists():
                try:
                    with open(output_path, 'r', encoding='utf-8') as f:
                        parsed_output = json.load(f)
                except (json.JSONDecodeError, IOError) as e:
                    self.logger.warning(f"Failed to parse output file: {e}")
            
            return CLIResult(
                success=result.returncode == 0,
                execution_time=execution_time,
                return_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                output_file=str(output_path) if output_path.exists() else None,
                parsed_output=parsed_output,
                error_message=result.stderr if result.returncode != 0 else None
            )
            
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            return CLIResult(
                success=False,
                execution_time=execution_time,
                return_code=-1,
                stdout="",
                stderr=f"Command timed out after {self.timeout} seconds",
                error_message=f"Timeout after {self.timeout}s"
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return CLIResult(
                success=False,
                execution_time=execution_time,
                return_code=-1,
                stdout="",
                stderr=str(e),
                error_message=str(e)
            )
    
    
    def execute_with_streaming(self, query: str, output_dir: Optional[str] = None) -> CLIResult:
        """Execute research with streaming enabled."""
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="sparkleforge_benchmark_")
        
        output_path = Path(output_dir) / f"streaming_result_{int(time.time())}.json"
        
        # Prepare CLI command with streaming
        cmd = [
            "python3", "main.py",
            "--request", query,
            "--output", str(output_path),
            "--format", "json",
            "--streaming"
        ]
        
        self.logger.info(f"Executing streaming CLI command: {' '.join(cmd)}")
        
        start_time = time.time()
        
        try:
            # Execute command
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            execution_time = time.time() - start_time
            
            # Parse output
            parsed_output = None
            if output_path.exists():
                try:
                    with open(output_path, 'r', encoding='utf-8') as f:
                        parsed_output = json.load(f)
                except (json.JSONDecodeError, IOError) as e:
                    self.logger.warning(f"Failed to parse streaming output file: {e}")
            
            return CLIResult(
                success=result.returncode == 0,
                execution_time=execution_time,
                return_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                output_file=str(output_path) if output_path.exists() else None,
                parsed_output=parsed_output,
                error_message=result.stderr if result.returncode != 0 else None
            )
            
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            return CLIResult(
                success=False,
                execution_time=execution_time,
                return_code=-1,
                stdout="",
                stderr=f"Streaming command timed out after {self.timeout} seconds",
                error_message=f"Timeout after {self.timeout}s"
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return CLIResult(
                success=False,
                execution_time=execution_time,
                return_code=-1,
                stdout="",
                stderr=str(e),
                error_message=str(e)
            )
    
    def execute_health_check(self) -> CLIResult:
        """Execute health check command."""
        cmd = ["python3", "main.py", "--health-check"]
        
        self.logger.info(f"Executing health check: {' '.join(cmd)}")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=30  # Shorter timeout for health check
            )
            
            execution_time = time.time() - start_time
            
            return CLIResult(
                success=result.returncode == 0,
                execution_time=execution_time,
                return_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                error_message=result.stderr if result.returncode != 0 else None
            )
            
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            return CLIResult(
                success=False,
                execution_time=execution_time,
                return_code=-1,
                stdout="",
                stderr="Health check timed out",
                error_message="Health check timeout"
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return CLIResult(
                success=False,
                execution_time=execution_time,
                return_code=-1,
                stdout="",
                stderr=str(e),
                error_message=str(e)
            )
    
    def parse_output_json(self, output_path: str) -> Optional[Dict[str, Any]]:
        """Parse JSON output file."""
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            self.logger.error(f"Failed to parse JSON output: {e}")
            return None
    
    def extract_metrics_from_output(self, parsed_output: Dict[str, Any]) -> Dict[str, Any]:
        """Extract benchmark-relevant metrics from parsed output."""
        if not parsed_output:
            return {}
        
        metrics = {
            "sources": [],
            "insights": [],
            "execution_results": [],
            "creative_insights": [],
            "similar_research": []
        }
        
        # Extract sources
        if "sources" in parsed_output:
            metrics["sources"] = parsed_output["sources"]
        elif "research_results" in parsed_output and "sources" in parsed_output["research_results"]:
            metrics["sources"] = parsed_output["research_results"]["sources"]
        
        # Extract insights
        if "insights" in parsed_output:
            metrics["insights"] = parsed_output["insights"]
        elif "analysis" in parsed_output and "insights" in parsed_output["analysis"]:
            metrics["insights"] = parsed_output["analysis"]["insights"]
        
        # Extract execution results
        if "execution_results" in parsed_output:
            metrics["execution_results"] = parsed_output["execution_results"]
        elif "workflow" in parsed_output and "results" in parsed_output["workflow"]:
            metrics["execution_results"] = parsed_output["workflow"]["results"]
        
        # Extract creative insights
        if "creative_insights" in parsed_output:
            metrics["creative_insights"] = parsed_output["creative_insights"]
        
        # Extract similar research
        if "similar_research" in parsed_output:
            metrics["similar_research"] = parsed_output["similar_research"]
        
        # Extract workflow information
        if "workflow" in parsed_output:
            workflow = parsed_output["workflow"]
            metrics["workflow_log"] = {
                "handoffs": workflow.get("handoffs", []),
                "communications": workflow.get("communications", []),
                "agent_states": workflow.get("agent_states", [])
            }
        
        return metrics
    
    def cleanup_output_files(self, output_dir: str) -> None:
        """Clean up temporary output files."""
        try:
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
                self.logger.info(f"Cleaned up output directory: {output_dir}")
        except Exception as e:
            self.logger.warning(f"Failed to cleanup output directory {output_dir}: {e}")
    
    def validate_environment(self) -> Tuple[bool, List[str]]:
        """Validate that the environment is ready for benchmarking (development mode)."""
        issues = []
        
        # Check if main.py exists
        if not (self.project_root / "main.py").exists():
            issues.append("main.py not found in project root")
            return False, issues
        
        # Check if Python is available
        try:
            result = subprocess.run(
                ["python3", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                issues.append("Python not available or not working")
        except Exception:
            issues.append("Python not available or not working")
        
        # Check required dependencies - ALL MUST BE AVAILABLE
        try:
            result = subprocess.run(
                ["python3", "-c", "import streamlit, langgraph"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                issues.append("Required dependencies not available - benchmark cannot run")
                return False, issues
        except Exception:
            issues.append("Required dependencies not available - benchmark cannot run")
            return False, issues
        
        # Check OpenRouter API key - REQUIRED
        if not os.getenv("OPENROUTER_API_KEY"):
            issues.append("OPENROUTER_API_KEY not set - benchmark cannot run")
            return False, issues
        
        # All requirements must be met for production benchmarking
        return len(issues) == 0, issues
