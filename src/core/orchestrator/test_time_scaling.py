"""Test-Time Scaling Engine with Best-of-N and Parallel-Distill-Refine (PDR).

Implements 2026 TTS techniques for closing pass@1 gap on small/flash-lite models:
- Best-of-N parallel rollouts with diverse seeds/models
- Hard external verification (ruff, syntax, pytest) as deterministic selector
- Parallel-Distill-Refine (PDR) synthesis when no single candidate passes all tests
"""

import asyncio
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CandidateStatus(Enum):
    """Status of a candidate patch."""

    PENDING = "pending"
    GENERATING = "generating"
    VERIFYING = "verifying"
    PASSED = "passed"
    FAILED_LINT = "failed_lint"
    FAILED_SYNTAX = "failed_syntax"
    FAILED_TESTS = "failed_tests"
    ERROR = "error"


class VerificationResult(BaseModel):
    """Result of hard verification on a candidate."""

    candidate_id: str
    passed_lint: bool = False
    passed_syntax: bool = False
    passed_tests: bool = False
    test_results: Dict[str, Any] = Field(default_factory=dict)
    lint_errors: List[str] = Field(default_factory=list)
    syntax_errors: List[str] = Field(default_factory=list)
    test_failures: List[str] = Field(default_factory=list)
    execution_time: float = 0.0

    @property
    def all_passed(self) -> bool:
        return self.passed_lint and self.passed_syntax and self.passed_tests

    @property
    def passed_test_names(self) -> List[str]:
        return [name for name, result in self.test_results.items() if result.get("passed", False)]

    @property
    def failed_test_names(self) -> List[str]:
        return [name for name, result in self.test_results.items() if not result.get("passed", False)]


class CandidatePatch(BaseModel):
    """A candidate patch with its verification results."""

    candidate_id: str
    patch_content: str
    seed: int
    model: str
    status: CandidateStatus = CandidateStatus.PENDING
    verification: Optional[VerificationResult] = None
    worktree_path: Optional[str] = None
    generation_time: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PDRResult(BaseModel):
    """Result of Parallel-Distill-Refine synthesis."""

    success: bool
    synthesized_patch: str = ""
    verification: Optional[VerificationResult] = None
    analysis: Dict[str, Any] = Field(default_factory=dict)
    source_candidates: List[str] = Field(default_factory=list)


class BestOfNConfig(BaseModel):
    """Configuration for Best-of-N runner."""

    num_candidates: int = Field(default=5, ge=1, le=10, description="Number of parallel candidates")
    timeout_seconds: float = Field(default=180.0, description="Timeout per candidate")
    seeds: List[int] = Field(default_factory=list, description="Random seeds for diversity")
    models: List[str] = Field(default_factory=list, description="Models to use for generation")
    temperature_range: Tuple[float, float] = Field(default=(0.3, 1.0), description="Temperature range for diversity")
    use_worktrees: bool = Field(default=True, description="Use git worktrees for isolation")
    verify_lint: bool = Field(default=True, description="Run ruff lint check")
    verify_syntax: bool = Field(default=True, description="Run syntax check")
    verify_tests: bool = Field(default=True, description="Run pytest reproduction tests")
    test_command: str = Field(default="pytest -x -v --tb=short", description="Test command to run")
    lint_command: str = Field(default="ruff check", description="Lint command to run")
    pdr_enabled: bool = Field(default=True, description="Enable PDR synthesis on failure")
    pdr_model: str = Field(default="gemini-2.5-flash", description="Model for PDR synthesis")


class BestOfNRunner:
    """Best-of-N parallel rollout runner with hard verification and PDR synthesis.

    Generates N candidate patches in parallel, verifies each with deterministic
    external tools (ruff, syntax check, pytest), and synthesizes a combined
    patch via PDR when no single candidate passes all tests.
    """

    def __init__(
        self,
        config: Optional[BestOfNConfig] = None,
        patch_generator: Optional[Callable] = None,
        llm_client: Optional[Callable] = None,
    ):
        self.config = config or BestOfNConfig()
        self.patch_generator = patch_generator
        self.llm_client = llm_client
        self._temp_dirs: List[Path] = []
        self._worktrees: List[str] = []

        # Generate seeds if not provided
        if not self.config.seeds:
            import random
            self.config.seeds = [random.randint(0, 2**32 - 1) for _ in range(self.config.num_candidates)]

        logger.info(
            f"BestOfNRunner initialized: candidates={self.config.num_candidates}, "
            f"seeds={self.config.seeds[:3]}..., pdr_enabled={self.config.pdr_enabled}"
        )

    async def run(
        self,
        issue_context: str,
        repo_path: Path,
        base_commit: str = "HEAD",
    ) -> Tuple[Optional[CandidatePatch], List[CandidatePatch], Optional[PDRResult]]:
        """Run Best-of-N with verification and optional PDR.

        Args:
            issue_context: Description of the issue/task to solve
            repo_path: Path to the repository
            base_commit: Base commit to generate patches from

        Returns:
            Tuple of (best_candidate, all_candidates, pdr_result)
            - best_candidate: The winning candidate (or PDR synthesis if successful)
            - all_candidates: All generated candidates with verification results
            - pdr_result: PDR synthesis result if triggered, else None
        """
        repo_path = Path(repo_path).resolve()
        candidates = []

        # Generate candidates in parallel
        logger.info(f"Generating {self.config.num_candidates} candidates in parallel...")
        generation_tasks = [
            self._generate_candidate(
                issue_context=issue_context,
                repo_path=repo_path,
                base_commit=base_commit,
                candidate_idx=i,
                seed=self.config.seeds[i] if i < len(self.config.seeds) else None,
                model=self.config.models[i] if i < len(self.config.models) else None,
            )
            for i in range(self.config.num_candidates)
        ]

        candidates = await asyncio.gather(*generation_tasks, return_exceptions=True)

        # Handle exceptions
        valid_candidates: List[CandidatePatch] = []
        for i, c in enumerate(candidates):
            if isinstance(c, Exception):
                logger.error(f"Candidate {i} generation failed: {c}")
                valid_candidates.append(CandidatePatch(
                    candidate_id=f"cand_{i}",
                    patch_content="",
                    seed=self.config.seeds[i] if i < len(self.config.seeds) else 0,
                    model=self.config.models[i] if i < len(self.config.models) else "unknown",
                    status=CandidateStatus.ERROR,
                    metadata={"error": str(c)},
                ))
            else:
                valid_candidates.append(c)

        candidates = valid_candidates

        # Verify each candidate in parallel
        logger.info(f"Verifying {len(candidates)} candidates...")
        verification_tasks = [
            self._verify_candidate(candidate, repo_path, base_commit)
            for candidate in candidates
        ]
        verified_candidates = await asyncio.gather(*verification_tasks, return_exceptions=True)

        # Handle verification exceptions
        final_candidates: List[CandidatePatch] = []
        for i, c in enumerate(verified_candidates):
            if isinstance(c, Exception):
                logger.error(f"Candidate {i} verification failed: {c}")
                candidates[i].status = CandidateStatus.ERROR
                candidates[i].verification = VerificationResult(
                    candidate_id=candidates[i].candidate_id,
                    test_failures=[str(c)],
                )
                final_candidates.append(candidates[i])
            else:
                final_candidates.append(c)

        candidates = final_candidates

        # Find best candidate (first one that passes all checks)
        best_candidate = None
        for c in candidates:
            if c.verification and c.verification.all_passed:
                best_candidate = c
                c.status = CandidateStatus.PASSED
                logger.info(f"Candidate {c.candidate_id} passed all verification!")
                break

        # If no candidate passes all, try PDR synthesis
        pdr_result = None
        if best_candidate is None and self.config.pdr_enabled:
            logger.info("No candidate passed all checks. Running PDR synthesis...")
            pdr_result = await self._run_pdr_synthesis(candidates, issue_context, repo_path, base_commit)
            if pdr_result.success and pdr_result.verification and pdr_result.verification.all_passed:
                best_candidate = CandidatePatch(
                    candidate_id="pdr_synthesized",
                    patch_content=pdr_result.synthesized_patch,
                    seed=-1,
                    model=self.config.pdr_model,
                    status=CandidateStatus.PASSED,
                    verification=pdr_result.verification,
                    metadata={"pdr_synthesis": True, "source_candidates": pdr_result.source_candidates},
                )
                logger.info("PDR synthesis produced a passing patch!")
            else:
                logger.warning("PDR synthesis did not produce a passing patch.")

        # Cleanup
        await self._cleanup()

        return best_candidate, candidates, pdr_result

    async def _generate_candidate(
        self,
        issue_context: str,
        repo_path: Path,
        base_commit: str,
        candidate_idx: int,
        seed: Optional[int] = None,
        model: Optional[str] = None,
    ) -> CandidatePatch:
        """Generate a single candidate patch."""
        candidate_id = f"cand_{candidate_idx}"
        start_time = time.time()

        worktree_path = None
        temp_dir = None

        try:
            if self.config.use_worktrees:
                worktree_path = await self._create_worktree(repo_path, base_commit, candidate_id)
                work_dir = Path(worktree_path)
            else:
                temp_dir = Path(tempfile.mkdtemp(prefix=f"sparkleforge_{candidate_id}_"))
                self._temp_dirs.append(temp_dir)
                # Copy repo to temp dir
                await self._copy_repo(repo_path, temp_dir, base_commit)
                work_dir = temp_dir

            # Generate patch using provided generator or default
            if self.patch_generator:
                patch_content = await self.patch_generator(
                    issue_context=issue_context,
                    repo_path=work_dir,
                    seed=seed,
                    model=model,
                    temperature=self._get_temperature(candidate_idx),
                )
            else:
                patch_content = await self._default_patch_generation(
                    issue_context, work_dir, seed, model
                )

            generation_time = time.time() - start_time

            return CandidatePatch(
                candidate_id=candidate_id,
                patch_content=patch_content,
                seed=seed or 0,
                model=model or "default",
                status=CandidateStatus.GENERATING,
                worktree_path=str(work_dir) if work_dir else None,
                generation_time=generation_time,
                metadata={"temperature": self._get_temperature(candidate_idx)},
            )

        except Exception as e:
            logger.error(f"Candidate {candidate_id} generation error: {e}")
            return CandidatePatch(
                candidate_id=candidate_id,
                patch_content="",
                seed=seed or 0,
                model=model or "default",
                status=CandidateStatus.ERROR,
                generation_time=time.time() - start_time,
                metadata={"error": str(e)},
            )

    async def _verify_candidate(
        self,
        candidate: CandidatePatch,
        repo_path: Path,
        base_commit: str,
    ) -> CandidatePatch:
        """Verify a candidate with hard checks: lint, syntax, tests."""
        candidate.status = CandidateStatus.VERIFYING

