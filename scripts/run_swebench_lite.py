"""Generate SWE-bench Lite predictions using SparkleForge's own fix-issue path.

This does not reimplement patch generation: each instance is a real git
checkout of its repo at `base_commit`, and `sparkleforge ci fix-issue`
(src/core/ci/fix_issue.py) -- the same production entrypoint Nightwelding and
`.github/workflows/opencode-auto-fix.yml` use for real GitHub issues -- is
invoked against it with the SWE-bench `problem_statement` as the issue
context. Whatever `opencode.patch` it produces (or nothing, on failure)
becomes that instance's prediction. A failed instance is recorded with an
empty `model_patch` rather than skipped, so the run's resolved/unresolved
counts honestly include it instead of quietly shrinking the sample.

Output is a predictions.jsonl in the schema swebench.harness.run_evaluation
expects (instance_id / model_patch / model_name_or_path) -- evaluation itself
is just the unmodified upstream `python -m swebench.harness.run_evaluation`
CLI, run separately, so there is no custom scoring logic to trust here.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SPARKLEFORGE_ENTRYPOINT = REPO_ROOT / "main.py"

# All 6 psf/requests instances in SWE-bench Lite. Same repo -> their
# environment images share layers, which keeps the first weekly run's Docker
# footprint small on a standard GitHub Actions runner (see docs/ANVIL_PLAN.md
# issue #910 follow-up note on SWE-bench Lite's disk requirements). Widen
# this once the pipeline has proven itself.
DEFAULT_INSTANCE_IDS = [
    "psf__requests-863",
    "psf__requests-1963",
    "psf__requests-2148",
    "psf__requests-2317",
    "psf__requests-2674",
    "psf__requests-3362",
]

DEFAULT_MODEL_NAME = "sparkleforge-nightwelding"
DEFAULT_DATASET_NAME = "princeton-nlp/SWE-bench_Lite"


def _load_instances(dataset_name: str, instance_ids: list[str]) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split="test")
    wanted = set(instance_ids)
    by_id = {row["instance_id"]: row for row in ds if row["instance_id"] in wanted}
    missing = wanted - set(by_id)
    if missing:
        raise SystemExit(f"Instance ids not found in {dataset_name}: {sorted(missing)}")
    return [by_id[i] for i in instance_ids]


def _clone_repo(repo: str, dest: Path) -> None:
    url = f"https://github.com/{repo}.git"
    subprocess.run(["git", "clone", "--quiet", url, str(dest)], check=True)


def _reset_to_commit(checkout: Path, commit: str) -> None:
    subprocess.run(
        ["git", "checkout", "--force", commit], cwd=checkout, check=True, capture_output=True
    )
    subprocess.run(["git", "clean", "-fdx"], cwd=checkout, check=True, capture_output=True)


def _generate_patch(checkout: Path, problem_statement: str, timeout: int) -> str:
    fd, context_path_str = tempfile.mkstemp(suffix=".md")
    context_path = Path(context_path_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(problem_statement)

        try:
            subprocess.run(
                [
                    sys.executable,
                    str(SPARKLEFORGE_ENTRYPOINT),
                    "ci",
                    "fix-issue",
                    "--issue-context",
                    str(context_path),
                ],
                cwd=checkout,
                env=dict(os.environ),
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            print(f"  patch generation timed out after {timeout}s", file=sys.stderr)
    finally:
        context_path.unlink(missing_ok=True)

    patch_path = checkout / "opencode.patch"
    if patch_path.exists():
        return patch_path.read_text(encoding="utf-8")
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--instance-ids", nargs="+", default=DEFAULT_INSTANCE_IDS)
    parser.add_argument("--predictions-out", default="swebench_predictions.jsonl")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--per-instance-timeout",
        type=int,
        default=1200,
        help="Seconds allowed for patch generation per instance",
    )
    parser.add_argument(
        "--checkouts-dir",
        default=None,
        help="Reuse an existing directory for repo checkouts instead of a fresh temp dir "
        "(useful for local debugging; CI should leave this unset)",
    )
    args = parser.parse_args()

    instances = _load_instances(args.dataset_name, args.instance_ids)

    checkouts_root = Path(args.checkouts_dir) if args.checkouts_dir else None
    cleanup_tmp: tempfile.TemporaryDirectory | None = None
    if checkouts_root is None:
        cleanup_tmp = tempfile.TemporaryDirectory(prefix="swebench-checkouts-")
        checkouts_root = Path(cleanup_tmp.name)
    else:
        checkouts_root.mkdir(parents=True, exist_ok=True)

    predictions = []
    try:
        repo_dirs: dict[str, Path] = {}
        for instance in instances:
            instance_id = instance["instance_id"]
            repo = instance["repo"]
            print(f"=== {instance_id} ({repo} @ {instance['base_commit'][:10]}) ===")

            if repo not in repo_dirs:
                repo_dir = checkouts_root / repo.replace("/", "__")
                if not repo_dir.exists():
                    _clone_repo(repo, repo_dir)
                repo_dirs[repo] = repo_dir
            checkout = repo_dirs[repo]

            _reset_to_commit(checkout, instance["base_commit"])
            patch = _generate_patch(checkout, instance["problem_statement"], args.per_instance_timeout)

            status = "patch generated" if patch.strip() else "no patch (counts as unresolved)"
            print(f"  -> {status}")

            predictions.append(
                {
                    "instance_id": instance_id,
                    "model_patch": patch,
                    "model_name_or_path": args.model_name,
                }
            )
    finally:
        if cleanup_tmp is not None:
            cleanup_tmp.cleanup()

    out_path = Path(args.predictions_out)
    with out_path.open("w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")

    resolved_attempts = sum(1 for p in predictions if p["model_patch"].strip())
    print(f"Wrote {len(predictions)} predictions to {out_path} ({resolved_attempts} produced a patch)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
