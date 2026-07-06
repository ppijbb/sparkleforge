"""Fixture + grader for scenario 3: "이 외장하드에서 작년 출장 영수증만 모아줘."

Seeds a mock external drive with receipts from last year (2025, to be
collected) mixed with this year's receipts (2026, decoys) and unrelated
noise files. Grading looks for copies of the 2025 markers landing in some
new location without any 2026 decoy markers following them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from tests.benchmark.scenario_grading import is_runtime_artifact, judge_score, read_text_safe

RECEIPTS_2025 = {
    "Trips/2025-03-Seoul/receipt_2025-03-12.txt": "RECEIPT_MARKER_2025_SEOUL",
    "Trips/2025-11-Busan/receipt_2025-11-02.txt": "RECEIPT_MARKER_2025_BUSAN",
}
RECEIPTS_2026 = {
    "Trips/2026-02-Tokyo/receipt_2026-02-01.txt": "RECEIPT_MARKER_2026_TOKYO",
}
NOISE_FILES = {
    "Misc/photo_notes.txt": "just some vacation photo captions, not a receipt",
    "Misc/random_notes.txt": "shopping list: milk, eggs, bread",
}


def build(workspace: Path) -> Dict[str, Any]:
    drive = workspace / "external_drive"
    all_seed = {**RECEIPTS_2025, **RECEIPTS_2026, **NOISE_FILES}
    for rel, marker in all_seed.items():
        path = drive / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"Business trip receipt.\nAmount: $482.10\n{marker}\n", encoding="utf-8")

    return {
        "workspace": str(workspace),
        "external_drive": "external_drive",
        "expected_markers": list(RECEIPTS_2025.values()),
        "decoy_markers": list(RECEIPTS_2026.values()),
        "original_paths": {**RECEIPTS_2025, **RECEIPTS_2026},
    }


async def grade(workspace: Path, ctx: Dict[str, Any], stdout: str) -> Dict[str, tuple[float, str]]:
    # original_paths keys are relative to the external_drive dir, not the workspace root
    drive_prefix = ctx["external_drive"]
    original_rel_paths = {f"{drive_prefix}/{rel}" for rel in ctx["original_paths"].keys()}
    collected_locations: dict[str, set[str]] = {m: set() for m in ctx["expected_markers"]}
    leaked_locations: dict[str, set[str]] = {m: set() for m in ctx["decoy_markers"]}

    for path in workspace.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(workspace).as_posix()
        if rel in original_rel_paths or is_runtime_artifact(rel):
            continue  # only count copies/moves elsewhere, not the seed files or agent-runtime noise
        text = read_text_safe(path)
        if not text:
            continue
        for marker in ctx["expected_markers"]:
            if marker in text:
                collected_locations[marker].add(str(Path(rel).parent))
        for marker in ctx["decoy_markers"]:
            if marker in text:
                leaked_locations[marker].add(str(Path(rel).parent))

    found_2025 = [m for m, locs in collected_locations.items() if locs]
    recall = len(found_2025) / len(ctx["expected_markers"]) if ctx["expected_markers"] else 0.0
    recall_score = (recall, f"collected {len(found_2025)}/{len(ctx['expected_markers'])} 2025 receipts")

    leaked = [m for m, locs in leaked_locations.items() if locs]
    if not leaked:
        precision_score = (1.0, "no 2026 decoy receipts leaked into the collection")
    else:
        precision_score = (0.0, f"leaked {len(leaked)} non-2025 receipt(s) into the collection: {leaked}")

    all_dest_dirs = set()
    for locs in collected_locations.values():
        all_dest_dirs |= locs
    if not all_dest_dirs:
        organized_score = (0.0, "no 2025 receipts found anywhere new")
    elif len(all_dest_dirs) == 1:
        organized_score = (1.0, f"all collected receipts consolidated into {all_dest_dirs}")
    else:
        organized_score = (0.5, f"receipts found but scattered across {all_dest_dirs}")

    judge = await judge_score(
        rubric="Does the agent's output clearly explain which receipts were collected and why (2025 trips only)?",
        transcript=stdout[:4000],
        context=f"expected markers: {ctx['expected_markers']}, decoys: {ctx['decoy_markers']}",
    )

    return {
        "recall": recall_score,
        "precision": precision_score,
        "organized": organized_score,
        "judge_explanation": judge,
    }
