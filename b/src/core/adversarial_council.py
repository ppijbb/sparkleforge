"""Multi-Agent Adversarial Red-Team Council."""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from src.core.llm_manager.entry import query_model

logger = logging.getLogger(__name__)

RED_TEAM_PERSONAS = [
    {"name": "LogicalFallacyHunter", "focus": "logical fallacies, circular reasoning, false dichotomies"},
    {"name": "EdgeCaseAuditor", "focus": "edge-case vulnerabilities, boundary conditions, failure modes"},
    {"name": "AssumptionInterrogator", "focus": "unstated assumptions, implicit dependencies, hidden preconditions"},
]

def _build_critique_prompt(user_query: str, synthesis_text: str, persona: Dict[str, str]) -> str:
    return f"""You are a member of an adversarial Red Team council.
Persona: {persona["name"]}
Focus: {persona["focus"]}

Original question: {user_query}
Synthesized answer: {synthesis_text}

Identify serious problems in your focus area. Respond as STRICT JSON:
{{
  "persona": "{persona["name"]}",
  "findings": [{"severity": "critical|high|medium|low", "flaw": "...", "explanation": "...", "mitigation": "..."}],
  "overall_risk": "critical|high|medium|low|none",
  "summary": "..."
}}"""

async def run_red_team_council(
    user_query: str,
    synthesis_result: Dict[str, Any],
    council_models: List[str],
    timeout: float = 120.0,
) -> Dict[str, Any]:
    synthesis_text = synthesis_result.get("response", "")
    tasks = []
    for i, persona in enumerate(RED_TEAM_PERSONAS):
        model = council_models[i % len(council_models)]
        prompt = _build_critique_prompt(user_query, synthesis_text, persona)
        tasks.append(query_model(model, [{"role": "user", "content": prompt}], timeout=timeout))

    results = []
    import asyncio
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    
    for i, resp in enumerate(responses):
        if isinstance(resp, Exception):
            continue
        text = resp.get("content", "")
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                results.append(json.loads(match.group(0)))
            except:
                pass
    
    return {"critiques": results, "overall_risk": "high" if results else "none"}

diff --git a/src/core/llm_council.py b/src/core/llm_council.py
index 8a1b2c3..9d4e5f6 100644
--- a/src/core/llm_council.py
+++ b/src/core/llm_council.py
@@ -14,6 +14,7 @@ from typing import Any, Dict, List, Tuple
 
 import httpx
 
+from src.core.adversarial_council import run_red_team_council
 from src.core.researcher_config import get_council_config
 
 logger = logging.getLogger(__name__)
@@ -456,6 +457,11 @@ async def run_full_council(
         timeout,
     )
 
+    # Stage 4: Multi-agent adversarial red-team critique
+    red_team_critique = await run_red_team_council(
+        user_query, stage3_result, council_models, timeout
+    )
+
     # 메타데이터 준비
     metadata = {
         "label_to_model": label_to_model,
         "aggregate_rankings": aggregate_rankings,
+        "red_team_critique": red_team_critique,
     }
 
     logger.info("Full council process completed successfully")
