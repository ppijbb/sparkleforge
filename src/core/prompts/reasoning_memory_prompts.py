"""Reasoning Memory Prompts

Google Research ReasoningBank에서 영감을 받은 프롬프트들.
성공, 실패, 그리고 병렬 trajectory로부터 일반화 가능한 추론 인사이트(Reasoning Memory)를 추출합니다.
"""

# 성공 궤적 기반 메모리 추출 프롬프트
SUCCESSFUL_TRAJECTORY_PROMPT = """
You are an expert autonomous agent analyst. You will be given a user query and the corresponding trajectory that represents **how an agent successfully accomplished the task**.

## Guidelines
Your goal is to extract and summarize useful insights in the format of memory items based on the agent's successful trajectory.
The goal of summarized memory items is to be helpful and generalizable for future similar tasks.

## Important Notes
- You must first reflect on *why* the trajectory was successful, and then summarize the insights.
- Extract **at most 3** memory items from the trajectory.
- Do not repeat similar or overlapping items.
- Prefer concrete, actionable procedures and reasoning patterns over abstract principles.
- **Do not** embed specific names, exact queries, or literal string contents from this specific task. Make it generalizable.

## Output Format
Your output must strictly follow the Markdown format shown below:

```markdown
# Memory Item 1
## Title <the title of the memory item>
## Description <one sentence summary describing when or when NOT to use the memory item>
## Content <1-3 sentences describing the reasoning insights learned to successfully accomplish similar tasks in the future>

# Memory Item 2
...
```
"""

# 실패 궤적 기반 메모리 추출 프롬프트 (회피 전략)
FAILED_TRAJECTORY_PROMPT = """
You are an expert autonomous agent analyst. You will be given a user query and the corresponding trajectory that represents **how an agent attempted to resolve the task but failed**.

## Guidelines
Your goal is to extract and summarize useful insights in the format of memory items based on the agent's failed trajectory.
The goal of summarized memory items is to be helpful and generalizable for future similar tasks to avoid repeating the same mistakes.

## Important Notes
- You must first reflect on *why* the trajectory failed, and then summarize what lessons you have learned or strategies to prevent the failure in the future.
- Extract **at most 3** memory items from the trajectory.
- Do not repeat similar or overlapping items.
- Prefer concrete, actionable recovery procedures and alternative reasoning paths over abstract principles.
- **Do not** embed specific names, exact queries, or literal string contents from this specific task. Make it generalizable.

## Output Format
Your output must strictly follow the Markdown format shown below:

```markdown
# Memory Item 1
## Title <the title of the memory item>
## Description <one sentence summary describing when or when NOT to use the memory item>
## Content <1-3 sentences describing the reasoning insights learned to avoid such failures and successfully accomplish similar tasks in the future>

# Memory Item 2
...
```
"""

# 병렬 궤적 대조(Self-Contrast) 기반 메모리 추출 프롬프트
PARALLEL_CONTRAST_PROMPT = """
You are an expert autonomous agent analyst. You will be given a user query and multiple trajectories showing how an agent attempted the task in parallel. 
Some trajectories may be successful, and others may have failed.

## Guidelines
Your goal is to **compare and contrast** these trajectories to identify the most useful and generalizable strategies as memory items.
Use **self-contrast reasoning**:
  - Identify reasoning patterns and strategies that consistently led to success.
  - Identify mistakes or inefficiencies from failed trajectories and formulate preventative strategies.
  - Prefer strategies that generalize beyond specific scenarios or exact wording.

## Important Notes
- Think first: Why did some trajectories succeed while others failed? What was the critical difference in reasoning or actions?
- Extract **at most 5** memory items from all trajectories combined.
- Do not repeat similar or overlapping items.
- **Do not** embed specific names, exact queries, or literal string contents from this specific task. Focus on generalizable behaviors and reasoning patterns.
- Make sure each memory item captures **actionable** and **transferable** insights.

## Output Format
Your output must strictly follow the Markdown format shown below:

```markdown
# Memory Item 1
## Title <the title of the memory item>
## Description <one sentence summary describing when or when NOT to use the memory item>
## Content <1-5 sentences describing the insights learned to avoid failures and successfully accomplish similar tasks in the future>

# Memory Item 2
...
```
"""

# 메모리 주입 템플릿
MEMORY_INJECTION_TEMPLATE = """
## Relevant Past Experiences (Reasoning Memory)
The following are generalized insights from past successful or failed attempts at similar tasks. 
Use these insights to guide your reasoning and avoid known pitfalls:

{memories}
"""

# 자동 평가(LLM-as-a-judge) 프롬프트
TRAJECTORY_EVALUATION_PROMPT = """
You are an expert evaluator. You will be given a user query and a trajectory of actions taken by an agent to fulfill that query.

Evaluate whether the agent successfully accomplished the task.

## Guidelines
- Analyze the user query to understand the core objective.
- Follow the agent's reasoning (Think) and actions to see what it actually did.
- Look at the final result or observation to determine if the objective was met.

Respond strictly in the following JSON format:
{
    "thoughts": "Brief reasoning explaining why the trajectory was successful or not.",
    "status": "success" | "fail"
}
"""
