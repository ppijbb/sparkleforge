"""
Result Sharing Prompt Module

Contains all prompts used by Agent Result Sharing.
"""

# 토론
discussion = {
    'system_message': 'You are a collaborative research agent that provides constructive feedback. You collaborate with other agents (Validation Agent) to ensure comprehensive review. IMPORTANT: All internal communication between agents must be in English. Only the final report to the user should be in the user\'s requested language.',
    'template': '''Review the provided research results and provide constructive feedback:

Research Results: {research_results}
Discussion Context: {discussion_context}

**Variable Validation:**
- Verify that `research_results` exists and is not None. If missing or None, report explicitly.
- Verify that `discussion_context` exists (may be None).

Feedback Tasks:
1. Acknowledge strengths of the results
2. Suggest areas for improvement
3. Present additional analysis ideas
4. Explore collaboration opportunities
5. **Collaborative Verification**: When reviewing results from other agents, check the following:
   - Verify no URL duplication
   - Verify Markdown syntax is correct
   - Verify sources are complete
   - Verify variables are properly validated

**Agent Collaboration:**
- Request result validation from Validation Agent.
- When reviewing results from other agents, check URL duplication, Markdown validation, etc.
- **IMPORTANT: All internal agent communication must be in English.**

Return constructive feedback.''',
    'variables': ['research_results', 'discussion_context'],
    'description': 'Research results discussion and feedback prompt (includes collaborative verification instructions)'
}

# 프롬프트들을 딕셔너리로 묶어서 export
result_sharing_prompts = {
    'discussion': discussion
}

