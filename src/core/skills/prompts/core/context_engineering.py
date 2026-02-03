"""
Context Engineering Prompt Module

Contains all prompts used by Context Engineer.
"""

# 시스템 프롬프트
system_prompt = {
    'template': '''{SYSTEM_PROMPT}''',
    'variables': ['SYSTEM_PROMPT'],
    'description': 'System prompt template'
}

# 프롬프트들을 딕셔너리로 묶어서 export
context_engineering_prompts = {
    'system_prompt': system_prompt
}

