"""
Autonomous Orchestrator Prompt Module

Contains all prompts used by Autonomous Orchestrator.
"""

# 분석
analysis = {
    'system_message': 'You are an expert research analyst with comprehensive domain knowledge.',
    'template': '''Analyze and evaluate the following data:

Data: {data}
Analysis Goal: {analysis_goal}

Analysis Tasks:
1. Assess data quality and completeness
2. Identify key patterns and trends
3. Discover potential issues
4. Provide improvement recommendations

Return analysis results in a structured format.''',
    'variables': ['data', 'analysis_goal'],
    'description': 'Data analysis prompt'
}

# 검증
verification = {
    'system_message': 'You are an expert research planner and quality auditor with deep knowledge of research methodologies and resource optimization.',
    'template': '''Verify research results and assess quality:

Research Results: {research_results}
Verification Criteria: {verification_criteria}

Verification Tasks:
1. Confirm accuracy and reliability of results
2. Assess appropriateness of methodology
3. Identify potential biases or errors
4. Provide quality score and improvement recommendations

Return verification results.''',
    'variables': ['research_results', 'verification_criteria'],
    'description': 'Research results verification prompt'
}

# 평가
evaluation = {
    'system_message': 'You are an expert research evaluator with comprehensive quality assessment capabilities.',
    'template': '''Comprehensively evaluate the research process and results:

Evaluation Target: {evaluation_target}
Evaluation Criteria: {evaluation_criteria}

Evaluation Tasks:
1. Assess effectiveness of the process
2. Analyze quality of results
3. Identify strengths and weaknesses
4. Provide improvement recommendations

Return evaluation results.''',
    'variables': ['evaluation_target', 'evaluation_criteria'],
    'description': 'Process and results evaluation prompt'
}

# 종합
synthesis = {
    'system_message': 'You are an expert research synthesizer with adaptive context window capabilities.',
    'template': '''Synthesize data from various sources to generate integrated results:

Data Sources: {data_sources}
Synthesis Goal: {synthesis_goal}

Synthesis Tasks:
1. Integrate information from different sources
2. Analyze consistency and contradictions
3. Construct a synthesized narrative
4. Derive actionable conclusions

Return synthesis results.''',
    'variables': ['data_sources', 'synthesis_goal'],
    'description': 'Data synthesis prompt'
}

# 작업 분해
decomposition = {
    'system_message': 'You are an expert research project manager with deep knowledge of task decomposition and resource allocation.',
    'template': '''Decompose complex research tasks into independently executable subtasks:

Research Task: {research_task}
Decomposition Criteria: {decomposition_criteria}

Task Decomposition:
1. Identify main components of the task
2. Define scope of each subtask
3. Analyze dependency relationships
4. Set priorities and estimated time
5. Specify resource requirements

Return decomposed task list.''',
    'variables': ['research_task', 'decomposition_criteria'],
    'description': 'Task decomposition prompt'
}

# 프롬프트들을 딕셔너리로 묶어서 export
autonomous_orchestrator_prompts = {
    'analysis': analysis,
    'verification': verification,
    'evaluation': evaluation,
    'synthesis': synthesis,
    'decomposition': decomposition
}

