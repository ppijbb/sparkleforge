"""
Research Agent 프롬프트 모듈

Research Agent에서 사용되는 모든 프롬프트들을 포함합니다.
"""

# 연구 계획 수립
planning = {
    'system_message': 'You are an expert research analyst with pattern recognition capabilities.',
    'template': '''다음 연구 작업을 위한 구체적인 연구 계획을 수립하세요.

작업: {task_description}
작업 유형: {task_type}

다음을 포함한 연구 계획을 수립하세요:
1. 필요한 정보 유형 식별
2. 적절한 정보원 선택
3. 검색 전략 수립
4. 데이터 수집 방법 결정
5. 분석 방법 선택

JSON 형태로 응답하세요:
{{
    "research_strategy": "전체 연구 전략",
    "steps": [
        {{
            "step_id": "step_1",
            "description": "단계 설명",
            "method": "web_search|academic_search|data_analysis|content_analysis",
            "query": "검색 쿼리",
            "sources": ["웹", "학술", "데이터베이스"],
            "expected_output": "예상 결과"
        }}
    ],
    "quality_criteria": ["품질 기준1", "품질 기준2"],
    "success_metrics": ["성공 지표1", "성공 지표2"]
}}''',
    'variables': ['task_description', 'task_type'],
    'description': '연구 계획 수립 프롬프트'
}

# Google 검색
google_search = {
    'system_message': 'You are an expert researcher using Google Custom Search API.',
    'template': '''Google Custom Search를 사용하여 다음 쿼리에 대한 검색을 수행하세요.

쿼리: {query}
최대 결과 수: {max_results}

검색 전략:
1. 쿼리를 효과적으로 구성하여 정확한 결과를 얻으세요
2. 다양한 관점에서 검색어를 조합하세요
3. 신뢰할 수 있는 출처를 우선시하세요

검색 결과를 구조화하여 반환하세요.''',
    'variables': ['query', 'max_results'],
    'description': 'Google 검색 실행 프롬프트'
}

# Bing 검색
bing_search = {
    'system_message': 'You are an expert researcher using Bing Search API.',
    'template': '''Bing Search를 사용하여 다음 쿼리에 대한 검색을 수행하세요.

쿼리: {query}
최대 결과 수: {max_results}

검색 전략:
1. Bing의 강점을 활용하여 다양한 결과를 얻으세요
2. 뉴스, 학술 자료, 일반 웹을 균형 있게 검색하세요
3. 최신 정보를 우선시하세요

검색 결과를 구조화하여 반환하세요.''',
    'variables': ['query', 'max_results'],
    'description': 'Bing 검색 실행 프롬프트'
}

# 웹 스크래핑
web_scraping = {
    'system_message': 'You are an expert web scraper with content extraction capabilities. You ensure URL normalization and avoid duplication.',
    'template': '''다음 URL에서 콘텐츠를 추출하고 분석하세요.

URL: {url}
추출 목표: {extraction_goal}

**변수 검증 (Variable Validation):**
- `url`이 존재하고 None이 아닌지 확인하세요. 없거나 None인 경우 명시적으로 보고하세요.
- `extraction_goal`이 존재하고 유효한지 확인하세요.

**URL 관리 규칙 (URL Management):**
- URL은 전체 표기하세요 (축약 금지).
- URL 정규화: trailing slash 제거, 소문자 변환.
- 소스 수집 시 중복 체크를 수행하세요.
- 이미 수집된 URL인지 확인하세요.

추출 작업:
1. 웹페이지 콘텐츠를 체계적으로 추출하세요
2. 텍스트, 이미지, 메타데이터를 포함하세요
3. 구조화된 형식으로 정리하세요
4. URL을 전체 표기하여 기록하세요

추출 결과를 분석하여 핵심 정보를 요약하세요.''',
    'variables': ['url', 'extraction_goal'],
    'description': '웹 스크래핑 프롬프트 (URL 정규화 및 중복 체크 포함)'
}

# 브라우저 연구
browser_research = {
    'system_message': 'You are an expert researcher with browser automation capabilities. You ensure URL normalization and avoid duplication.',
    'template': '''브라우저 자동화를 사용하여 다음 연구 작업을 수행하세요.

연구 계획: {research_plan}
최대 결과 수: {max_results}

**변수 검증 (Variable Validation):**
- `research_plan`이 존재하고 None이 아닌지 확인하세요. 없거나 None인 경우 명시적으로 보고하세요.
- `max_results`가 유효한 숫자인지 확인하세요.

**URL 관리 규칙 (URL Management):**
- 모든 URL은 전체 표기하세요 (축약 금지).
- URL 정규화: trailing slash 제거, 소문자 변환.
- 소스 수집 시 중복 체크를 수행하세요.
- 이미 수집된 URL인지 확인하세요.

연구 전략:
1. 브라우저를 사용하여 동적 웹사이트에 접근하세요
2. JavaScript 기반 콘텐츠를 추출하세요
3. 상호작용이 필요한 페이지를 처리하세요
4. 모든 URL을 전체 표기하여 기록하세요

연구 결과를 종합하여 반환하세요.''',
    'variables': ['research_plan', 'max_results'],
    'description': '브라우저 자동화 연구 프롬프트 (URL 정규화 및 중복 체크 포함)'
}

# 상호작용 연구
interactive_research = {
    'system_message': 'You are an expert researcher performing interactive web research.',
    'template': '''브라우저 자동화를 사용하여 다음 연구 작업을 수행하세요.

연구 계획: {research_plan}
최대 결과 수: {max_results}

연구 전략:
1. 여러 웹사이트를 체계적으로 탐색하세요
2. 관련 정보를 종합하세요
3. 상호작용을 통해 추가 정보를 수집하세요

연구 결과를 구조화하여 반환하세요.''',
    'variables': ['research_plan', 'max_results'],
    'description': '상호작용 웹 연구 프롬프트'
}

# 분석
analysis = {
    'system_message': 'You are an expert research analyst.',
    'template': '''수집된 연구 데이터를 분석하세요.

데이터: {data}
분석 유형: {analysis_type}

분석 작업:
1. 데이터를 체계적으로 분석하세요
2. 패턴과 트렌드를 식별하세요
3. 통찰력을 도출하세요

분석 결과를 구조화하여 반환하세요.''',
    'variables': ['data', 'analysis_type'],
    'description': '데이터 분석 프롬프트'
}

# 통찰력 도출
insights = {
    'system_message': 'You are an expert research analyst generating insights.',
    'template': '''연구 데이터에서 통찰력을 도출하세요.

데이터: {data}
분석 유형: {analysis_type}

통찰력 도출:
1. 데이터의 의미를 해석하세요
2. 숨겨진 패턴을 발견하세요
3. 실질적인 통찰력을 제시하세요

통찰력을 구조화하여 반환하세요.''',
    'variables': ['data', 'analysis_type'],
    'description': '통찰력 도출 프롬프트'
}

# 보고서 작성
report_writing = {
    'system_message': 'You are an expert research report writer.',
    'template': '''연구 결과를 바탕으로 보고서를 작성하세요.

연구 데이터: {data}
보고서 유형: {report_type}

보고서 작성:
1. 구조화된 보고서를 작성하세요
2. 실행 가능한 권장사항을 포함하세요
3. 명확하고 전문적인 언어를 사용하세요

완성된 보고서를 반환하세요.''',
    'variables': ['data', 'report_type'],
    'description': '보고서 작성 프롬프트'
}

# 종합
synthesis = {
    'system_message': 'You are an expert research synthesizer. You collaborate with other agents (Validation Agent, Synthesis Agent) to ensure quality. IMPORTANT: All internal communication between agents must be in English. Only the final report to the user should be in the user\'s requested language.',
    'template': '''여러 출처의 연구 데이터를 종합하세요.

데이터: {data}
종합 유형: {synthesis_type}

**변수 검증 (Variable Validation):**
- `data`가 존재하고 None이 아닌지 확인하세요. 없거나 None인 경우 명시적으로 보고하세요.
- `synthesis_type`이 존재하고 유효한지 확인하세요.

**URL 관리 규칙 (URL Management):**
- 모든 URL은 전체 표기하세요 (축약 금지).
- URL 정규화: trailing slash 제거, 소문자 변환.
- 소스 수집 시 중복 체크를 수행하세요.

데이터 종합:
1. 다양한 출처의 정보를 통합하세요
2. 일관성과 모순을 분석하세요
3. 종합된 결과를 도출하세요
4. 모든 URL을 전체 표기하여 기록하세요

**Agent 간 협동 (Agent Collaboration):**
- Validation Agent에게 결과 검증을 요청하세요.
- Synthesis Agent와 협동하여 품질을 보장하세요.
- **중요: 모든 내부 agent 간 소통은 영어로 해야 합니다.**

종합 결과를 반환하세요.''',
    'variables': ['data', 'synthesis_type'],
    'description': '데이터 종합 프롬프트 (URL 관리 및 협동 기능 포함)'
}

# 검증
validation = {
    'system_message': 'You are an expert research validator. You ensure comprehensive validation including URL duplication checks and source verification. IMPORTANT: All internal communication between agents must be in English. Only the final report to the user should be in the user\'s requested language.',
    'template': '''연구 결과를 검증하세요.

결과: {results}
검증 기준: {criteria}

**변수 검증 (Variable Validation):**
- `results`가 존재하고 None이 아닌지 확인하세요. 없거나 None인 경우 명시적으로 보고하세요.
- `criteria`가 존재하고 유효한지 확인하세요.

검증 작업:
1. 결과의 정확성을 검증하세요
2. 출처의 신뢰성을 확인하세요
3. **URL 중복 검사**: 모든 URL이 한 번만 나타나는지 확인하세요
4. **소스 검증**: 각 URL이 실제로 사용되었는지 확인하세요
5. 검증 결과를 정리하세요

검증 결과를 반환하세요.''',
    'variables': ['results', 'criteria'],
    'description': '연구 결과 검증 프롬프트 (URL 중복 검사 및 소스 검증 강화)'
}

# 권장사항
recommendations = {
    'system_message': 'You are an expert consultant providing recommendations.',
    'template': '''연구 결과를 바탕으로 권장사항을 제시하세요.

데이터: {data}
컨텍스트: {context}

권장사항 제시:
1. 연구 결과를 분석하세요
2. 실질적인 권장사항을 도출하세요
3. 실행 계획을 포함하세요

권장사항을 구조화하여 반환하세요.''',
    'variables': ['data', 'context'],
    'description': '권장사항 제시 프롬프트'
}

# 품질 평가
quality_assessment = {
    'system_message': 'You are an expert quality assessor.',
    'template': '''연구 결과의 품질을 평가하세요.

결과: {results}
품질 기준: {criteria}

품질 평가:
1. 다양한 측면에서 품질을 평가하세요
2. 강점과 약점을 식별하세요
3. 종합 평가를 제시하세요

품질 평가 결과를 반환하세요.''',
    'variables': ['results', 'criteria'],
    'description': '품질 평가 프롬프트'
}

# 완전성 체크
completeness_check = {
    'system_message': 'You are an expert completeness evaluator.',
    'template': '''연구 결과의 완전성을 평가하세요.

결과: {results}
완전성 기준: {criteria}

완전성 평가:
1. 커버된 범위를 분석하세요
2. 누락된 부분을 식별하세요
3. 추가 조사가 필요한 영역을 제안하세요

완전성 평가 결과를 반환하세요.''',
    'variables': ['results', 'criteria'],
    'description': '완전성 평가 프롬프트'
}

# 정확성 체크
accuracy_check = {
    'system_message': 'You are an expert accuracy assessor.',
    'template': '''연구 결과의 정확성을 평가하세요.

결과: {results}
정확성 기준: {criteria}

정확성 평가:
1. 사실 관계를 검증하세요
2. 오류를 식별하세요
3. 정확성 수준을 평가하세요

정확성 평가 결과를 반환하세요.''',
    'variables': ['results', 'criteria'],
    'description': '정확성 평가 프롬프트'
}

# 관련성 체크
relevance_check = {
    'system_message': 'You are an expert relevance evaluator.',
    'template': '''연구 결과의 관련성을 평가하세요.

결과: {results}
쿼리: {query}

관련성 평가:
1. 쿼리와의 관련성을 분석하세요
2. 관련성 수준을 평가하세요
3. 개선 방안을 제시하세요

관련성 평가 결과를 반환하세요.''',
    'variables': ['results', 'query'],
    'description': '관련성 평가 프롬프트'
}

# 프롬프트들을 딕셔너리로 묶어서 export
research_agent_prompts = {
    'planning': planning,
    'google_search': google_search,
    'bing_search': bing_search,
    'web_scraping': web_scraping,
    'browser_research': browser_research,
    'interactive_research': interactive_research,
    'analysis': analysis,
    'insights': insights,
    'report_writing': report_writing,
    'synthesis': synthesis,
    'validation': validation,
    'recommendations': recommendations,
    'quality_assessment': quality_assessment,
    'completeness_check': completeness_check,
    'accuracy_check': accuracy_check,
    'relevance_check': relevance_check
}

