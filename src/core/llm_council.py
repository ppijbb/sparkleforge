"""
LLM Council - 3-stage multi-model consensus system

3단계 협의 시스템을 통한 다중 모델 합의 도출.
바이브 코딩 선서 준수: fallback 제거, 하드 코딩 제거, gemini-2.5-flash 계열 우선.
"""

import asyncio
import logging
import re
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import httpx

from src.core.researcher_config import get_council_config

logger = logging.getLogger(__name__)


class CouncilError(Exception):
    """Council 실행 중 발생한 에러."""
    pass


async def query_model_via_openrouter(
    model: str,
    messages: List[Dict[str, str]],
    api_key: str,
    api_url: str,
    timeout: float = 120.0
) -> Dict[str, Any]:
    """
    OpenRouter API를 통해 단일 모델 쿼리.
    
    Args:
        model: OpenRouter 모델 식별자 (예: "google/gemini-2.5-flash-lite")
        messages: 메시지 딕셔너리 리스트 ('role'과 'content' 키 포함)
        api_key: OpenRouter API 키
        api_url: OpenRouter API URL
        timeout: 요청 타임아웃 (초)
    
    Returns:
        'content'와 선택적 'reasoning_details'를 포함한 응답 딕셔너리
    
    Raises:
        CouncilError: 쿼리 실패 시 명확한 에러 반환 (fallback 제거)
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": model,
        "messages": messages,
    }
    
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                api_url,
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            
            data = response.json()
            message = data['choices'][0]['message']
            
            return {
                'content': message.get('content', ''),
                'reasoning_details': message.get('reasoning_details')
            }
    
    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP error querying model {model}: {e.response.status_code} - {e.response.text}"
        logger.error(error_msg)
        raise CouncilError(error_msg) from e
    except httpx.TimeoutException as e:
        error_msg = f"Timeout querying model {model}: {timeout}s exceeded"
        logger.error(error_msg)
        raise CouncilError(error_msg) from e
    except Exception as e:
        error_msg = f"Error querying model {model}: {str(e)}"
        logger.error(error_msg)
        raise CouncilError(error_msg) from e


async def query_models_parallel(
    models: List[str],
    messages: List[Dict[str, str]],
    api_key: str,
    api_url: str,
    timeout: float = 120.0
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    여러 모델을 병렬로 쿼리.
    
    Args:
        models: OpenRouter 모델 식별자 리스트
        messages: 각 모델에 보낼 메시지 딕셔너리 리스트
        api_key: OpenRouter API 키
        api_url: OpenRouter API URL
        timeout: 요청 타임아웃 (초)
    
    Returns:
        모델 식별자를 키로 하고 응답 딕셔너리(또는 None)를 값으로 하는 딕셔너리
    """
    # 모든 모델에 대한 태스크 생성
    tasks = [
        query_model_via_openrouter(model, messages, api_key, api_url, timeout)
        for model in models
    ]
    
    # 모든 태스크 완료 대기 (일부 실패해도 계속 진행)
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 결과 매핑 (에러는 None으로 처리)
    response_dict = {}
    for model, result in zip(models, results):
        if isinstance(result, Exception):
            logger.warning(f"Model {model} failed: {result}")
            response_dict[model] = None
        else:
            response_dict[model] = result
    
    return response_dict


async def stage1_collect_responses(
    user_query: str,
    council_models: List[str],
    api_key: str,
    api_url: str,
    timeout: float = 120.0
) -> List[Dict[str, Any]]:
    """
    Stage 1: 모든 council 모델에서 개별 응답 수집.
    
    Args:
        user_query: 사용자 질문
        council_models: Council 모델 목록
        api_key: OpenRouter API 키
        api_url: OpenRouter API URL
        timeout: 요청 타임아웃 (초)
    
    Returns:
        'model'과 'response' 키를 가진 딕셔너리 리스트
    
    Raises:
        CouncilError: 모든 모델이 실패한 경우
    """
    messages = [{"role": "user", "content": user_query}]
    
    # 모든 모델을 병렬로 쿼리
    responses = await query_models_parallel(council_models, messages, api_key, api_url, timeout)
    
    # 성공한 응답만 포맷팅
    stage1_results = []
    for model, response in responses.items():
        if response is not None:
            stage1_results.append({
                "model": model,
                "response": response.get('content', '')
            })
    
    # 모든 모델이 실패한 경우 명확한 에러 반환 (fallback 제거)
    if not stage1_results:
        error_msg = f"All {len(council_models)} council models failed to respond"
        logger.error(error_msg)
        raise CouncilError(error_msg)
    
    logger.info(f"Stage 1: Collected {len(stage1_results)}/{len(council_models)} responses")
    return stage1_results


async def stage2_collect_rankings(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    council_models: List[str],
    api_key: str,
    api_url: str,
    timeout: float = 120.0
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Stage 2: 각 모델이 익명화된 응답을 검토하고 순위 매기기.
    
    Args:
        user_query: 원본 사용자 쿼리
        stage1_results: Stage 1 결과
        council_models: Council 모델 목록
        api_key: OpenRouter API 키
        api_url: OpenRouter API URL
        timeout: 요청 타임아웃 (초)
    
    Returns:
        (순위 리스트, label_to_model 매핑) 튜플
    """
    # 익명화된 레이블 생성 (Response A, Response B, etc.)
    labels = [chr(65 + i) for i in range(len(stage1_results))]  # A, B, C, ...
    
    # 레이블에서 모델 이름으로의 매핑 생성
    label_to_model = {
        f"Response {label}": result['model']
        for label, result in zip(labels, stage1_results)
    }
    
    # 순위 매기기 프롬프트 구성
    responses_text = "\n\n".join([
        f"Response {label}:\n{result['response']}"
        for label, result in zip(labels, stage1_results)
    ])
    
    ranking_prompt = f"""You are evaluating different responses to the following question:

Question: {user_query}

Here are the responses from different models (anonymized):

{responses_text}

Your task:
1. First, evaluate each response individually. For each response, explain what it does well and what it does poorly.
2. Then, at the very end of your response, provide a final ranking.

IMPORTANT: Your final ranking MUST be formatted EXACTLY as follows:
- Start with the line "FINAL RANKING:" (all caps, with colon)
- Then list the responses from best to worst as a numbered list
- Each line should be: number, period, space, then ONLY the response label (e.g., "1. Response A")
- Do not add any other text or explanations in the ranking section

Example of the correct format for your ENTIRE response:

Response A provides good detail on X but misses Y...
Response B is accurate but lacks depth on Z...
Response C offers the most comprehensive answer...

FINAL RANKING:
1. Response C
2. Response A
3. Response B

Now provide your evaluation and ranking:"""
    
    messages = [{"role": "user", "content": ranking_prompt}]
    
    # 모든 council 모델에서 순위 수집 (병렬)
    responses = await query_models_parallel(council_models, messages, api_key, api_url, timeout)
    
    # 결과 포맷팅
    stage2_results = []
    for model, response in responses.items():
        if response is not None:
            full_text = response.get('content', '')
            parsed = parse_ranking_from_text(full_text)
            stage2_results.append({
                "model": model,
                "ranking": full_text,
                "parsed_ranking": parsed
            })
    
    logger.info(f"Stage 2: Collected {len(stage2_results)}/{len(council_models)} rankings")
    return stage2_results, label_to_model


def parse_ranking_from_text(ranking_text: str) -> List[str]:
    """
    모델 응답에서 FINAL RANKING 섹션 파싱.
    
    Args:
        ranking_text: 모델의 전체 텍스트 응답
    
    Returns:
        순위가 매겨진 순서대로 응답 레이블 리스트
    """
    # "FINAL RANKING:" 섹션 찾기
    if "FINAL RANKING:" in ranking_text:
        # "FINAL RANKING:" 이후 추출
        parts = ranking_text.split("FINAL RANKING:")
        if len(parts) >= 2:
            ranking_section = parts[1]
            # 번호가 매겨진 리스트 형식 추출 시도 (예: "1. Response A")
            numbered_matches = re.findall(r'\d+\.\s*Response [A-Z]', ranking_section)
            if numbered_matches:
                # "Response X" 부분만 추출
                return [re.search(r'Response [A-Z]', m).group() for m in numbered_matches]
            
            # Fallback: 모든 "Response X" 패턴을 순서대로 추출
            matches = re.findall(r'Response [A-Z]', ranking_section)
            return matches
    
    # Fallback: 텍스트 전체에서 "Response X" 패턴 찾기
    matches = re.findall(r'Response [A-Z]', ranking_text)
    return matches


def calculate_aggregate_rankings(
    stage2_results: List[Dict[str, Any]],
    label_to_model: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    모든 모델의 집계 순위 계산.
    
    Args:
        stage2_results: 각 모델의 순위 결과
        label_to_model: 익명 레이블에서 모델 이름으로의 매핑
    
    Returns:
        모델 이름과 평균 순위를 포함한 딕셔너리 리스트 (최고에서 최악 순으로 정렬)
    """
    # 각 모델의 위치 추적
    model_positions = defaultdict(list)
    
    for ranking in stage2_results:
        ranking_text = ranking['ranking']
        
        # 구조화된 형식에서 순위 파싱
        parsed_ranking = parse_ranking_from_text(ranking_text)
        
        for position, label in enumerate(parsed_ranking, start=1):
            if label in label_to_model:
                model_name = label_to_model[label]
                model_positions[model_name].append(position)
    
    # 각 모델의 평균 위치 계산
    aggregate = []
    for model, positions in model_positions.items():
        if positions:
            avg_rank = sum(positions) / len(positions)
            aggregate.append({
                "model": model,
                "average_rank": round(avg_rank, 2),
                "rankings_count": len(positions)
            })
    
    # 평균 순위로 정렬 (낮을수록 좋음)
    aggregate.sort(key=lambda x: x['average_rank'])
    
    return aggregate


async def stage3_synthesize_final(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    stage2_results: List[Dict[str, Any]],
    chairman_model: str,
    api_key: str,
    api_url: str,
    timeout: float = 120.0
) -> Dict[str, Any]:
    """
    Stage 3: Chairman가 최종 응답 종합.
    
    Args:
        user_query: 원본 사용자 쿼리
        stage1_results: Stage 1의 개별 모델 응답
        stage2_results: Stage 2의 순위 결과
        chairman_model: Chairman 모델 식별자
        api_key: OpenRouter API 키
        api_url: OpenRouter API URL
        timeout: 요청 타임아웃 (초)
    
    Returns:
        'model'과 'response' 키를 가진 딕셔너리
    
    Raises:
        CouncilError: Chairman 모델 쿼리 실패 시
    """
    # Chairman를 위한 포괄적인 컨텍스트 구성
    stage1_text = "\n\n".join([
        f"Model: {result['model']}\nResponse: {result['response']}"
        for result in stage1_results
    ])
    
    stage2_text = "\n\n".join([
        f"Model: {result['model']}\nRanking: {result['ranking']}"
        for result in stage2_results
    ])
    
    chairman_prompt = f"""You are the Chairman of an LLM Council. Multiple AI models have provided responses to a user's question, and then ranked each other's responses.

Original Question: {user_query}

STAGE 1 - Individual Responses:
{stage1_text}

STAGE 2 - Peer Rankings:
{stage2_text}

Your task as Chairman is to synthesize all of this information into a single, comprehensive, accurate answer to the user's original question. Consider:
- The individual responses and their insights
- The peer rankings and what they reveal about response quality
- Any patterns of agreement or disagreement

Provide a clear, well-reasoned final answer that represents the council's collective wisdom:"""
    
    messages = [{"role": "user", "content": chairman_prompt}]
    
    # Chairman 모델 쿼리
    try:
        response = await query_model_via_openrouter(
            chairman_model, messages, api_key, api_url, timeout
        )
    except CouncilError as e:
        error_msg = f"Chairman model {chairman_model} failed: {str(e)}"
        logger.error(error_msg)
        raise CouncilError(error_msg) from e
    
    return {
        "model": chairman_model,
        "response": response.get('content', '')
    }


async def run_full_council(
    user_query: str,
    council_models: Optional[List[str]] = None,
    chairman_model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    timeout: Optional[float] = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """
    전체 3단계 council 프로세스 실행.
    
    Args:
        user_query: 사용자 질문
        council_models: Council 모델 목록 (None이면 설정에서 로드)
        chairman_model: Chairman 모델 (None이면 설정에서 로드)
        api_key: OpenRouter API 키 (None이면 설정에서 로드)
        api_url: OpenRouter API URL (None이면 설정에서 로드)
        timeout: 요청 타임아웃 (None이면 설정에서 로드)
    
    Returns:
        (stage1_results, stage2_results, stage3_result, metadata) 튜플
    
    Raises:
        CouncilError: Council 실행 실패 시
    """
    # 설정에서 로드 (하드 코딩 제거)
    if council_models is None or chairman_model is None or api_key is None or api_url is None or timeout is None:
        council_config = get_council_config()
        council_models = council_models or council_config.council_models
        chairman_model = chairman_model or council_config.chairman_model
        api_key = api_key or council_config.openrouter_api_key
        api_url = api_url or council_config.openrouter_api_url
        timeout = timeout or council_config.request_timeout
    
    logger.info(f"Running full council with {len(council_models)} models")
    
    # Stage 1: 개별 응답 수집
    stage1_results = await stage1_collect_responses(
        user_query, council_models, api_key, api_url, timeout
    )
    
    # Stage 2: 순위 수집
    stage2_results, label_to_model = await stage2_collect_rankings(
        user_query, stage1_results, council_models, api_key, api_url, timeout
    )
    
    # 집계 순위 계산
    aggregate_rankings = calculate_aggregate_rankings(stage2_results, label_to_model)
    
    # Stage 3: 최종 답변 종합
    stage3_result = await stage3_synthesize_final(
        user_query, stage1_results, stage2_results,
        chairman_model, api_key, api_url, timeout
    )
    
    # 메타데이터 준비
    metadata = {
        "label_to_model": label_to_model,
        "aggregate_rankings": aggregate_rankings
    }
    
    logger.info("Full council process completed successfully")
    return stage1_results, stage2_results, stage3_result, metadata

