import asyncio
import logging
import sys
import os
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger("HarnessTest")

async def test_harness_execution():
    logger.info("Starting Agent Harness Integration Test...")
    
    try:
        from src.core.researcher_config import load_config_from_env
        load_config_from_env()
        
        from src.core.agent_orchestrator import AgentOrchestrator
        
        orchestrator = AgentOrchestrator()
        
        # 테스트 요청 (금융 분석 시나리오 - TaskRouter가 금융 파이프라인으로 인식해야 함)
        test_request = "엔비디아(NVDA) 주가 전망과 AI 반도체 시장 점유율 분석해줘."
        session_id = "test_session_2026"
        
        logger.info(f"Target Request: {test_request}")
        
        # 실행
        result = await orchestrator.execute(request=test_request, session_id=session_id)
        
        logger.info("--- Execution Result ---")
        logger.info(f"Success: {result.get('success')}")
        logger.info(f"Plan summary: {result.get('plan')[:200] if result.get('plan') else 'No plan'}")
        logger.info(f"Tasks generated: {len(result.get('tasks', []))}")
        
        for i, task in enumerate(result.get('tasks', [])):
            logger.info(f"  Task {i+1}: {task.get('description')[:100]}")
            
        logger.info(f"Final output hint: {result.get('results')}")
        
        if result.get('success'):
            logger.info("✅ Harness Integration Test PASSED")
        else:
            logger.error("❌ Harness Integration Test FAILED")
            logger.error(f"Error: {result.get('error')}")

    except Exception as e:
        logger.exception(f"❌ Critical error during test: {e}")

if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        logger.error("GOOGLE_API_KEY environment variable is missing. LLM calls will fail.")
    else:
        asyncio.run(test_harness_execution())
