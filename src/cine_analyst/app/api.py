from fastapi import APIRouter, HTTPException
from cine_analyst.common.schemas import AnalysisRequest, AnalysisResponse
from cine_analyst.app.agents.workflow import app as agent_app
from langchain_core.messages import HumanMessage
from loguru import logger

router = APIRouter()

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_movie(request: AnalysisRequest):
    """LangGraph 에이전트를 호출하여 분석 결과를 반환하는 API"""
    try:
        # 에이전트 초기 입력 상태
        initial_state = {
            "messages": [HumanMessage(content=request.query)],
            "retrieved_context": [],
            "confidence_score": 0.0
        }

        # 워크플로우 실행
        result = await agent_app.ainvoke(initial_state)

        # 결과 데이터 추출
        final_answer = result["messages"][-1].content
        context_str = "\n".join(result.get("retrieved_context", []))

        return AnalysisResponse(
            answer=final_answer,
            context=context_str,
            recommendations=[],  # 필요 시 추가 로직 구현
            confidence_score=0.95 # QoS 예시 점수
        )

    except Exception as e:
        logger.error(f"❌ 에이전트 실행 실패: {str(e)}")
        raise HTTPException(status_code=500, detail="에이전트 처리 중 내부 오류 발생")