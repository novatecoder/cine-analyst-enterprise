import pytest
from unittest.mock import patch, MagicMock # 1. MagicMock 임포트 확인
from langchain_core.messages import HumanMessage
# workflow.py에서 정의한 정확한 노드 함수명을 가져옵니다.
from cine_analyst.app.agents.workflow import plan_node, vector_retrieve_node, analyze_node

def test_planner_routing_logic():
    """질문에 따른 분기 로직 검증 (Vector vs Graph)"""
    # 벡터 검색 경로 (workflow.py의 return값과 일치해야 함)
    state_v = {"messages": [HumanMessage(content="액션 영화 추천해줘")]}
    assert plan_node(state_v)["next_step"] == "vector"

    # 그래프 검색 경로
    state_g = {"messages": [HumanMessage(content="봉준호 감독 작품 알려줘")]}
    assert plan_node(state_g)["next_step"] == "graph"

@patch('cine_analyst.app.agents.workflow.VectorSearch')
def test_vector_retrieve_node_mock(mock_vector):
    """DB 없이 Mock으로 검색 노드 로직 검증"""
    mock_inst = mock_vector.return_value
    # base.py 인터페이스인 search 메서드를 mock 처리
    mock_inst.search.return_value = [{"title": "기생충", "overview": "테스트 데이터"}]

    state = {"messages": [HumanMessage(content="쿼리")]}
    result = vector_retrieve_node(state)

    assert "기생충" in str(result["retrieved_context"])
    mock_inst.search.assert_called_once()

@pytest.mark.asyncio
async def test_full_workflow_mock():
    """에이전트 전체 실행 흐름 통합 테스트 (vLLM 제외)"""
    from cine_analyst.app.agents.workflow import app as agent_app
    
    # VectorSearch와 requests.post(vLLM 호출)를 모두 Mock 처리하여 환경변수/네트워크 에러 방지
    with patch("cine_analyst.rag.vector.VectorSearch.search") as mock_s, \
         patch("cine_analyst.app.agents.workflow.requests.post") as mock_post:
        
        # 검색 결과 Mock
        mock_s.return_value = [{"title": "기생충"}]
        
        # vLLM 응답 Mock
        mock_res = MagicMock()
        mock_res.status_code = 200
        mock_res.json.return_value = {
            "choices": [{"message": {"content": "분석 완료되었습니다."}}]
        }
        mock_post.return_value = mock_res
        
        input_data = {
            "messages": [HumanMessage(content="영화 분석해줘")],
            "retrieved_context": [],
            "confidence_score": 0.0
        }
        
        try:
            # 워크플로우 실행
            final_output = await agent_app.ainvoke(input_data)
            assert len(final_output["messages"]) > 1
            assert final_output["messages"][-1].content == "분석 완료되었습니다."
        except Exception as e:
            pytest.fail(f"워크플로우 실행 실패: {e}")