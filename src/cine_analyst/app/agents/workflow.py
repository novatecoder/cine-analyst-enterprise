import os
import requests
from langgraph.graph import StateGraph, END
from cine_analyst.app.agents.state import AgentState
from cine_analyst.rag.vector import VectorSearch
from cine_analyst.rag.graph import GraphSearch
from cine_analyst.common.config import settings
from loguru import logger

# 노드 1: 질문 의도 분석 (Planner)
def plan_node(state: AgentState):
    """사용자 질문을 분석하여 벡터 검색을 할지, 그래프 검색을 할지 결정합니다."""
    last_message = state["messages"][-1].content
    
    # 감독, 배우, 관계 등 구조적 정보가 필요한 키워드 탐지
    graph_keywords = ["감독", "배우", "관계", "출연", "나온", "작품"]
    if any(keyword in last_message for keyword in graph_keywords):
        logger.info("Decision: Graph Search (Neo4j)")
        return {"next_step": "graph"}
    
    logger.info("Decision: Vector Search (OpenSearch)")
    return {"next_step": "vector"}

# 노드 2-A: 벡터 검색 (OpenSearch)
def vector_retrieve_node(state: AgentState):
    """비정형 텍스트 기반 시놉시스 및 리뷰 검색"""
    query = state["messages"][-1].content
    store = VectorSearch()
    # 인터페이스 규격에 맞춰 search 호출
    results = store.search(query, k=3) 
    return {"retrieved_context": [str(r) for r in results]}

# 노드 2-B: 그래프 검색 (Neo4j)
def graph_retrieve_node(state: AgentState):
    """지식 그래프 기반 인물-영화 관계 검색"""
    query = state["messages"][-1].content
    store = GraphSearch()
    # 앞서 수정한 search(query, k=3) 인터페이스 호출
    results = store.search(query, k=3)
    return {"retrieved_context": [str(r) for r in results]}

# 노드 3: 결과 분석 및 답변 (Analyst - vLLM 연동)
def analyze_node(state: AgentState):
    """검색된 문맥을 바탕으로 vLLM(LoRA 적용 모델)을 호출하여 최종 답변 생성"""
    context = "\n".join(state["retrieved_context"])
    query = state["messages"][-1].content
    
    # vLLM API URL (설정 파일의 주소 사용: http://vllm:8000/v1)
    vllm_url = f"{settings.VLLM_URL}/chat/completions"
    
    payload = {
        "model": settings.MODEL_NAME, # .env 및 config에 설정된 'tuned-sql' 사용
        "messages": [
            {
                "role": "system", 
                "content": (
                    "당신은 CineAnalyst의 영화 전문가 에이전트입니다. "
                    f"제공된 문맥(Context)을 바탕으로 사용자의 질문에 친절하게 답변하세요.\n\n"
                    f"문맥: {context}"
                )
            },
            {"role": "user", "content": query}
        ],
        "temperature": 0.7,
        "max_tokens": 512
    }
    
    try:
        # 모델 추론 시간을 고려하여 타임아웃을 30초로 설정
        response = requests.post(vllm_url, json=payload, timeout=30)
        response.raise_for_status()
        
        result_json = response.json()
        answer = result_json["choices"][0]["message"]["content"]
        logger.info("Successfully generated answer from vLLM")
        
    except Exception as e:
        logger.error(f"❌ vLLM 호출 실패: {str(e)}")
        answer = (
            "죄송합니다. 모델 서버와 통신하는 중 오류가 발생했습니다. "
            f"검색된 정보는 다음과 같습니다: {context[:200]}..."
        )

    return {"messages": [("assistant", answer)]}

# --- 그래프 구성 및 엣지 정의 ---
workflow = StateGraph(AgentState)

# 각 단계(Node) 등록
workflow.add_node("planner", plan_node)
workflow.add_node("vector_retrieve", vector_retrieve_node)
workflow.add_node("graph_retrieve", graph_retrieve_node)
workflow.add_node("analyst", analyze_node)

# 시작점 설정
workflow.set_entry_point("planner")

# 조건부 라우팅 설정 (Planner 결과에 따라 분기)
workflow.add_conditional_edges(
    "planner",
    lambda x: x["next_step"],
    {
        "vector": "vector_retrieve",
        "graph": "graph_retrieve"
    }
)

# 검색 노드에서 분석 노드로 연결
workflow.add_edge("vector_retrieve", "analyst")
workflow.add_edge("graph_retrieve", "analyst")
# 분석 완료 후 종료
workflow.add_edge("analyst", END)

# 최종 워크플로우 컴파일
app = workflow.compile()