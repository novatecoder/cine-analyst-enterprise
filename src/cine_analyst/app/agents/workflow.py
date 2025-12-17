import os
import requests
from langgraph.graph import StateGraph, END
from cine_analyst.app.agents.state import AgentState
from cine_analyst.rag.vector import VectorSearch
from cine_analyst.rag.graph import GraphSearch
from cine_analyst.common.config import settings

# 노드 1: 질문 의도 분석 (Planner)
def plan_node(state: AgentState):
    last_message = state["messages"][-1].content
    # 감독, 배우 등 관계 정보가 포함되면 그래프 검색(graph)으로 분기
    if any(keyword in last_message for keyword in ["감독", "배우", "관계", "출연"]):
        return {"next_step": "graph"}
    return {"next_step": "vector"}

# 노드 2-A: 벡터 검색 (OpenSearch)
def vector_retrieve_node(state: AgentState):
    query = state["messages"][-1].content
    store = VectorSearch()
    # base.py 인터페이스 규격인 search() 호출
    results = store.search(query, k=3) 
    return {"retrieved_context": [str(r) for r in results]}

# 노드 2-B: 그래프 검색 (Neo4j)
def graph_retrieve_node(state: AgentState):
    query = state["messages"][-1].content
    store = GraphSearch()
    # 지식 그래프 기반 관계 데이터 검색
    results = store.search(query, k=3)
    return {"retrieved_context": [str(r) for r in results]}

# 노드 3: 결과 분석 및 답변 (Analyst - vLLM 연동)
def analyze_node(state: AgentState):
    context = "\n".join(state["retrieved_context"])
    query = state["messages"][-1].content
    
    # vLLM API 호출 (OpenAI 호환 규격)
    vllm_url = f"{settings.VLLM_URL}/chat/completions"
    payload = {
        "model": "tuned_adapter",
        "messages": [
            {"role": "system", "content": f"당신은 영화 전문가입니다. 다음 문맥을 참고하세요: {context}"},
            {"role": "user", "content": query}
        ]
    }
    
    try:
        # 실제 vLLM 서버가 떠있지 않을 경우를 대비한 예외 처리
        response = requests.post(vllm_url, json=payload, timeout=5)
        answer = response.json()["choices"][0]["message"]["content"]
    except Exception:
        answer = "모델 서버(vLLM) 연결에 실패했습니다. 하지만 RAG 로직은 정상 작동 중입니다."

    return {"messages": [("assistant", answer)]}

# 그래프 구성 및 엣지 정의
workflow = StateGraph(AgentState)

workflow.add_node("planner", plan_node)
workflow.add_node("vector_retrieve", vector_retrieve_node)
workflow.add_node("graph_retrieve", graph_retrieve_node)
workflow.add_node("analyst", analyze_node)

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

workflow.add_edge("vector_retrieve", "analyst")
workflow.add_edge("graph_retrieve", "analyst")
workflow.add_edge("analyst", END)

app = workflow.compile()