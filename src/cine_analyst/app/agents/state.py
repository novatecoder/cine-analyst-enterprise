from typing import Annotated, TypedDict, List, Union
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # 대화 기록 및 에이전트 간 메시지 (자동 합치기 기능 포함)
    messages: Annotated[List[BaseMessage], add_messages]
    
    # OpenSearch 및 Neo4j에서 검색된 지식 데이터
    retrieved_context: List[str]
    
    # 현재 분석의 신뢰도 점수 (QoS 평가용)
    confidence_score: float
    
    # 다음에 실행할 노드 이름
    next_step: str