from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# --- [기존 스키마] ETL 및 학습용 ---

class MovieData(BaseModel):
    """ETL 파이프라인에서 사용되는 내부 데이터 스키마"""
    id: Optional[str] = None
    title: str
    overview: str
    genres: List[str]
    release_date: Optional[str] = None
    keywords: List[str] = []
    
class TrainingExample(BaseModel):
    """학습 데이터셋(JSONL) 포맷"""
    messages: List[Dict[str, str]]


# --- [새로 추가] 서비스 및 API 통신용 ---
# 공고 요건: AI 서비스 품질(QoS) 및 비즈니스 영향도 측정을 위한 필드 고려

class AnalysisRequest(BaseModel):
    """사용자로부터 받는 분석 요청 규격"""
    query: str = Field(..., example="봉준호 감독의 기생충과 비슷한 사회 비판적인 영화 추천해줘")
    user_id: Optional[str] = Field("guest", description="사용자 식별자")

class AnalysisResponse(BaseModel):
    """에이전트 분석 결과 응답 규격"""
    answer: str = Field(..., description="LLM 에이전트가 생성한 최종 답변")
    context: Optional[str] = Field(None, description="RAG 과정에서 참조된 데이터베이스 컨텍스트")
    recommendations: List[str] = Field(default_factory=list, description="그래프 DB 기반 추천 영화 목록")
    confidence_score: float = Field(0.0, description="답변의 신뢰도 점수 (QoS 평가용)")