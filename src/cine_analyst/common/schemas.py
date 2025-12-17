from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

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