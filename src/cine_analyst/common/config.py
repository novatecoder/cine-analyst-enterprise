import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "CineAnalyst Enterprise"
    ENV: str = "local"
    
    # [Paths]
    DATA_DIR: str = "./data"
    RAW_DATA_PATH: str = "./data/raw/movies.csv"
    PROCESSED_DATA_PATH: str = "./data/processed/train.jsonl"
    MODEL_BASE_DIR: str = "./models"
    MODEL_SAVE_DIR: str = "./models/tuned_adapter"
    
    # [Model]
    BASE_MODEL_NAME: str = "unsloth/Qwen2.5-1.5B-Instruct"
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    MAX_SEQ_LENGTH: int = 2048
    LOAD_IN_4BIT: bool = True
    
    # [Infra]
    OPENSEARCH_URL: str = "http://localhost:9200"
    OPENSEARCH_INDEX: str = "movies-v1"
    
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_AUTH_USER: str = "neo4j"
    NEO4J_AUTH_PASS: str = "password"
    
    VLLM_API_URL: str = "http://localhost:8000/v1"

    # LoRA 관련 설정 추가
    LORA_RANK: int = 16  # 기본값으로 16 또는 32 권장
    LORA_ALPHA: int = 32
    LORA_DROPOUT: float = 0.05

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        extra="ignore"
    )

settings = Settings()