from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "CineAnalyst Enterprise"
    ENV: str = "dev"
    DEBUG: bool = True

    # [Paths]
    DATA_DIR: str = "./data"
    RAW_DATA_PATH: str = "./data/raw/movies.csv"
    PROCESSED_DATA_PATH: str = "./data/processed/train.jsonl"
    MODEL_SAVE_DIR: str = "./models/tuned_adapter"

    # [vLLM]
    # docker-compose 호스트 이름과 포트를 기본값으로 설정
    VLLM_URL: str = "http://vllm:8000/v1"
    MODEL_NAME: str = "tuned-sql"

    # [OpenSearch]
    OPENSEARCH_URL: str = "http://opensearch:9200"
    OPENSEARCH_USER: str = "admin"
    OPENSEARCH_PASSWORD: str = "admin"
    OPENSEARCH_INDEX: str = "movies"

    # [Neo4j]
    # docker-compose 설정값과 일치시킴
    NEO4J_URI: str = "bolt://neo4j:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password123"

    # [Model Settings]
    BASE_MODEL_NAME: str = "unsloth/Qwen2.5-1.5B-Instruct"
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    MAX_SEQ_LENGTH: int = 2048
    LOAD_IN_4BIT: bool = True
    HF_TOKEN: Optional[str] = None
    
    # [LoRA]
    LORA_RANK: int = 16
    LORA_ALPHA: int = 32
    LORA_DROPOUT: float = 0.05

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()