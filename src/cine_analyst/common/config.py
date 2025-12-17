from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "CineAnalyst Enterprise"
    ENV: str = "local"
    DEBUG: bool = True

    # [Paths]
    DATA_DIR: str = "./data"
    RAW_DATA_PATH: str = "./data/raw/movies.csv"            # 파일명까지 포함하는 것이 안전함
    PROCESSED_DATA_PATH: str = "./data/processed/train.jsonl" # 파일명까지 포함하는 것이 안전함
    MODEL_SAVE_DIR: str = "./models/tuned_adapter"

    # [vLLM]
    # docker-compose에서 8081:8000으로 포트 매핑을 했으므로 외부 호출 시 8081이 맞습니다.
    VLLM_URL: str = "http://localhost:8081/v1"
    MODEL_NAME: str = "tuned_adapter"

    # [OpenSearch]
    OPENSEARCH_URL: str = "http://localhost:9200"
    OPENSEARCH_USER: str = "admin"
    OPENSEARCH_PASSWORD: str = "admin"

    # [Neo4j]
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"

    # [Model Settings] - trainer.py 실행에 필요한 누락된 필드들 추가
    BASE_MODEL_NAME: str = "unsloth/Qwen2.5-1.5B-Instruct"
    MAX_SEQ_LENGTH: int = 2048  # <-- 이 부분이 없어서 에러가 발생했습니다.
    LOAD_IN_4BIT: bool = True   # <-- trainer.py에서 참조하는 필드입니다.
    HF_TOKEN: Optional[str] = None
    
    # LoRA 관련
    LORA_RANK: int = 16
    LORA_ALPHA: int = 32
    LORA_DROPOUT: float = 0.05  # <-- 추가

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()