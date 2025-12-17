import os
import json
import pandas as pd
from typing import List
from loguru import logger
from cine_analyst.common.config import settings
from cine_analyst.common.schemas import TrainingExample

SYSTEM_PROMPT = """You are an expert movie analyst. Analyze the movie details and output a JSON response."""

def preprocess_for_training(
    input_path: str = settings.RAW_DATA_PATH,
    output_path: str = settings.PROCESSED_DATA_PATH,
    sample_size: int = None
):
    """
    CSV 데이터를 읽어 LLM 학습용 JSONL 포맷(Chat Template)으로 변환합니다.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    logger.info(f"⚙️ Preprocessing data: {input_path}")
    
    df = pd.read_csv(input_path)
    if sample_size:
        df = df.sample(n=min(len(df), sample_size), random_state=42)
    
    formatted_data = []
    
    for _, row in df.iterrows():
        title = row.get('title', '')
        overview = row.get('overview', '')
        genres = row.get('genres', '[]')
        
        if pd.isna(overview) or len(str(overview)) < 10:
            continue

        # User Input Construction
        user_content = f"Title: {title}\nGenres: {genres}\nOverview: {overview}"
        
        # AI Output Construction (Pseudo-Labeling for PoC)
        assistant_content = json.dumps({
            "summary": str(overview)[:50] + "...",
            "analysis": "This movie features strong narrative elements."
        }, ensure_ascii=False)
        
        # Construct Chat Message
        example = TrainingExample(messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ])
        formatted_data.append(example.model_dump())

    # Save to JSONL
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in formatted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    logger.success(f"✅ Preprocessing complete: {output_path} ({len(formatted_data)} items)")

def run_cli():
    """터미널에서 명령어로 실행하기 위한 함수"""
    preprocess_for_training()

if __name__ == "__main__":
    run_cli()