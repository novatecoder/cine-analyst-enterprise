import os
import json
from cine_analyst.data.preprocessor import preprocess_for_training

def test_preprocess_logic(mock_movie_df, tmp_path):
    """임의 데이터를 통한 JSONL 변환 로직 검증"""
    raw_path = tmp_path / "raw.csv"
    mock_movie_df.to_csv(raw_path, index=False)
    processed_path = tmp_path / "train.jsonl"
    
    preprocess_for_training(input_path=str(raw_path), output_path=str(processed_path))
    
    assert os.path.exists(processed_path)
    with open(processed_path, 'r') as f:
        first_line = json.loads(f.readline())
        assert "messages" in first_line # 스키마 구조 확인