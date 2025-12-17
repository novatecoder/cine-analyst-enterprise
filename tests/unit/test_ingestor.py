from unittest.mock import patch, MagicMock
from cine_analyst.data.ingestor import run_ingestion

@patch('cine_analyst.data.ingestor.OpenSearchStore')
@patch('cine_analyst.data.ingestor.Neo4jStore')
@patch('os.path.exists')
def test_ingest_pipeline_calls(mock_exists, mock_neo, mock_os, mock_movie_df, tmp_path):
    """DB 연결 없이 인제션 파이프라인 흐름 검증"""
    mock_exists.return_value = True
    raw_path = tmp_path / "raw.csv"
    
    with patch('pandas.read_csv', return_value=mock_movie_df):
        run_ingestion(input_path=str(raw_path), sample_size=2)
    
    # 각 저장소의 ingest 메서드 호출 여부 확인
    mock_os.return_value.ingest.assert_called_once()
    mock_neo.return_value.ingest.assert_called_once()