from unittest.mock import patch, MagicMock
from cine_analyst.data.crawler import download_raw_data

@patch('requests.get')
def test_download_raw_data_success(mock_get, tmp_path):
    """네트워크 연결 없이 다운로드 로직 검증"""
    csv_content = "title,overview\nMovie1,Overview1"
    mock_get.return_value.status_code = 200
    mock_get.return_value.content = csv_content.encode('utf-8')
    
    # tmp_path를 사용하여 실제 파일 시스템을 건드리지 않음
    save_path = tmp_path / "raw.csv"
    download_raw_data(output_path=str(save_path))
    
    assert save_path.exists()