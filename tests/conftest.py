import pytest
import pandas as pd
import json

@pytest.fixture
def mock_movie_df():
    """테스트용 임의의 영화 데이터 생성"""
    data = {
        'id': [1, 2],
        'title': ['Test Movie 1', 'Test Movie 2'],
        'overview': ['Description 1 for testing.', 'Description 2 for testing.'],
        'genres': [
            json.dumps([{"id": 28, "name": "Action"}]),
            json.dumps([{"id": 35, "name": "Comedy"}])
        ]
    }
    return pd.DataFrame(data)