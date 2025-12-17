from fastapi.testclient import TestClient
from cine_analyst.app.main import app
from unittest.mock import patch, MagicMock
import pytest

client = TestClient(app)

def test_analyze_api_with_mock_agent():
    # 에이전트의 실제 실행(ainvoke)을 가짜로 가로챔
    with patch("cine_analyst.app.api.agent_app.ainvoke") as mock_invoke:
        mock_invoke.return_value = {
            "messages": [MagicMock(content="Mocked Agent Answer")],
            "retrieved_context": ["Mocked Context"]
        }

        response = client.post(
            "/api/v1/analyze",
            json={"query": "테스트 질문"}
        )

        assert response.status_code == 200
        assert response.json()["answer"] == "Mocked Agent Answer"