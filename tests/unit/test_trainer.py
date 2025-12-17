from unittest.mock import patch
from cine_analyst.training.trainer import train_model

@patch('cine_analyst.training.trainer.logger')
def test_trainer_import_error_handling(mock_logger):
    """unsloth 라이브러리 미설치 시 에러 핸들링 검증"""
    with patch('builtins.__import__', side_effect=ImportError):
        train_model()
        # "not found" 에러 로그가 찍혔는지 확인
        mock_logger.error.assert_called()