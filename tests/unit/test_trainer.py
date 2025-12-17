from unittest.mock import patch, MagicMock
from cine_analyst.training.trainer import train_model

@patch('cine_analyst.training.trainer.logger')
@patch('cine_analyst.training.trainer.wandb') # wandb 모킹 추가
def test_trainer_import_error_handling(mock_wandb, mock_logger):
    """라이브러리 미설치 시 에러 핸들링 및 모니터링 미작동 검증"""
    with patch('builtins.__import__', side_effect=ImportError):
        train_model()
        # "not found" 에러 로그 확인
        mock_logger.error.assert_called()
        # 라이브러리 없으면 wandb도 실행되지 않아야 함
        mock_wandb.init.assert_not_called()