from unittest.mock import patch, MagicMock
import pytest
from cine_analyst.training.trainer import train_model

@patch('cine_analyst.training.trainer.logger')
# wandb는 함수 내부에서 import되므로 'wandb'라는 이름이 trainer 모듈에 등록되지 않았을 수 있음
# sys.modules를 조작하거나 함수 내부 patch를 권장하지만, 일단 아래와 같이 수정
def test_trainer_import_error_handling(mock_logger):
    """unsloth 라이브러리 미설치 시 에러 핸들링 검증"""
    
    # 1. 특정 라이브러리 임포트 시 실패하도록 설정
    with patch('builtins.__import__', side_effect=ImportError):
        # 2. wandb도 임포트 실패 대상에 포함됨
        train_model()
        
        # 3. logger.error가 호출되었는지 확인
        mock_logger.error.assert_called()