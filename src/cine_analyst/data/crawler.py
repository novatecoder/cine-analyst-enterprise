import os
import io
import requests
import pandas as pd
from loguru import logger
from cine_analyst.common.config import settings

TARGET_URL = "https://raw.githubusercontent.com/CTopham/TophamRepo/master/Movie%20Project/Resources/tmdb_5000_movies.csv"

def download_raw_data(output_path: str = settings.RAW_DATA_PATH):
    """
    외부 소스에서 Raw Data를 다운로드합니다.
    Args:
        output_path: 저장 경로 (Dependency Injection)
    """
    logger.info(f"🚀 Downloading raw data from {TARGET_URL}")
    
    try:
        response = requests.get(TARGET_URL)
        response.raise_for_status()
        
        df = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        
        logger.success(f"✅ Download complete: {output_path} ({len(df)} rows)")
        return output_path
        
    except Exception as e:
        logger.error(f"❌ Failed to download data: {e}")
        raise

def run_cli():
    download_raw_data()