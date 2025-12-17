from cine_analyst.rag.base import VectorStoreBase
from cine_analyst.common.config import settings
from opensearchpy import OpenSearch
import pandas as pd
from loguru import logger

class VectorSearch(VectorStoreBase):
    def __init__(self):
        # 실제 OpenSearch 클라이언트 연결
        self.client = OpenSearch(
            hosts=[settings.OPENSEARCH_URL],
            http_compress=True,
            use_ssl=False,
            verify_certs=False,
        )
        self.index_name = "movies"

    def ingest(self, df: pd.DataFrame):
        """데이터프레임을 OpenSearch 인덱스에 벌크 적재"""
        for _, row in df.iterrows():
            doc = row.to_dict()
            self.client.index(index=self.index_name, body=doc)
        logger.info(f"✅ OpenSearch에 {len(df)}건 적재 완료")

    def search(self, query: str, k: int = 5):
        """실제 OpenSearch 쿼리 실행"""
        body = {
            "size": k,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["title", "overview"]
                }
            }
        }
        response = self.client.search(index=self.index_name, body=body)
        # 검색 결과 가공
        return [hit["_source"] for hit in response["hits"]["hits"]]