import os
import json
import click
import pandas as pd
from loguru import logger
from opensearchpy import OpenSearch, helpers
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

from cine_analyst.common.config import settings
from cine_analyst.rag.base import VectorStoreBase, GraphStoreBase

class OpenSearchStore(VectorStoreBase):
    """OpenSearch를 이용한 VectorStore 구현체"""
    def __init__(self):
        self.client = OpenSearch(
            hosts=[settings.OPENSEARCH_URL],
            http_compress=True, 
            use_ssl=False, 
            verify_certs=False
        )
        self.embedder = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)

    def ingest(self, df: pd.DataFrame):
        index_name = settings.OPENSEARCH_INDEX
        index_body = {
            "settings": {"index": {"knn": True}},
            "mappings": {
                "properties": {
                    "title": {"type": "text"},
                    "overview": {"type": "text"},
                    "overview_vector": {
                        "type": "knn_vector", "dimension": 384, "method": {"name": "hnsw", "engine": "nmslib"}
                    }
                }
            }
        }
        
        if not self.client.indices.exists(index=index_name):
            self.client.indices.create(index=index_name, body=index_body)
        
        requests = []
        logger.info(f"Generating embeddings for {len(df)} docs...")
        
        for _, row in df.iterrows():
            if pd.isna(row['overview']): continue
            
            vector = self.embedder.encode(row['overview']).tolist()
            doc = {
                "_index": index_name,
                "_source": {
                    "title": row['title'],
                    "overview": row['overview'],
                    "overview_vector": vector
                }
            }
            requests.append(doc)
        
        helpers.bulk(self.client, requests)
        logger.success(f"✅ Vector DB Ingestion complete: {len(requests)} docs")

    def search(self, query: str, k: int = 5):
        """벡터 검색 구현 (필수 추상 메서드)"""
        query_vector = self.embedder.encode(query).tolist()
        
        search_query = {
            "size": k,
            "query": {
                "knn": {
                    "overview_vector": {
                        "vector": query_vector,
                        "k": k
                    }
                }
            }
        }
        
        response = self.client.search(
            index=settings.OPENSEARCH_INDEX,
            body=search_query
        )
        return response

class Neo4jStore(GraphStoreBase):
    """Neo4j를 이용한 GraphStore 구현체"""
    def __init__(self):
        self.driver = GraphDatabase.driver(
            settings.NEO4J_URI, 
            auth=(settings.NEO4J_AUTH_USER, settings.NEO4J_AUTH_PASS)
        )

    def ingest(self, df: pd.DataFrame):
        query_create = """
        MERGE (m:Movie {title: $title})
        SET m.overview = $overview
        WITH m
        UNWIND $genres as g_data
        MERGE (g:Genre {name: g_data.name})
        MERGE (m)-[:HAS_GENRE]->(g)
        """
        
        logger.info("Ingesting Knowledge Graph to Neo4j...")
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT movie_title IF NOT EXISTS FOR (m:Movie) REQUIRE m.title IS UNIQUE")
            
            count = 0
            for _, row in df.iterrows():
                try:
                    genres = json.loads(row['genres'])
                    session.run(query_create, 
                                title=row['title'], 
                                overview=str(row['overview']), 
                                genres=genres)
                    count += 1
                except Exception:
                    continue
        
        self.driver.close()
        logger.success(f"✅ Graph DB Ingestion complete: {count} nodes created")

def run_ingestion(input_path: str, sample_size: int = 100):
    """전체 인제션 파이프라인 실행 엔진"""
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        return

    # 데이터 로드 및 샘플링 적용
    df = pd.read_csv(input_path).head(sample_size)
    
    # 추상화된 구현체 사용 (의존성 주입 형태)
    vector_store = OpenSearchStore()
    graph_store = Neo4jStore()
    
    try:
        vector_store.ingest(df)
        graph_store.ingest(df)
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")

@click.command()
@click.option('--input', 'input_path', default=settings.RAW_DATA_PATH, help='적재할 원본 CSV 경로')
@click.option('--limit', 'limit', default=100, type=int, help='적재할 최대 데이터 개수')
def run_cli(input_path, limit):
    """
    CLI 명령어 실행. 
    인자가 있으면 입력받은 값을 사용하고, 없으면 config의 기본값을 사용합니다.
    """
    run_ingestion(input_path=input_path, sample_size=limit)