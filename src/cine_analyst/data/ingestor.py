import json
import pandas as pd
from loguru import logger
from opensearchpy import OpenSearch, helpers
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

from cine_analyst.common.config import settings

def get_opensearch_client():
    return OpenSearch(
        hosts=[settings.OPENSEARCH_URL],
        http_compress=True, use_ssl=False, verify_certs=False
    )

def ingest_vector_db(df: pd.DataFrame):
    """OpenSearch에 임베딩 벡터 적재"""
    client = get_opensearch_client()
    embedder = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
    
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
    
    if not client.indices.exists(index=index_name):
        client.indices.create(index=index_name, body=index_body)
    
    requests = []
    logger.info("Generating embeddings for Vector DB...")
    
    # Batch processing recommended for production
    for _, row in df.iterrows():
        if pd.isna(row['overview']): continue
        
        vector = embedder.encode(row['overview']).tolist()
        doc = {
            "_index": index_name,
            "_source": {
                "title": row['title'],
                "overview": row['overview'],
                "overview_vector": vector
            }
        }
        requests.append(doc)
    
    helpers.bulk(client, requests)
    logger.success(f"✅ Vector DB Ingestion complete: {len(requests)} docs")

def ingest_graph_db(df: pd.DataFrame):
    """Neo4j에 지식 그래프 적재"""
    driver = GraphDatabase.driver(
        settings.NEO4J_URI, 
        auth=(settings.NEO4J_AUTH_USER, settings.NEO4J_AUTH_PASS)
    )
    
    query_create = """
    MERGE (m:Movie {title: $title})
    SET m.overview = $overview
    WITH m
    UNWIND $genres as g_data
    MERGE (g:Genre {name: g_data.name})
    MERGE (m)-[:HAS_GENRE]->(g)
    """
    
    logger.info("Ingesting Knowledge Graph to Neo4j...")
    with driver.session() as session:
        # Constraints
        session.run("CREATE CONSTRAINT movie_title IF NOT EXISTS FOR (m:Movie) REQUIRE m.title IS UNIQUE")
        
        for _, row in df.iterrows():
            try:
                genres = json.loads(row['genres'])
                session.run(query_create, 
                            title=row['title'], 
                            overview=str(row['overview']), 
                            genres=genres)
            except Exception:
                continue
                
    driver.close()
    logger.success("✅ Graph DB Ingestion complete")

def run_ingestion(input_path: str = settings.RAW_DATA_PATH):
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        return

    df = pd.read_csv(input_path).head(100) # PoC용 샘플링
    
    try:
        ingest_vector_db(df)
        ingest_graph_db(df)
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")

def run_cli():
    run_ingestion()