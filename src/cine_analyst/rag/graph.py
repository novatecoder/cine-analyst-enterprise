from cine_analyst.rag.base import GraphStoreBase
from cine_analyst.common.config import settings
from neo4j import GraphDatabase
import pandas as pd
from loguru import logger

class GraphSearch(GraphStoreBase):
    def __init__(self):
        # 실제 Neo4j 드라이버 연결
        self.driver = GraphDatabase.driver(
            settings.NEO4J_URI, 
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        )

    def ingest(self, df: pd.DataFrame):
        """영화를 노드로, 감독/배우를 관계로 적재 (Cypher 사용)"""
        with self.driver.session() as session:
            for _, row in df.iterrows():
                query = """
                MERGE (m:Movie {id: $id, title: $title})
                MERGE (p:Person {name: $director})
                MERGE (p)-[:DIRECTED]->(m)
                """
                session.run(query, id=row['id'], title=row['title'], director=row['director'])
        logger.info("✅ Neo4j 그래프 데이터 적재 완료")

    def get_related_movies(self, title: str):
        """특정 영화와 감독이 같은 다른 영화 찾기 (관계형 검색)"""
        with self.driver.session() as session:
            query = """
            MATCH (m:Movie {title: $title})<-[:DIRECTED]-(d:Person)-[:DIRECTED]->(other:Movie)
            RETURN other.title AS title
            """
            result = session.run(query, title=title)
            return [record["title"] for record in result]