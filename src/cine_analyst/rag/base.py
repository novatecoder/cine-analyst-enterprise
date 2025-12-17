from abc import ABC, abstractmethod
import pandas as pd

class VectorStoreBase(ABC):
    @abstractmethod
    def ingest(self, df: pd.DataFrame): pass
    
    @abstractmethod
    def search(self, query: str, k: int = 5): pass

class GraphStoreBase(ABC):
    @abstractmethod
    def ingest(self, df: pd.DataFrame): pass