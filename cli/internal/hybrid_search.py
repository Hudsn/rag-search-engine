import os

from internal.index import InvertedIndex, CACHE_INDEX
from internal.semantic_search import ChunkedSemanticSearch


class HybridSearch():
    def __init__(self, documents: list[str]):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(CACHE_INDEX):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int):
        self.idx.load()
        return self.idx.bm25_search(query, limit)
    
    def weighted_search(self, query: str, alpha: float, limit: int=5):
        raise NotImplementedError("Weighted hybrid search is not implemented yet.")

    def rrf_search(self, query: str, k: float, limit: int=100):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")