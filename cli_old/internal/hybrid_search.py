import os

from internal.index import InvertedIndex, CACHE_INDEX
from internal.semantic_search import ChunkedSemanticSearch


class HybridSearch():
    def __init__(self, documents: list[dict]):
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
    
    def weighted_search(self, query: str, alpha: float, limit: int=5) -> dict:
        norm_dict: dict[int, dict] = {}


        bm25_results = self._bm25_search(query, limit=limit*500)

        bm_25_scores: list[float] = [] 
        bm_25_doc_ids: list[int] = [] 
        for entry in bm25_results:
            id, score, _ = entry
            bm_25_scores.append(score)
            bm_25_doc_ids.append(id)
        bm25_normalized = normalize_scores(bm_25_scores)

        

        for idx, norm_score in enumerate(bm25_normalized):
            id = bm_25_doc_ids[idx]
            norm_dict[id] = {
                    "id": id,
                    "document": self.idx.docmap.get(id),
                    "semantic_score": 0,
                    "keyword_score": norm_score
                } 
        
        sem_results = self.semantic_search.search_chunks(query, limit=limit*500)
        sem_scores: list[float] = []
        sem_doc_ids: list[float] = []
        for entry in sem_results:
            score = entry.get("score")
            id = entry.get("id")
            sem_scores.append(score)
            sem_doc_ids.append(id)
        sem_normalized = normalize_scores(sem_scores)
        
        for idx, norm_score in enumerate(sem_normalized):
            id = sem_doc_ids[idx]
            if id in norm_dict:
                norm_dict[id]["semantic_score"] = norm_score
            else:
                norm_dict[id] = {
                    "id": id,
                    "document": self.idx.docmap.get(id),
                    "semantic_score": norm_score,
                    "keyword_score": 0,
                } 
        for id in norm_dict:
            hybrid = hybrid_score(norm_dict[id].get("keyword_score"), norm_dict[id].get("semantic_score"), alpha)
            norm_dict[id]["hybrid_score"] = hybrid
        
        norm_dict = dict(sorted(norm_dict.items(), key=lambda entry: entry[1]["hybrid_score"], reverse=True))
        
        if len(norm_dict) > limit:
            norm_dict = dict(list(norm_dict.items())[:limit])
        return norm_dict
        

    def rrf_search(self, query: str, k: float, limit: int=100):
        score_dict = {} # id -> scores
        bm_25_results = self._bm25_search(query, limit=limit*500)
        for idx, bm_25_result in enumerate(bm_25_results):
            _, _, doc = bm_25_result
            rank = idx + 1
            score = rrf_score(rank, k)
            doc_id = doc["id"]
            if doc_id not in score_dict:
                score_dict[doc_id] = {
                    "id": doc_id,
                    "title": doc["title"],
                    "document": doc["description"],
                    "rrf_score": 0.0,
                    "keyword_rank": None,
                    "semantic_rank": None,
                }
            if score_dict[doc_id]["keyword_rank"] is None:
                score_dict[doc_id]["keyword_rank"] = rank
                score_dict[doc_id]["rrf_score"] += score
        
        sem_results = self.semantic_search.search_chunks(query, limit=limit*500)
        for idx, sem_result in enumerate(sem_results):
            rank = idx + 1
            score = rrf_score(rank, k)
            doc_id = sem_result["id"]
            if doc_id not in score_dict:
                 score_dict[doc_id] = {
                    "id": doc_id,
                    "title": doc["title"],
                    "document": doc["description"],
                    "rrf_score": 0.0,
                    "keyword_rank": None,
                    "semantic_rank": None,
                }
            if score_dict[doc_id]["semantic_rank"] is None:
                score_dict[doc_id]["semantic_rank"] = rank
                score_dict[doc_id]["rrf_score"] += score

        score_dict = dict(sorted(score_dict.items(),key=lambda entry: entry[1]["rrf_score"], reverse=True))
        return dict(list(score_dict.items())[:limit])

        

# def reciprocal_rank_fusion(
#     bm25_results: list[dict], semantic_results: list[dict], k: int = RRF_K
# ) -> list[dict]:
#     rrf_scores = {}

#     for rank, result in enumerate(bm25_results, start=1):
#         doc_id = result["id"]
#         if doc_id not in rrf_scores:
#             rrf_scores[doc_id] = {
#                 "title": result["title"],
#                 "document": result["document"],
#                 "rrf_score": 0.0,
#                 "bm25_rank": None,
#                 "semantic_rank": None,
#             }
#         if rrf_scores[doc_id]["bm25_rank"] is None:
#             rrf_scores[doc_id]["bm25_rank"] = rank
#             rrf_scores[doc_id]["rrf_score"] += rrf_score(rank, k)

#     for rank, result in enumerate(semantic_results, start=1):
#         doc_id = result["id"]
#         if doc_id not in rrf_scores:
#             rrf_scores[doc_id] = {
#                 "title": result["title"],
#                 "document": result["document"],
#                 "rrf_score": 0.0,
#                 "bm25_rank": None,
#                 "semantic_rank": None,
#             }
#         if rrf_scores[doc_id]["semantic_rank"] is None:
#             rrf_scores[doc_id]["semantic_rank"] = rank
#             rrf_scores[doc_id]["rrf_score"] += rrf_score(rank, k)

#     rrf_results = []
#     for doc_id, data in rrf_scores.items():
#         result = format_search_result(
#             doc_id=doc_id,
#             title=data["title"],
#             document=data["document"],
#             score=data["rrf_score"],
#             rrf_score=data["rrf_score"],
#             bm25_rank=data["bm25_rank"],
#             semantic_rank=data["semantic_rank"],
#         )
#         rrf_results.append(result)

#     return sorted(rrf_results, key=lambda x: x["score"], reverse=True)


def hybrid_score(bm25_score, semantic_score, alpha=0.5):
        return alpha * bm25_score + (1 - alpha) * semantic_score

def rrf_score(rank, k=60):
    return 1 / (k + rank)

def normalize_scores(scores: list) -> list:
    if len(scores) == 0:
        return []
    low = float(min(scores))
    high = float(max(scores))
    if high == low:
        return list(map(lambda _: 1, scores))
    norm_list = []
    for score in scores:
        score = float(score)
        normalized = (score - low) / (high - low)
        norm_list.append(normalized)
    return norm_list