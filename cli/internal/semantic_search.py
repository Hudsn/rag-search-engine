import os
from sentence_transformers import SentenceTransformer
from internal.dataset import load_movies
import numpy as np

MODEL_SENTENCE_TRANSFORM = "all-MiniLM-L6-v2"
CACHE_BASE = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CACHE_EMBEDDINGS = os.path.join(CACHE_BASE, "cache", "movie_embeddings.npy")

class SemanticSearch():
    def __init__(self):
        self.model = SentenceTransformer(MODEL_SENTENCE_TRANSFORM)
        self.embeddings: np.ndarray = None
        self.documents: list[dict] = None
        self.document_map: dict[int, dict] = {}

    def search(self, query: str, limit: int) -> list[dict]:
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        q_embed = self.generate_embedding(query)
        sim_list: list[tuple[float, dict]] = []
        for i in range(len(self.embeddings)):
            doc_embedding = self.embeddings[i]
            csim = cosine_similarity(q_embed, doc_embedding) 
            sim_list.append((csim, self.documents[i]))
        sim_list_ordered = sorted(sim_list, key=lambda entry: entry[0], reverse=True)
        if len(sim_list_ordered) > limit:
            sim_list_ordered = sim_list_ordered[:limit]
        ret: list[dict] = []
        for entry in sim_list_ordered:
            to_add = {
                "score": entry[0],
                "title": entry[1]["title"],
                "description": entry[1]["description"]
            }
            ret.append(to_add)
        return ret

    def generate_embedding(self, text: str):
        embedding = self.model.encode([text]) 
        return embedding[0]
    
    def build_embeddings(self, documents: list[dict]) -> np.ndarray:
        self._build_docs_and_docmap(documents)
        movie_strings: list[str] = []
        for movie in self.documents:
            movie_strings.append(f"{movie["title"]}: {movie["description"]}")
        os.makedirs(os.path.dirname(CACHE_EMBEDDINGS), exist_ok=True)
        self.embeddings = self.model.encode(movie_strings, show_progress_bar=True)
        with open(CACHE_EMBEDDINGS, "wb") as f:
            np.save(f, self.embeddings)
        return self.embeddings
    
    def load_or_create_embeddings(self, documents: list[dict]) -> np.ndarray:
        self._build_docs_and_docmap(documents)
        if os.path.exists(CACHE_EMBEDDINGS):
            with open(CACHE_EMBEDDINGS, "rb") as f:
                self.embeddings = np.load(f)
                return self.embeddings
        return self.build_embeddings(documents)

    def _build_docs_and_docmap(self, documents: list[dict]):
        self.documents = documents
        for movie in documents:
            doc_map_key = movie["id"]
            self.document_map[doc_map_key] = movie


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_prod = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_prod / (norm1 * norm2)

def embed_query_text(query: str) -> np.ndarray:
    sem = SemanticSearch()
    embeddings = sem.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embeddings[:5]}")
    print(f"Shape: {embeddings.shape}")

def verify_embeddings():
    sem = SemanticSearch()
    movies = load_movies()
    li = movies["movies"]
    embeddings = sem.load_or_create_embeddings(li)
    print(f"Number of docs: {len(sem.documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_text(text):
    sem = SemanticSearch()
    embedding = sem.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_model():
    search = SemanticSearch()
    print(f"Model loaded: {search.model}")
    print(f"Max sequence length: {search.model.max_seq_length}")