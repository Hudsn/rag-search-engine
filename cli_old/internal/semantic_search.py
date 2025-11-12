import os, re, json
from sentence_transformers import SentenceTransformer
from internal.dataset import load_movies
import numpy as np

MODEL_SENTENCE_TRANSFORM = "all-MiniLM-L6-v2"
CACHE_BASE = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CACHE_EMBEDDINGS = os.path.join(CACHE_BASE, "cache", "movie_embeddings.npy")
CACHE_CHUNK_EMBEDDINGS = os.path.join(CACHE_BASE, "cache", "chunk_embeddings.npy")
CACHE_CHUNK_METADATA = os.path.join(CACHE_BASE, "cache", "chunk_metadata.json")

class SemanticSearch():
    def __init__(self, model: str=MODEL_SENTENCE_TRANSFORM):
        self.model = SentenceTransformer(model)
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


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str=MODEL_SENTENCE_TRANSFORM):
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
    
    def build_chunk_embeddings(self, documents: list[str]):
        super()._build_docs_and_docmap(documents)
        all_chunks: list[str] = []
        metadata: list[dict] = []
        for idx, doc in enumerate(self.documents):
            desc = doc.get("description")
            if desc is None:
                continue
            doc_chunks = chunk_text_semantic(desc, 4, 1)
            all_chunks.extend(doc_chunks)
            for chunk_idx, _ in enumerate(doc_chunks):
                to_add = {
                    "movie_idx": idx,
                    "chunk_idx": chunk_idx,
                    "total_chunks": len(doc_chunks)
                }
                metadata.append(to_add)
        self.chunk_metadata = metadata
        self.chunk_embeddings = self.model.encode(all_chunks)
        os.makedirs(os.path.dirname(CACHE_CHUNK_EMBEDDINGS), exist_ok=True)
        with open(CACHE_CHUNK_EMBEDDINGS, "wb") as f:
            np.save(f, self.chunk_embeddings)
        os.makedirs(os.path.dirname(CACHE_CHUNK_METADATA), exist_ok=True)
        with open(CACHE_CHUNK_METADATA, "w") as f:
            json.dump(self.chunk_metadata, f)
       
    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        super()._build_docs_and_docmap(documents)
        if os.path.exists(CACHE_CHUNK_EMBEDDINGS) and os.path.exists(CACHE_CHUNK_METADATA):
            with open(CACHE_CHUNK_METADATA, "r") as f:
                self.chunk_metadata = json.load(f)
            with open(CACHE_CHUNK_EMBEDDINGS, "rb") as f:
                self.chunk_embeddings = np.load(f)
            return self.chunk_embeddings
        self.build_chunk_embeddings(documents)
        return self.chunk_embeddings
    
    def search_chunks(self, query: str, limit: int=10):
        query_embedding = self.generate_embedding(query)
        chunk_scores: list[dict] = []
        for idx, chunk_embed in enumerate(self.chunk_embeddings):
            csim = cosine_similarity(query_embedding, chunk_embed)
            to_add = {
                # "absolute_idx": idx,
                # "chunk_idx": self.chunk_metadata[idx].get("chunk_idx"),
                "chunk_idx": idx,
                "movie_idx": self.chunk_metadata[idx].get("movie_idx"),
                "score": csim
            }
            chunk_scores.append(to_add)
        movies_scores: dict[int, dict] = {}

        for chunk_enriched in chunk_scores:
            movie = self.documents[chunk_enriched.get("movie_idx")]
            m_id = movie.get("id")
            if m_id not in movies_scores or movies_scores.get(m_id).get("score") < chunk_enriched.get("score"):
                movies_scores[m_id] = chunk_enriched
        ordered_scores: list[tuple[int, dict]] = list(sorted(movies_scores.items(), key=lambda entry: entry[1].get("score"), reverse=True))
        ret: list[dict] = []
        for m_id, score_dict in ordered_scores[:limit]:
        
            to_add = {
                "id": m_id,
                "title": self.document_map[m_id]["title"],
                "document": self.document_map[m_id]["description"],
                "score": round(score_dict["score"], 4),
                "metadata": self.chunk_metadata[score_dict.get("chunk_idx")] or {}
            }
            ret.append(to_add)
        return ret


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

# 
# Chunking
def chunk_text_semantic(text: str, chunk_size:int, overlap: int) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("size of overlap must be less than the given chunk size")
    chunks: list[str] = []
    chunk_sentence_count = 0
    current_chunk: list[str] = [] 
    text = text.strip()
    if len(text) == 0:
        return []
    sentences = re.split(r"(?<=[!?.])\s+", text)
    if len(sentences) == 1 and not sentences[0].endswith((".","?","!")):
        return [sentences[0]]
    for sentence in sentences:
        if chunk_sentence_count >= chunk_size:
            current_chunk = list(map(lambda entry: entry.strip(), current_chunk))
            current_chunk = list(filter(lambda entry: len(entry) > 0, current_chunk))
            if len(current_chunk) > 0:
                chunks.append(" ".join(current_chunk))
            chunk_sentence_count = 0
            current_chunk = []
            if overlap > 0 and len(chunks) > 0:
                prev_chunk = chunks[-1]
                prev_sentences = re.split(r"(?<=[!?.])\s+", prev_chunk)
                current_chunk = prev_sentences[len(prev_sentences)-overlap:]
                chunk_sentence_count = len(current_chunk)

        sentence = sentence.strip()
        if len(sentence) == 0:
            continue
        current_chunk.append(sentence)
        chunk_sentence_count +=1
    if chunk_sentence_count > 0:
        chunks.append(" ".join(current_chunk))
    return chunks
    

def chunk_text(text: str, chunk_size: int, overlap: int):
    if overlap >= chunk_size:
        raise ValueError("size of overlap must be less than the given chunk size")
    words = text.split(None)
    chunk_word_count = 0
    chunks: list[str] = []
    current_chunk: list[str] = []
    for word in words:
        if chunk_word_count >= chunk_size:
            chunks.append(" ".join(current_chunk))
            chunk_word_count = 0
            current_chunk = []
            if overlap > 0 and len(chunks) > 0: #only enter if we have a previous chunk
                prev_chunk = chunks[-1].split(None)
                current_chunk = prev_chunk[len(prev_chunk)-overlap:]
                chunk_word_count = len(current_chunk)

        current_chunk.append(word)
        chunk_word_count+=1
    if chunk_word_count > 0:
        chunks.append(" ".join(current_chunk))
    return chunks

