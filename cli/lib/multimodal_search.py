from PIL import Image
from sentence_transformers import SentenceTransformer
from lib.semantic_search import cosine_similarity
# from transformers import AutoModel

class MultimodalSearch():
    def __init__(self, model_name="clip-ViT-B-32", docs: list[dict]=[]):
        self.model = SentenceTransformer(model_name)
        self.documents = docs
        self.texts: list[str] = []

        for doc in self.documents:
            self.texts.append(f"{doc["title"]}: {doc["description"]}")

        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, img_path: str):
        image = Image.open(img_path)
        embeddings = self.model.encode([image])
        return embeddings[0]
    
    def search_with_image(self, img_path:str):
        embedding = self.embed_image(img_path)
        results: list[dict] = []
        for doc_idx, text_emb in enumerate(self.text_embeddings):
            csim = cosine_similarity(embedding, text_emb)
            doc = self.documents[doc_idx]
            results.append({
                "id": doc["id"],
                "title": doc["title"],
                "description": doc["description"],
                "score": csim,
            })
        return list(sorted(results, key=lambda entry: entry["score"], reverse=True))[:5]

def search_image(img_path: str, docs: list[dict]) -> list[dict]:
    ms = MultimodalSearch(docs=docs)
    return ms.search_with_image(img_path=img_path)

def verify_image_embedding(img_path: str):
    ms = MultimodalSearch()
    embedding = ms.embed_image(img_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")
