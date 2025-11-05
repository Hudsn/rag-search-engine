import os, math, pickle, collections
from nltk.stem import PorterStemmer
from internal.tokens import tokenize_text
from internal.dataset import load_movies, get_stopword_list

CACHE_FILE_BASE = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) 
CACHE_DOCMAP = os.path.join("cache", "docmap.pkl")
CACHE_INDEX = os.path.join("cache", "index.pkl")
CACHE_FREQS = os.path.join("cache", "term_frequencies.pkl")

class InvertedIndex:
    _stemmer = PorterStemmer()
    def __init__(self):
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, collections.Counter] = {}

    def _add_document(self, doc_id: int, text: str):
        toks = tokenize_text(text, self._stemmer, get_stopword_list())
        for tok in toks:
            if tok in self.index:
                self.index[tok].add(doc_id)
            else:
                self.index[tok] = {doc_id}
            if doc_id not in self.term_frequencies:
                self.term_frequencies[doc_id] = collections.Counter()
            self.term_frequencies[doc_id][tok] += 1

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term.lower())
        if doc_ids == None:
            return []
        return sorted(list(doc_ids))
    
    def get_tf(self, doc_id: int, term: str) -> int:
        counter = self.term_frequencies.get(doc_id)
        if counter == None:
            return 0
        term_toks = tokenize_text(term, self._stemmer, get_stopword_list())
        if len(term_toks) > 1:
            raise ValueError("term must be a single token")
        term = term_toks[0]
        return counter.get(term.lower(), 0)
    
    def get_idf(self, term: str) -> float:
        tokens = self.tokenize(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        term = tokens[0]
        doc_count = len(self.docmap)
        terms_doc_count = len(self.get_documents(term))
        return math.log((doc_count+1) / (terms_doc_count + 1))

    def get_tfidf(self, doc_id: int, term: str) -> float:
        tokens = self.tokenize(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        term = tokens[0]
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf

    def build(self):
        movies = load_movies()
        for movie in movies["movies"]:
            id = movie.get("id")
            if id == None:
                continue
            self.docmap[int(id)] = movie
            index_key = f"{movie["title"]} {movie["description"]}"
            self._add_document(id, index_key)

    def save(self):
            mkdirs_for_file(CACHE_DOCMAP)
            with open(CACHE_DOCMAP, "wb") as f:
                pickle.dump(self.docmap, f)
            mkdirs_for_file(CACHE_INDEX)
            with open(CACHE_INDEX, "wb") as f:
                pickle.dump(self.index, f)
            mkdirs_for_file(CACHE_FREQS)
            with open(CACHE_FREQS, "wb") as f:
                pickle.dump(self.term_frequencies, f)
      
    def load(self):
        with open(CACHE_DOCMAP, "rb") as f:
            self.docmap = pickle.load(f)
        with open(CACHE_INDEX, "rb") as f:
            self.index = pickle.load(f)
        with open(CACHE_FREQS, "rb") as f:
            self.term_frequencies = pickle.load(f)

        
    def tokenize(self, text: str) -> list[str]:
        return tokenize_text(text, self._stemmer, get_stopword_list())




def mkdirs_for_file(file_path: str):
    dir = os.path.dirname(file_path)
    os.makedirs(dir, exist_ok=True)