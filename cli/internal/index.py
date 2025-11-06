import os, math, pickle, collections
from nltk.stem import PorterStemmer
from internal.tokens import tokenize_text
from internal.dataset import load_movies, get_stopword_list

BM25_K1 = 1.5 #parameter that determines how steep the diminishing returns are for BM25 term-frequency. Higher = more steep falloff
BM25_B = 0.75 # parameter that determines how much we want to normalize on document length (0 to 1). 0 = we don't normalize document length at all

CACHE_FILE_BASE = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) 
CACHE_DOCMAP = os.path.join("cache", "docmap.pkl")
CACHE_INDEX = os.path.join("cache", "index.pkl")
CACHE_FREQS = os.path.join("cache", "term_frequencies.pkl")
CACHE_DOCLEN = os.path.join("cache", "doc_lengths.pkl")

class InvertedIndex:
    _stemmer = PorterStemmer()
    def __init__(self):
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, collections.Counter] = {}
        self.doc_lengths: dict[int, int] = {}

    def _get_avg_doc_length(self) -> float:
        if len(self.doc_lengths) == 0:
            raise ValueError("index has empty document length store. try rebuilding the index")
        sum = 0
        for id in self.doc_lengths:
            sum += self.doc_lengths.get(id, 0)
        return sum / len(self.doc_lengths)

    def _add_document(self, doc_id: int, text: str):
        toks = tokenize_text(text, self._stemmer, get_stopword_list())
        for tok in toks:
            # index
            if tok in self.index:
                self.index[tok].add(doc_id)
            else:
                self.index[tok] = {doc_id}
            
            # frequencies
            if doc_id not in self.term_frequencies:
                self.term_frequencies[doc_id] = collections.Counter()
            self.term_frequencies[doc_id][tok] += 1

        # doc_length
        self.doc_lengths[doc_id] = len(toks)

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
    
    def get_bm25_idf(self, term: str) -> float:
        tokens = self.tokenize(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        term = tokens[0]
        total_count = len(self.docmap)
        terms_doc_count = len(self.get_documents(term))
        no_terms_doc_count = total_count - terms_doc_count
        return math.log((no_terms_doc_count + 0.5) / (terms_doc_count + 0.5) + 1)

    def get_bm25_tf(self, doc_id: int, term: str, k1=BM25_K1, b=BM25_B) -> int:
        tokens = self.tokenize(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        term = tokens[0]
        tf_raw = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths.get(doc_id, 0)
        avg_doc_length = self._get_avg_doc_length()
        length_norm = 1 - b + b * (doc_length / avg_doc_length)
        return (tf_raw * (k1 + 1) / (tf_raw + k1 * length_norm))
    
    def get_bm25(self, doc_id: int, term: str) -> float:
        tf = self.get_bm25_tf(doc_id, term)
        idf = self.get_bm25_idf(term)
        return tf * idf

    def bm25_search(self, query: str, limit: int) -> list[tuple]:
        tokens = self.tokenize(query)
        scores: dict[int, float] = {}
        for doc_id in self.docmap:
            doc = self.docmap.get(doc_id)
            if doc == None:
                continue
            for tok in tokens:
                tok_score = self.get_bm25(doc_id, tok)
                if doc_id in scores:
                    scores[doc_id] += tok_score
                else:
                    scores[doc_id] = tok_score
        
        desc_scores = dict(sorted(scores.items(), key=lambda entry: entry[1], reverse=True))
        ret = []
        for doc_id in desc_scores:
            if len(ret) >= limit:
                break
            score = desc_scores[doc_id]
            doc = self.docmap[doc_id]
            ret.append((doc_id, score, doc))
        return ret

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
            mkdirs_for_file(CACHE_DOCLEN)
            with open(CACHE_DOCLEN, "wb") as f:
                pickle.dump(self.doc_lengths, f)
      
    def load(self):
        with open(CACHE_DOCMAP, "rb") as f:
            self.docmap = pickle.load(f)
        with open(CACHE_INDEX, "rb") as f:
            self.index = pickle.load(f)
        with open(CACHE_FREQS, "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(CACHE_DOCLEN, "rb") as f:
            self.doc_lengths = pickle.load(f)

        
    def tokenize(self, text: str) -> list[str]:
        return tokenize_text(text, self._stemmer, get_stopword_list())




def mkdirs_for_file(file_path: str):
    dir = os.path.dirname(file_path)
    os.makedirs(dir, exist_ok=True)