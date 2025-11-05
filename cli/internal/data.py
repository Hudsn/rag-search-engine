import json
import string
from nltk.stem import PorterStemmer

def search_titles(target: str, movie_d: dict, stemmer: PorterStemmer, stops: list[str]=[]):
    movies = movie_d["movies"]
    movies = sorted(movies, key=lambda entry: entry["id"])
    matches = 0
    for movie in movies:
        if matches >= 5:
            break
        query_toks = tokenize_text(target, stemmer, stops)
        title_toks = tokenize_text(movie["title"], stemmer, stops)
        if is_token_match(query_toks, title_toks):
            matches+=1
            print(f"{matches}. {movie["title"]}")



def load_movies(path: str) -> dict:
    data = None
    with open(path) as f:
        data = json.load(f)
    return data

def get_stopword_list(path: str) -> list[str]:
    ret = []
    with open(path) as f:
        ret = f.read().splitlines()
    return ret

def tokenize_text(text: str, stemmer: PorterStemmer, stopwords: list[str]) -> list[str]:
    text = text.lower()
    text = str.translate(text, str.maketrans("", "", string.punctuation))
    toks = text.lower().split(None)
    toks = remove_stops(toks, stopwords)
    toks = list(map(lambda entry: stemmer.stem(entry), toks))
    return toks

def remove_stops(tokens: list[str], stops: list[str]) -> list[str]:
    return list(filter(lambda entry: entry not in stops, tokens))

def is_token_match(q_tokens: list[str], t_tokens: list[str]) -> bool:
    for qtok in q_tokens:
        for ttok in t_tokens:
            if qtok in ttok:
                return True
    return False

