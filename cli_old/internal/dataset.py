import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MOVIE_FILE = os.path.join(BASE_DIR, "data", "movies.json")
STOPWORD_FILE = os.path.join(BASE_DIR, "data", "stopwords.txt")
GOLDEN_FILE = os.path.join(BASE_DIR, "data", "golden_dataset.json")

def load_golden() -> dict:
    ret = None
    with open(GOLDEN_FILE) as f:
        ret = json.load(f)
    return ret

def load_movies() -> dict:
    ret = None
    with open(MOVIE_FILE) as f:
        ret = json.load(f)
    return ret

def get_stopword_list() -> list[str]:
    ret = []
    with open(STOPWORD_FILE) as f:
        ret = f.read().splitlines()
    return ret
