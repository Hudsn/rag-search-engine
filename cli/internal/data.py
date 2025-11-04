import json
import string

def load_movies(path: str) -> dict:
    data = None
    with open(path) as f:
        data = json.load(f)
    return data

def search_titles(target: str, movie_d: dict):
    movies = movie_d["movies"]
    movies = sorted(movies, key=lambda entry: entry["id"])
    matches = 0
    for movie in movies:
        if matches >= 5:
            break
        transform_table = str.maketrans("", "", string.punctuation)
        candidate = str.translate(movie["title"], transform_table).lower()
        str.translate
        if target.lower() in candidate:
            matches+=1
            print(f"{matches}. {movie["title"]}")