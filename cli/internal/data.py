import json

def load_movies(path: str) -> dict:
    data = None
    with open(path) as f:
        data = json.load(f)
    return data

def search_titles(title: str, movie_d: dict):
    movies = movie_d["movies"]
    movies = sorted(movies, key=lambda entry: entry["id"])
    matches = 0
    for movie in movies:
        if matches >= 5:
            break
        if title.lower() in str(movie["title"]).lower():
            matches+=1
            print(f"{matches}. {movie["title"]}")