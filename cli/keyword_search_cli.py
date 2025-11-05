#!/usr/bin/env python3

import argparse
from internal.data import load_movies, search_titles, get_stopword_list
from nltk.stem import PorterStemmer

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            # print the search query here
            movies_dict = load_movies("data/movies.json")
            stop_words = get_stopword_list("data/stopwords.txt")
            search_titles(args.query, movies_dict, PorterStemmer(), stop_words)
            print(f"Searching for: {args.query}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()