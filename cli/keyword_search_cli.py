#!/usr/bin/env python3

import argparse
import os

from internal.dataset import load_movies, get_stopword_list
from internal.tokens import tokenize_text, is_token_match
from internal.index import InvertedIndex, BM25_K1, BM25_B



def main() -> None:

    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build an inverted index of the movie list")

    tf_parser = subparsers.add_parser("tf", help="Search the index for the term frequency in a specific document")
    tf_parser.add_argument("doc_id", type=int, help="The document id to query against")
    tf_parser.add_argument("term", type=str, help="The term to query")

    idf_parser = subparsers.add_parser("idf", help="Search the entire index for movies matching a term based on inverse-documemt frequency (idf)")
    idf_parser.add_argument("term", type=str, help="the term to search for idf matching")

    tf_idf_parser = subparsers.add_parser("tfidf", help="Search the entire index for movies matching a term based on term-frequency-invers-document-frequency (tf-idf)")
    tf_idf_parser.add_argument("doc_id", type=int, help="The document id to query against")
    tf_idf_parser.add_argument("term", type=str, help="The term to query")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Search the entire index for movies matching a term based on the BM25-IDF score")
    bm25_idf_parser.add_argument("term", type=str, help="The term to query")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="The term to query")
    bm25_tf_parser.add_argument("k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter")

    bm25_search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25_search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            idx = InvertedIndex()
            load_index(idx)
            matches = search_index(args.query, idx)
            for movie in matches:
                print(f"{movie["title"]} {movie["id"]}")

        case "build":
            idx = InvertedIndex()
            idx.build()
            save_index(idx)
          
        case "tf":
            idx = InvertedIndex()
            load_index(idx)
            freq = idx.get_tf(args.doc_id, args.term)
            print(f"frequency: {freq}, document_id: {args.doc_id}")
        
        case "idf":
            idx = InvertedIndex()
            load_index(idx)
            idf = idx.get_idf(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case "tfidf":
            idx = InvertedIndex()
            load_index(idx)
            tfidf = idx.get_tfidf(args.doc_id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tfidf:.2f}")
        
        case "bm25idf":
            idx = InvertedIndex()
            load_index(idx)
            bm25idf = idx.get_bm25_idf(args.term)
            print(f"BM25 IDF score of '{args.term}: {bm25idf:.2f}")

        case "bm25tf":
            idx = InvertedIndex()
            load_index(idx)
            bm25tf = idx.get_bm25_tf(args.doc_id, args.term, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
        
        case "bm25search":
            idx = InvertedIndex()
            load_index(idx)
            bm25search_result = idx.bm25_search(args.query, 5) 
            for i, result_tuple in enumerate(bm25search_result):
                doc_id = result_tuple[0]
                score = result_tuple[1]
                movie = result_tuple[2]
                print(f"{i}. ({doc_id}) {movie["title"]} - Score: {score:.2f}")

        case _:
            parser.print_help()



def load_index(idx: InvertedIndex):
    try:
        idx.load()
    except FileNotFoundError as fe:
        print(f"Unable to load data into the index. One or more files missing: {fe}")
        print("Try running the 'build' command to generate a new index and create this file.")
        os._exit(1) 
    except Exception as e:
        print(f"Unable to load data into the index: {e}")
        os._exit(1)

def save_index(idx: InvertedIndex):
    try:
        idx.save()
    except Exception as e:
        print(f"Unable to save index data to disk: {e}")
        os._exit(1)


def search_index(text: str, idx: InvertedIndex) -> list[dict]:
    q_toks = idx.tokenize(text)
    ids = []
    for tok in q_toks:
        if len(ids) >= 5:
            ids = ids[:5]
            break
        results = idx.get_documents(tok)
        if len(results) == 0:
            continue
        remaining_slots = 5 - len(ids)
        if len(results) >= remaining_slots:
            results = results[:remaining_slots]
        ids.extend(results)
        ids = list(set(ids))


    ret: list[dict] = []
    for id in ids:
        movie = idx.docmap.get(id)
        if movie == None:
            continue
        ret.append(movie)
    return ret


# def search_titles(target: str):
#     stemmer =  PorterStemmer()
#     movies_dict = load_movies()
#     stop_words = get_stopword_list()
#     movies = movies_dict["movies"]
#     movies = sorted(movies, key=lambda entry: entry["id"])
#     matches = 0
#     for movie in movies:
#         if matches >= 5:
#             break
#         query_toks = tokenize_text(target, stemmer, stop_words)
#         title_toks = tokenize_text(movie["title"], stemmer, stop_words)
#         if is_token_match(query_toks, title_toks):
#             matches+=1
#             print(f"{matches}. {movie["title"]}")


if __name__ == "__main__":
    main()