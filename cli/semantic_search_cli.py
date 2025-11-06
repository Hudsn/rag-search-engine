#!/usr/bin/python

import argparse, textwrap
from internal.semantic_search import SemanticSearch, verify_model, embed_text, verify_embeddings, embed_query_text
from internal.dataset import load_movies
def main():

    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    subparser = parser.add_subparsers(dest="command", help="Available commands")

    # verify
    subparser.add_parser("verify", help="print the model information used by the semantic search engine")

    # embed
    embeddings_parser = subparser.add_parser("embed_text", help="Generate embeddings for the given text")
    embeddings_parser.add_argument("text", type=str, help="The target text to embed")

    # verify_embeddings
    subparser.add_parser("verify_embeddings", help="Generate embeddings for the movie corpus")

    # embed_query
    embed_query_parser = subparser.add_parser("embedquery", help="Generate embeddings for the given query text")
    embed_query_parser.add_argument("query", type=str, help="The query text to generate embeeddings")

    # search
    search_parser = subparser.add_parser("search", help="Search the movie database for semantic matches")
    search_parser.add_argument("query", type=str, help="The term to search")
    search_parser.add_argument("--limit", type=int, nargs="?", default=5)

    args = parser.parse_args()



    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            sem = SemanticSearch()
            m_list = load_movies().get("movies")
            sem.load_or_create_embeddings(m_list)
            results = sem.search(args.query, args.limit)
            for idx, result in enumerate(results):
                out = f"{idx + 1}. {result["title"]} (score: {result["score"]})\n\t{result["description"]}\n\n"
                out = textwrap.fill(out, initial_indent="", subsequent_indent="\t")
                print(out)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()