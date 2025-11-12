#!/usr/bin/python

import argparse, textwrap
from internal.semantic_search import SemanticSearch, ChunkedSemanticSearch, verify_model, embed_text, verify_embeddings, embed_query_text, chunk_text, chunk_text_semantic
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
    search_parser.add_argument("--limit", dest="limit", type=int, nargs="?", default=5)

    # chunk
    chunk_parser = subparser.add_parser("chunk", help="Chunk the given text to an appropriate length")
    chunk_parser.add_argument("text", type=str, help="text to chunk")
    chunk_parser.add_argument("--chunk-size", dest="chunk_size", nargs="?", type=int, default=200, help="the desired size of the text chunks, in words")
    chunk_parser.add_argument("--overlap", dest="overlap", type=int, nargs="?", default=0, help="how many words should be overlapping between chunks")
    

    semantic_chunk_parser = subparser.add_parser("semantic_chunk", help="chunk the target text based on semantic chunking")
    semantic_chunk_parser.add_argument("text", type=str, help="The text to chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", dest="max_chunk_size", type=int, nargs="?", default=4, help="the desired size of the text chunks, in sentences")
    semantic_chunk_parser.add_argument("--overlap", dest="overlap", nargs="?", type=int, default=0, help="How many sentences should be overlapping between chunks")

    subparser.add_parser("embed_chunks", help="generate and load semantic embeddings for chunks of text")
    
    search_chunked_parser = subparser.add_parser("search_chunked", help="search for the term using semantic search with chunking")
    search_chunked_parser.add_argument("query", type=str, help="the query to run in search")
    search_chunked_parser.add_argument("--limit", dest="limit", nargs="?", type=int, default=5, help="The limit of number of results to return")


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
        case "chunk":
            print(f"Chunking {len(args.text)} characters")
            chunks = chunk_text(args.text, args.chunk_size, args.overlap)
            for idx, chunk in enumerate(chunks):
                print(f"{idx + 1}. {chunk}")
        case "semantic_chunk":
            print(f"Semantically chunking {len(args.text)} characters")
            chunks = chunk_text_semantic(args.text, args.max_chunk_size, args.overlap)
            for idx, chunk in enumerate(chunks):
                print(f"{idx + 1}. {chunk}")
        case "embed_chunks":
            m_list = load_movies().get("movies")
            if m_list is None:
                raise Exception("movie list not found")
            chunker = ChunkedSemanticSearch()
            embeddings = chunker.load_or_create_chunk_embeddings(m_list)
            print(f"Generated {len(embeddings)} chunked embeddings")
        case "search_chunked":
            m_list = load_movies().get("movies")
            if m_list is None:
                raise Exception("movie list not found")
            chunker = ChunkedSemanticSearch()
            embeddings = chunker.load_or_create_chunk_embeddings(m_list)
            results = chunker.search_chunks(args.query, args.limit)
            for idx, result in enumerate(results):
                print(f"\n{idx+1}. {result.get("title")} (score: {result.get("score"):.4f})")
                print(f"    {result.get("document")}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()