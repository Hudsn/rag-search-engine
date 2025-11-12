import argparse

from lib.hybrid_search import (
    HybridSearch
)
from lib.search_utils import (
    load_movies,
    RRF_K
)
from lib.augmented_generation import (
    llm_augmented_gen,
    llm_summary_gen,
    llm_citation_gen,
    llm_quesstion_answering_gen
)

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser("summarize", help="generate a summary of the results")
    summarize_parser.add_argument("query", type=str, help="Search query for summarization")
    summarize_parser.add_argument("--limit", type=int, nargs="?", default=5, help="number of initial search results to summarize")

    citation_parser = subparsers.add_parser("citations", help="generate an answer with citations")
    citation_parser.add_argument("query", type=str, help="Search query for summarization")
    citation_parser.add_argument("--limit", type=int, nargs="?", default=5, help="number of initial search results to summarize")
    
    question_parser = subparsers.add_parser("question", help="generate an answer with citations")
    question_parser.add_argument("question", type=str, help="Search query for summarization")
    question_parser.add_argument("--limit", type=int, nargs="?", default=5, help="number of initial search results to summarize")


    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            doc_list = run_hybrid_search(query, RRF_K, 5)
            for doc in doc_list:
                print(f"- {doc.get("title")}")
            print()
            print("RAG Response:")
            print(llm_augmented_gen(query, doc_list))
        case "summarize":
            query = args.query
            limit = args.limit
            doc_list = run_hybrid_search(query, RRF_K, limit)
            for doc in doc_list:
                print(f"- {doc.get("title")}")
            print()
            print("LLM Summary:")
            print(llm_summary_gen(query, doc_list))
        case "citations":
            query = args.query
            limit = args.limit
            doc_list = run_hybrid_search(query, RRF_K, limit)
            for doc in doc_list:
                print(f"- {doc.get("title")}")
            print()
            print("LLM Answer:")
            print(llm_citation_gen(query, doc_list))
        case "question":
            question = args.question
            limit = args.limit
            doc_list = run_hybrid_search(question, RRF_K, limit)
            for doc in doc_list:
                print(f"- {doc.get("title")}")
            print()
            print("Answer:")
            print(llm_quesstion_answering_gen(question, doc_list))
        case _:
            parser.print_help()

def run_hybrid_search(query: str, k: int=RRF_K, limit: int=5) -> list[dict]:
    movies = load_movies()
    searcher = HybridSearch(movies)
    return searcher.rrf_search(query, k ,limit=limit)

if __name__ == "__main__":
    main()