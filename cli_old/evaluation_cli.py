import argparse
from internal.dataset import load_golden, load_movies
from internal.hybrid_search import HybridSearch

def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    result = evaluate_command(args.limit)

    print(f"k={args.limit}\n")
    for query, res in result["results"].items():
        print(f"- Query: {query}")
        print(f"  - Precision@{args.limit}: {res['precision']:.4f}")
        print(f"  - Recall@{args.limit}: {res['recall']:.4f}")
        print(f"  - Retrieved: {', '.join(res['retrieved'])}")
        print(f"  - Relevant: {', '.join(res['relevant'])}")
        print()


def evaluate_command(limit: int = 5) -> dict:
    movies_list = load_movies().get("movies")
    golden_data = load_golden()
    test_cases = golden_data["test_cases"]

    hybrid_search = HybridSearch(movies_list)

    total_precision = 0
    results_by_query = {}
    for test_case in test_cases:
        query = test_case["query"]
        relevant_docs = set(test_case["relevant_docs"])
        search_results = hybrid_search.rrf_search(query, k=60, limit=limit)
        retrieved_docs = []
        for doc_id in search_results:
            result = search_results[doc_id]
            title = result.get("title", "")
            if title:
                retrieved_docs.append(title)

        precision = precision_at_k(retrieved_docs, relevant_docs, limit)
        recall = recall_at_k(retrieved_docs, relevant_docs, limit)

        results_by_query[query] = {
            "precision": precision,
            "recall": recall,
            "retrieved": retrieved_docs[:limit],
            "relevant": list(relevant_docs),
        }

        total_precision += precision

    return {
        "test_cases_count": len(test_cases),
        "limit": limit,
        "results": results_by_query,
    }
def precision_at_k(
    retrieved_docs: list[str], relevant_docs: set[str], k: int = 5
) -> float:
    top_k = retrieved_docs[:k]
    relevant_count = 0
    for doc in top_k:
        if doc in relevant_docs:
            relevant_count += 1
    return relevant_count / k


def recall_at_k(
    retrieved_docs: list[str], relevant_docs: set[str], k: int = 5
) -> float:
    top_k = retrieved_docs[:k]
    relevant_count = 0
    for doc in top_k:
        if doc in relevant_docs:
            relevant_count += 1
    return relevant_count / len(relevant_docs)


if __name__ == "__main__":
    main()