import argparse
from internal.hybrid_search import HybridSearch, normalize_scores
from internal.dataset import load_movies

def main():
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparser = parser.add_subparsers(dest="command", help="hybrid search commands")

    norm_parser = subparser.add_parser("normalize", help="normalize a list of search scores")
    norm_parser.add_argument("scores", nargs="*", help="the list of scores to normalize")

    weighted_search_parser = subparser.add_parser("weighted-search", help="perform a weighted hybrid search")
    weighted_search_parser.add_argument("query", type=str, help="the search terms to evaluate")
    weighted_search_parser.add_argument("--alpha", dest="alpha", type=float, nargs="?", default=5, help="how much to bias the search towards either keyword or semantic search. 0 represents full semantic weight, and 1 represents full keyword weight")
    weighted_search_parser.add_argument("--limit", dest="limit", type=int, nargs="?", default=5, help="the maximum number of results to return")

    rrf_search_parser = subparser.add_parser("rrf-search", help="peform a search based on reciporical rank fusion")
    rrf_search_parser.add_argument("query", type=str, help="the search terms to evaluate")
    rrf_search_parser.add_argument("--k", dest="k", type=int, nargs="?", default=60, help="how much to bias towards a high rank. lower k values weigh rank more highly")
    rrf_search_parser.add_argument("--limit", dest="limit", type=int, default=5, help="the maximum number of results to return")
    args = parser.parse_args()
    
    match args.command:
        case "normalize":
            norm_list = normalize_scores(args.scores)
            for score in norm_list:
                print(f"* {score:.4f}")
        case "weighted-search":
            movies_docs = load_movies().get("movies")
            searcher = HybridSearch(movies_docs)
            results = searcher.weighted_search(args.query, args.alpha, args.limit)
            for idx, id in enumerate(results):
                result = results[id]
                print(f"\n{idx+1}. {result["document"]["title"]}")
                print(f"Hybrid Score: {result["hybrid_score"]:.3f}")
                print(f"BM25: {result["keyword_score"]:.3f}, Semantic: {result["semantic_score"]:.3f}")
                desc = result["document"]["description"]
                if len(desc) > 100:
                    desc = desc[:100] + "..."
                print(desc)
        case "rrf-search":
            movies_docs = load_movies().get("movies")
            searcher = HybridSearch(movies_docs)
            results = searcher.rrf_search(args.query, args.k, args.limit)
            for idx, id in enumerate(results):
                result = results[id]
                print(f"\n{idx+1}. {result["document"]["title"]}")
                print(f"RRF Score: {result["rrf_score"]:.3f}")
                print(f"BM25 Rank: {result.get("keyword_rank",-1):.3f}, Semantic: {result.get("semantic_rank",-1):.3f}")
                desc = result["document"]["description"]
                if len(desc) > 100:
                    desc = desc[:100] + "..."
                print(desc)
        case _:
            parser.print_help()




if __name__ == "__main__":
    main()