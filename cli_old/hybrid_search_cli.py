import argparse, time, json

from sentence_transformers import CrossEncoder

from internal.hybrid_search import HybridSearch, normalize_scores
from internal.dataset import load_movies
from internal.llm import make_client, gen_content

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
    rrf_search_parser.add_argument("--enhance", dest="enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enhancement method")
    rrf_search_parser.add_argument("--rerank-method", dest="rerank_method", type=str, choices=["individual", "batch", "cross_encoder"], help="Query rerank method")
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
            query_text = handle_enhance(args.query, args.enhance)
            limit = maybe_increase_limit(args.rerank_method, args.limit)
            results = searcher.rrf_search(query_text, args.k, limit)
            results = handle_rerank(query_text, results, args.rerank_method)
            for idx, id in enumerate(results):
                result = results[id]
                print(f"\n{idx+1}. {result["title"]}")
                if "cross_encode_score" in result:
                    print(f"Cross Encoder Score: {result["cross_encode_score"]:.3f}")
                print(f"RRF Score: {result["rrf_score"]:.3f}")
                print(f"BM25 Rank: {result.get("keyword_rank",-1):.3f}, Semantic: {result.get("semantic_rank",-1):.3f}")
                desc = result["document"]
                if len(desc) > 100:
                    desc = desc[:100] + "..."
                print(desc)
        case _:
            parser.print_help()

def maybe_increase_limit(rerank_arg: str, limit: int) -> int:
    if rerank_arg is None:
        return limit
    match rerank_arg:
        case "individual":
            return limit * 5
        case "batch":
            return limit * 5
        case "cross_encoder":
            return limit * 5
        case _:
            return limit
        
def handle_rerank(query: str, docs: dict, rerank_arg: str, limit: int) -> dict:
    if rerank_arg is None:
        return docs
    ret: dict = {}
    match rerank_arg:
        case "individual":
            ret = handle_rerank_individual_multi(query, docs)
        case "batch":
            ret = handle_rerank_batch(query, docs)
        case "cross_encoder":
            ret = handle_rerank_cross_encode(query, docs)
        case _:
            return docs
    if len(ret) > limit:
        return dict(list(ret.items())[:limit])

def handle_rerank_cross_encode(query: str, docs: dict) -> dict:
    pairs: list[list[str, str]] = []
    for doc_id in docs:
        doc = docs[doc_id]
        pairs.append([query, f"{doc.get("title", "")} - {doc.get("document", "")}"])
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
    scores = cross_encoder.predict(pairs)
    
    for idx, doc_id in enumerate(docs):
        docs[doc_id]["cross_encode_score"] = scores[idx].item()
    return dict(sorted(docs.items(), key=lambda entry: entry[1]["cross_encode_score"], reverse=True))

def handle_rerank_batch(query: str, docs: dict) -> dict:
    doc_list_str = json.dumps([docs])
    proompt = f"""Rank these movies by relevance to the search query.

Query: "{query}"

Movies:
{doc_list_str}

Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

[75, 12, 34, 2, 1]
"""
    c = make_client()
    ids_str = gen_content(c, proompt)
    ids = list(json.loads(ids_str))
    ret = {}
    for id in ids:
        ret[id] = docs[id]
    return ret

def handle_rerank_individual_multi(query: str, docs: dict) -> dict:
    for doc_id in docs:
        doc = docs[doc_id]
        docs[doc_id]["rerank_score"] = handle_rerank_individual(query, doc)
        time.sleep(3)
    return dict(sorted(docs.items(), key=lambda entry: entry[1]["rerank_score"], reverse=True))



def handle_rerank_individual(query: str, doc: dict) -> int:
    proooooompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("document", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Give me ONLY the number in your response, no other text or explanation.

Score:"""
    c = make_client()
    try:
        ret = int(gen_content(c, proooooompt))
    except:
        ret = 0
    return ret

def handle_enhance(query: str, enhance_type: str) -> str:
    if enhance_type is None:
        return query
    enhanced = None
    match enhance_type:
        case "spell":
            enhanced = enhance_query_spell(query)
        case "rewrite":
            enhanced = enhance_query_rewrite(query)
        case "expand":
            enhanced = enhance_query_expand(query)
        case _:
            print(f"unknown enhancement type; using original query...")
            return query
    print(f"Enhanced query ({enhance_type}): '{query}' -> '{enhanced}'\n")
    return enhanced

def enhance_query_expand(query: str) -> str:
    proooooompt = f"""Expand this movie search query with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
This will be appended to the original query.

Examples:

- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

Query: "{query}"
"""
    c = make_client()
    return gen_content(c, proooooompt)
    

def enhance_query_rewrite(query: str) -> str:
    prooooooooooooompt = f"""Rewrite this movie search query to be more specific and searchable.

Original: "{query}"

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep it concise (under 10 words)
- It should be a google style search query that's very specific
- Don't use boolean logic

Examples:

- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

Rewritten query:"""
    c = make_client()
    return gen_content(c, prooooooooooooompt)

def enhance_query_spell(query: str) -> str:
    prooompt = f"""Fix any spelling errors in this movie search query.

Only correct obvious typos. Don't change correctly spelled words.

Query: "{query}"

If no errors, return the original query.
Corrected:"""
    c = make_client()
    return gen_content(c, prooompt)
    



if __name__ == "__main__":
    main()


