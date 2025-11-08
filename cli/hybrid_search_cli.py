import argparse


def main():
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparser = parser.add_subparsers(dest="command", help="hybrid search commands")

    norm_parser = subparser.add_parser("normalize", help="normalize a list of search scores")
    norm_parser.add_argument("scores", nargs="*", help="the list of scores to normalize")

    args = parser.parse_args()
    
    match args.command:
        case "normalize":
            norm_list = normalize_scores(args.scores)
            for score in norm_list:
                print(f"* {score:.4f}")
        case _:
            parser.print_help()


def normalize_scores(scores: list) -> list:
    if len(scores) == 0:
        return []
    low = float(min(scores))
    high = float(max(scores))
    if high == low:
        return list(map(lambda _: 1, scores))
    norm_list = []
    for score in scores:
        score = float(score)
        normalized = (score - low) / (high - low)
        norm_list.append(normalized)
    return norm_list

if __name__ == "__main__":
    main()