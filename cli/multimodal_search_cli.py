import argparse
from lib.multimodal_search import verify_image_embedding, search_image
from lib.search_utils import load_movies

def main():
        
    parser = argparse.ArgumentParser(description="CLI for multimodal search")
    subparsers = parser.add_subparsers(dest="command", help="available commands for multimodal search")

    verify_img_parser = subparsers.add_parser("verify_image_embedding")
    verify_img_parser.add_argument("img_path", type=str, help="the path to the image to verify embeddings")

    img_search_parser = subparsers.add_parser("image_search")
    img_search_parser.add_argument("img_path", type=str, help="the path to the image to use as a search query")

    args = parser.parse_args()

    match  args.command:
        case "verify_image_embedding":
            path = args.img_path
            verify_image_embedding(path)
        case "image_search":
            path = args.img_path
            movies = load_movies()
            results = search_image(path, movies)
            for i, result in enumerate(results):
                print(f"{i + 1}. {result["title"]} (similarity: {result["score"]:.3f})")
                print(f"\t{result["description"][:150]}...")
                print()
        case _:
            parser.print_help()

if __name__=="__main__":
    main()