import os
import mimetypes
import argparse

from lib.describe_image import (
    llm_describe_image
)

def main():

    parser = argparse.ArgumentParser(description="Image CLI")
    parser.add_argument("--image", type=str, help="path to the image file")
    parser.add_argument("--query", type=str, help="query to run against the image")

    args = parser.parse_args()
    query = args.query
    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    img_bytes = None
    with open(args.image, "rb") as f:
        img_bytes = f.read()
    rewritten_resp_dict = llm_describe_image(img_bytes, mime, query)
    tok_count = rewritten_resp_dict["token_count"]
    rewritten_query = rewritten_resp_dict["response"]

    print(f"Rewritten query: {rewritten_query.strip()}")
    if tok_count is not None:
        print(f"Total tokens: {tok_count}")

if __name__ == "__main__":
    main()