import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
model = "gemini-2.0-flash"
client = genai.Client(api_key=api_key)

def llm_generate_parts(proompt: str, img_bytes: bytes, mime: str, query: str=None) -> dict:
    sys_prompt = proompt
    content_parts = [
        sys_prompt,
        genai.types.Part.from_bytes(data=img_bytes, mime_type=mime)
    ]
    if query is not None:
        content_parts.append(query.strip())


    resp = client.models.generate_content(
        model=model, 
        contents=content_parts
    )
    ret = {
        "response": (resp.text or "").strip(),
    }
    if resp.usage_metadata is not None:
        ret["token_count"] = resp.usage_metadata.total_token_count
    return ret


def llm_describe_image(img_bytes: bytes, mime:str, query: str) -> dict:
    prompt = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary"""
    return llm_generate_parts(prompt, img_bytes, mime, query)