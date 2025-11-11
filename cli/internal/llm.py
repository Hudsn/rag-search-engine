import os
from dotenv import load_dotenv
from google import genai

def make_client() -> genai.Client:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    print(f"Using key {api_key[:6]}...")

    return genai.Client(api_key=api_key)

def gen_content(client: genai.Client, contents: str) -> str:
    resp = client.models.generate_content(model="gemini-2.0-flash-001", contents=contents)
    return resp.text
