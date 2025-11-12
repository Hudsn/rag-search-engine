import json
import os
from dotenv import load_dotenv
from google import genai

api_key = os.getenv("GEMINI_API_KEY")
model = "gemini-2.0-flash"
client = genai.Client(api_key=api_key)
def llm_generate(proompt: str) -> str:
    resp = client.models.generate_content(model=model, contents=proompt)
    return (resp.text or "").strip()

def llm_augmented_gen(query: str, docs: list[dict]) -> str:
    docs_prompt_lines = []
    for doc in docs:
        formatted_for_llm = f"Title: {doc["title"]}  --  Description: {doc["document"]}"
        docs_prompt_lines.append(formatted_for_llm)
    prompt_docs_section = "\n".join(docs_prompt_lines)
    prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Documents:
{prompt_docs_section}

Provide a comprehensive answer that addresses the query:"""
    return llm_generate(prompt)
    

    
def llm_summary_gen(query: str, docs: list[dict]) -> str:
    docs_prompt_lines = []
    for doc in docs:
        formatted_for_llm = f"Title: {doc["title"]}  --  Description: {doc["document"]}"
        docs_prompt_lines.append(formatted_for_llm)
    prompt_docs_section = "\n".join(docs_prompt_lines)
    prompt = f"""
Provide information useful to this query by synthesizing information from multiple search results in detail.
The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
This should be tailored to Hoopla users. Hoopla is a movie streaming service.
Query: {query}
Search Results:
{prompt_docs_section}

Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
"""
    return llm_generate(prompt)
    

def llm_citation_gen(query: str, docs: list[dict]) -> str:
    docs_prompt_lines = []
    for doc in docs:
        formatted_for_llm = f"Title: {doc["title"]}  --  Description: {doc["document"]}"
        docs_prompt_lines.append(formatted_for_llm)
    prompt_docs_section = "\n".join(docs_prompt_lines)
    prompt = f"""Answer the question or provide information based on the provided documents.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

Query: {query}

Documents:
{prompt_docs_section}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources using [1], [2], etc. format when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the documents, say "I don't have enough information"
- Be direct and informative

Answer:"""
    return llm_generate(prompt)


def llm_quesstion_answering_gen(query: str, docs: list[dict]) -> str:
    docs_prompt_lines = []
    for doc in docs:
        formatted_for_llm = f"Title: {doc["title"]}  --  Description: {doc["document"]}"
        docs_prompt_lines.append(formatted_for_llm)
    prompt_docs_section = "\n".join(docs_prompt_lines)
    prompt = f"""Answer the following question based on the provided documents.

Question: {query}

Documents:
{prompt_docs_section}

General instructions:
- Answer directly and concisely
- Use only information from the documents
- If the answer isn't in the documents, say "I don't have enough information"
- Cite sources when possible

Guidance on types of questions:
- Factual questions: Provide a direct answer
- Analytical questions: Compare and contrast information from the documents
- Opinion-based questions: Acknowledge subjectivity and provide a balanced view

Answer:"""
    return llm_generate(prompt)

