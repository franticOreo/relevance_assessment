from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import os

def get_openai_embeddings():
    """Returns an OpenAI embeddings instance."""
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    print(openai_api_key)
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set in the environment.")
    return OpenAIEmbeddings(api_key=openai_api_key)