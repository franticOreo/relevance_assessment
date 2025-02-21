from dotenv import load_dotenv
from langchain.embeddings import CohereEmbeddings
import os

def get_cohere_embeddings():
    """Returns a Cohere embeddings instance."""
    load_dotenv()
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if not cohere_api_key:
        raise ValueError("COHERE_API_KEY is not set in the environment.")
    return CohereEmbeddings(cohere_api_key=cohere_api_key)