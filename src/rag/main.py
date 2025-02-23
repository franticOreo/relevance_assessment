import logging
import warnings
import os

# Suppress all warnings
warnings.filterwarnings("ignore")

# Suppress specific loggers
logging.getLogger("pikepdf").setLevel(logging.ERROR)
logging.getLogger("faiss").setLevel(logging.ERROR)
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

# Suppress GPU FAISS warning by setting environment variable
os.environ["FAISS_NO_GPU"] = "1"

from embeddings.openai_embeddings import get_openai_embeddings
from retrieval.hybrid_retriever import HybridRetriever
from generation.agentic_rag import AgenticRAG
from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from generation.openai_llm import OpenAILLM
from generation.huggingface_llm import HuggingFaceLLM
from loader.document_loader import get_default_documents
import argparse
from langchain_community.vectorstores import FAISS
from retrieval.vector_retriever import VectorRetriever
from retrieval.bm25_retriever import BM25Retriever

def setup_rag():
    """Initialize and return the RAG system components."""
    # Load environment variables
    load_dotenv()
    
    # Load documents
    documents = get_default_documents()
    
    # Setup embedding model and create vector store
    embedding_model = get_openai_embeddings()
    vector_store = FAISS.from_documents(documents, embedding_model)
    
    # Create retrievers
    vector_retriever = VectorRetriever(vector_store)
    bm25_retriever = BM25Retriever(documents)
    
    # Create hybrid retriever
    hybrid_retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        alpha=0.5
    )
    
    # Initialize the RAG pipeline
    rag = AgenticRAG(
        retriever=hybrid_retriever,
        llm_model="gpt-3.5-turbo"
    )
    
    return rag

def interactive_mode(rag):
    """Run the RAG system in interactive mode."""
    print("\nWelcome to the Insurance RAG System!")
    print("Type 'quit' or 'exit' to end the session.")
    
    while True:
        # Get user input
        query = input("\nEnter your question: ").strip()
        
        # Check for exit command
        if query.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
        
        # Process query
        if query:
            try:
                answer = rag.run_query(query)
                print("\nAnswer:", answer)
            except Exception as e:
                print(f"\nError: {str(e)}")
        else:
            print("Please enter a valid question.")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Insurance RAG System')
    parser.add_argument('--query', '-q', type=str, help='Single query mode: Ask one question and exit')
    
    args = parser.parse_args()
    
    try:
        # Initialize RAG system
        rag = setup_rag()
        
        if args.query:
            # Single query mode
            answer = rag.run_query(args.query)
            print(f"\nQuery: {args.query}")
            print(f"Answer: {answer}")
        else:
            # Interactive mode
            interactive_mode(rag)
            
    except Exception as e:
        print(f"Error initializing RAG system: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())