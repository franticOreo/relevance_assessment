from embeddings.openai_embeddings import get_openai_embeddings
from retrieval.hybrid_retriever import HybridRetriever
from generation.agentic_rag import AgenticRAG
import os
from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader


def main():
    # Load environment variables
    load_dotenv()
    
    # Configure the loader with parameters to handle different elements
    loader = UnstructuredLoader(
        "./data/docs/nrma.pdf",
        chunking_strategy="by_title"
    )

    # Load the document
    documents = loader.load()
    
    # Setup embedding model and retriever
    embedding_model = get_openai_embeddings()

    hybrid_retriever = HybridRetriever.from_documents(
        documents,
        embedding_model,
        alpha=0.5
    )

    # Initialize the RAG pipeline
    rag = AgenticRAG(
        retriever=hybrid_retriever,
        llm_model="gpt-3.5-turbo"
    )
    
    # Example query
    query = "What is the maximum cover for a single vehicle?"
    answer = rag.run_query(query)
    print(f"Query: {query}\nAnswer: {answer}")

if __name__ == "__main__":
    main()