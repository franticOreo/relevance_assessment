from typing import List, Tuple
from langchain.schema import BaseRetriever
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS

class VectorRetriever:
    """
    A dense (vector-based) retriever component that can be used standalone or as part 
    of a hybrid retrieval system.
    """

    def __init__(self, vector_store: FAISS):
        """
        Initialize the vector retriever with an existing FAISS store.

        Args:
            vector_store (FAISS): A FAISS vector store that has already been built from documents.
        """
        self.vector_store = vector_store
    
    def retrieve(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """
        Retrieve documents and their scores from the vector store.

        Args:
            query (str): The user's query
            k (int): Number of top documents to retrieve

        Returns:
            List[Tuple[Document, float]]: List of (Document, score) pairs
        """
        # Get documents and scores
        results = self.vector_store.similarity_search_with_score(query, k=k)
        
        # Enhance metadata
        enhanced_results = []
        for doc, score in results:
            doc.metadata["vector_score"] = score  # Use specific name to avoid confusion
            doc.metadata["retrieval_method"] = "dense_vector"
            enhanced_results.append((doc, score))
            
        return enhanced_results

class StandaloneVectorRetriever(BaseRetriever):
    """
    A wrapper that turns the VectorRetriever into a standalone BaseRetriever
    (in case you want to use it directly without hybrid retrieval).
    """
    
    def __init__(self, vector_retriever: VectorRetriever):
        """Initialize with a VectorRetriever instance."""
        super().__init__()  # Add this line to properly initialize the parent class
        self._vector_retriever = vector_retriever  # Change to protected attribute
    
    def _get_relevant_documents(self, query: str, k: int = 5) -> List[Document]:
        """
        Implements the BaseRetriever interface for standalone use.
        """
        results = self._vector_retriever.retrieve(query, k=k)
        return [doc for doc, _ in results]  # Strip scores for standard interface