from rank_bm25 import BM25Okapi
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List, Tuple, Any, Sequence
import re
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough

class BM25Retriever(BaseRetriever, BaseModel):
    """BM25 retriever implementation."""
    
    documents: List[Document] = Field(description="List of documents to search through")
    tokenized_corpus: List[List[str]] = Field(default_factory=list, description="Pre-tokenized document contents")
    bm25: Any = Field(default=None, description="BM25 index")
    k: int = Field(default=4, description="Number of documents to retrieve")

    def __init__(self, documents: List[Document], k: int = 4):
        """Initialize the BM25 retriever."""
        super().__init__(
            documents=documents,
            tokenized_corpus=[self._tokenize(doc.page_content) for doc in documents],
            bm25=BM25Okapi([self._tokenize(doc.page_content) for doc in documents]),
            k=k
        )
    
    def create_retrieval_chain(self):
        """
        Create a retrieval chain using LCEL syntax.
        Returns a chain that can be used in LCEL pipelines.
        """
        return RunnablePassthrough.assign(
            context=lambda x: self.get_relevant_documents(x["question"])
        )

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: Any = None
    ) -> List[Document]:
        """Asynchronously get documents relevant to a query."""
        return self.get_relevant_documents(query)

    def get_relevant_documents(
        self, query: str, *, run_manager: Any = None
    ) -> List[Document]:
        """
        Returns a list of Document objects relevant to the query.
        The BM25 scores are attached to the document metadata.
        """
        # Add type checking
        if not isinstance(query, str):
            print(query)
            raise TypeError(f"Expected string query, got {type(query)}: {query}")
        
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        
        # Get indices of top k scores (default k=4)
        k = self.k  # Use the instance's k attribute
        top_n_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        
        docs = []
        for idx in top_n_indices:
            doc = self.documents[idx].copy()  # Create a copy to avoid modifying original
            # Store the BM25 score in the document metadata
            doc.metadata["score"] = float(scores[idx])  # Convert to float for serialization
            doc.metadata["retrieval_method"] = "bm25"
            docs.append(doc)
            
        return docs
    
    def retrieve_with_scores(
        self, query: str, k: int = 4
    ) -> List[Tuple[Document, float]]:
        """
        Alternative method that returns documents with their scores for hybrid retrieval.
        """
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        
        # Get indices of top k scores
        top_n_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        
        docs = []
        for idx in top_n_indices:
            doc = self.documents[idx].copy()  # Create a copy to avoid modifying original
            score = float(scores[idx])  # Convert to float for serialization
            # Store the BM25 score in the document metadata
            doc.metadata["score"] = score
            doc.metadata["retrieval_method"] = "bm25"
            docs.append((doc, score))
            
        return docs
    
    def _tokenize(self, text: str) -> List[str]:
        """
        A simple tokenizer that lowercases text and extracts word tokens.
        """
        return re.findall(r"\w+", text.lower())

