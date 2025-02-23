from langchain.schema import BaseRetriever
from typing import Any, List, Optional
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.embeddings.base import Embeddings
from vector_retriever import VectorRetriever
from typing import List, Dict, Tuple, Optional
from langchain.schema import BaseRetriever, Document
from collections import defaultdict

class HybridRetriever(BaseRetriever):
    """
    A hybrid retriever that combines scores from vector and BM25 retrievers.
    """
    
    def __init__(
        self,
        vector_retriever: VectorRetriever,
        bm25_retriever: BM25Retriever,
        alpha: float = 0.5,
        k: int = 5,
        fetch_k: Optional[int] = None
    ):
        """
        Initialize the hybrid retriever.

        Args:
            vector_retriever: Dense vector retriever component
            bm25_retriever: Sparse BM25 retriever component
            alpha: Weight for vector scores (1-alpha will be BM25 weight)
            k: Number of final results to return
            fetch_k: Number of results to fetch from each retriever (defaults to 2*k)
        """
        if not 0 <= alpha <= 1:
            raise ValueError("Alpha must be between 0 and 1")
            
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.alpha = alpha
        self.k = k
        self.fetch_k = fetch_k or 2 * k  # Fetch more docs than needed for better hybrid results
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Min-max normalize scores to [0, 1] range.
        """
        if not scores:
            return scores
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            return [1.0] * len(scores)
        return [(s - min_score) / (max_score - min_score) for s in scores]

    def _combine_scores(
        self,
        vector_results: List[Tuple[Document, float]],
        bm25_results: List[Tuple[Document, float]]
    ) -> List[Document]:
        """
        Combine and normalize scores from both retrievers.
        """
        # Create score maps for each retriever
        doc_scores: Dict[str, Dict] = defaultdict(
            lambda: {"vector_score": 0.0, "bm25_score": 0.0, "doc": None}
        )

        # Collect vector scores
        vector_scores = [score for _, score in vector_results]
        normalized_vector_scores = self._normalize_scores(vector_scores)
        for (doc, _), norm_score in zip(vector_results, normalized_vector_scores):
            doc_id = doc.metadata.get("doc_id", str(hash(doc.page_content)))
            doc_scores[doc_id]["vector_score"] = norm_score
            doc_scores[doc_id]["doc"] = doc

        # Collect BM25 scores
        bm25_scores = [score for _, score in bm25_results]
        normalized_bm25_scores = self._normalize_scores(bm25_scores)
        for (doc, _), norm_score in zip(bm25_results, normalized_bm25_scores):
            doc_id = doc.metadata.get("doc_id", str(hash(doc.page_content)))
            doc_scores[doc_id]["bm25_score"] = norm_score
            doc_scores[doc_id]["doc"] = doc

        # Combine scores and prepare final results
        final_results = []
        for doc_id, scores in doc_scores.items():
            doc = scores["doc"]
            if doc is None:
                continue
                
            # Calculate combined score
            combined_score = (
                self.alpha * scores["vector_score"] +
                (1 - self.alpha) * scores["bm25_score"]
            )
            
            # Update metadata with all scores for transparency
            doc.metadata.update({
                "vector_score": scores["vector_score"],
                "bm25_score": scores["bm25_score"],
                "combined_score": combined_score,
                "retrieval_method": "hybrid",
                "alpha": self.alpha
            })
            
            final_results.append((doc, combined_score))

        # Sort by combined score and return top k documents
        final_results.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in final_results[:self.k]]

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieve and combine results from both retrievers.
        """
        # Get results from both retrievers
        vector_results = self.vector_retriever.retrieve(query, k=self.fetch_k)
        bm25_results = self.bm25_retriever.retrieve(query, k=self.fetch_k)
        
        # Combine scores and return top k documents
        return self._combine_scores(vector_results, bm25_results)