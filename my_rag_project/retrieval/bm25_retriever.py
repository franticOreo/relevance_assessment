from rank_bm25 import BM25Okapi
from langchain.schema import BaseRetriever

import re
from typing import Any

class BM25Retriever:
    def __init__(self, documents):
        """
        Initializes the BM25 retriever.
        
        Parameters:
            documents (list): A list of Document objects. Each document should have a 'page_content' attribute.
        """
        self.documents = documents
        # Pre-tokenize each document’s content for BM25 indexing.
        self.tokenized_corpus = [self._tokenize(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
    def _tokenize(self, text):
        """
        A simple tokenizer that lowercases text and extracts word tokens.
        """
        tokens = re.findall(r"\w+", text.lower())
        return tokens
    
    def get_relevant_documents(self, query: str, k: int = 5):
        """
        Returns a list of Document objects relevant to the query.
        The BM25 scores are attached to the document metadata if needed.
        """
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        # Get indices of top k scores.
        top_n_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        
        docs = []
        for idx in top_n_indices:
            doc = self.documents[idx]  # This is an instance of Document.
            # Optionally, store the BM25 score in the document metadata.
            doc.metadata["score"] = scores[idx]
            docs.append(doc)
        return docs

