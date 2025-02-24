from typing import List, Dict, Any, Optional
from rag.evaluation.metric_calculations import MetricsCalculator
from rag.retrieval.hybrid_retriever import HybridRetriever
from rag.retrieval.bm25_retriever import BM25Retriever
from rag.retrieval.vector_retriever import VectorRetriever, StandaloneVectorRetriever
from rag.generation.openai_llm import OpenAILLM
from rag.generation.huggingface_llm import HuggingFaceLLM
from rag.generation.agentic_rag import AgenticRAG
from rag.generation.standard_rag import StandardRAG
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rag.embeddings.openai_embeddings import get_openai_embeddings
from rag.evaluation.judge_llm import JudgeLLM
import os


class PipelineEvaluator:
    def __init__(
        self, 
        test_dataset: List[Dict[str, Any]], 
        documents: List[Document],
        api_key: Optional[str] = None
    ):
        """Initialize the evaluator with test dataset and documents.
        
        Args:
            test_dataset: List of test cases
            documents: List of documents to use for retrieval
            api_key: Optional API key for OpenAI services
        """
        self.test_dataset = test_dataset
        self.documents = documents
        self.api_key = api_key or os.getenv("api_key")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided either through constructor or environment variable")
        self.metrics_calculator = MetricsCalculator()
        
    def evaluate_pipeline_variant(
        self,
        retriever: Any,
        llm: Any,
        is_agentic: bool = False
    ) -> Dict[str, Any]:
        """Evaluate a specific pipeline variant."""
        if is_agentic:
            pipeline = AgenticRAG(
                retriever=retriever, 
                llm_model=llm,
                api_key=self.api_key
            )
        else:
            pipeline = StandardRAG(
                retriever=retriever, 
                llm=llm,
                api_key=self.api_key
            )
            
        # Create a JudgeLLM instance with the API key
        judge = JudgeLLM(
            model_name="gpt-4o",
            api_key=self.api_key
        )
            
        return self.metrics_calculator.evaluate_pipeline(
            pipeline=pipeline,
            test_dataset=self.test_dataset,
            judge_llm=judge
        )

    def run_all_evaluations(self) -> Dict[str, Any]:
        """Run evaluations for all pipeline variants."""
        results = {}
        
        # Initialize embeddings
        embedding_model = get_openai_embeddings()
        
        # Create FAISS vector store
        vector_store = FAISS.from_documents(self.documents, embedding_model)
        
        # Define retrievers
        retrievers = {
            "bm25": BM25Retriever(self.documents),
            "vector": StandaloneVectorRetriever(VectorRetriever(vector_store)),
            "hybrid": HybridRetriever(
                vector_retriever=VectorRetriever(vector_store),
                bm25_retriever=BM25Retriever(self.documents)
            )
        }
        
        # Define LLMs
        llms = {
            "gpt": OpenAILLM(),
            "llama": HuggingFaceLLM()
        }
        
        # Test combinations
        for retriever_name, retriever in retrievers.items():
            for llm_name, llm in llms.items():
                # Test standard RAG
                variant_name = f"{retriever_name}_{llm_name}_standard"
                results[variant_name] = self.evaluate_pipeline_variant(
                    retriever=retriever,
                    llm=llm,
                    is_agentic=False
                )
                
                # Test agentic RAG
                variant_name = f"{retriever_name}_{llm_name}_agentic"
                results[variant_name] = self.evaluate_pipeline_variant(
                    retriever=retriever,
                    llm=llm,
                    is_agentic=True
                )
        
        return results 