from typing import List, Dict, Any
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


class PipelineEvaluator:
    def __init__(self, test_dataset: List[Dict[str, Any]], documents: List[Document]):
        self.test_dataset = test_dataset
        self.documents = documents
        self.metrics_calculator = MetricsCalculator()
        
    def evaluate_pipeline_variant(
        self,
        retriever: Any,
        llm: Any,
        is_agentic: bool = False
    ) -> Dict[str, Any]:
        """Evaluate a specific pipeline variant."""
        if is_agentic:
            pipeline = AgenticRAG(retriever=retriever, llm_model=llm)
        else:
            # Assuming you'll create a standard RAG pipeline class
            pipeline = StandardRAG(retriever=retriever, llm=llm)
            
        # Create a JudgeLLM instance instead of using OpenAILLM directly
        judge = JudgeLLM(model_name="gpt-3.5-turbo")
            
        return self.metrics_calculator.evaluate_pipeline(
            pipeline=pipeline,
            test_dataset=self.test_dataset,
            judge_llm=judge  # Use the JudgeLLM instance
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