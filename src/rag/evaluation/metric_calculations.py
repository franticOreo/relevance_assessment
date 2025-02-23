from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
from collections import defaultdict
from langchain_core.documents import Document

@dataclass
class RetrievalMetrics:
    precision_at_k: float
    recall_at_k: float
    mrr: float  # Mean Reciprocal Rank
    latency: float

@dataclass
class GenerationMetrics:
    rouge_l: float
    bleu: float
    latency: float

@dataclass
class PipelineEvaluation:
    retrieval_metrics: RetrievalMetrics
    generation_metrics: GenerationMetrics
    judge_score: float
    total_latency: float


@dataclass
class JudgeMetrics:
    raw_score: float  # Original 1-4 Likert score
    normalized_score: float  # Normalized to 0-1 scale
    feedback: str

@dataclass
class PipelineEvaluation:
    retrieval_metrics: RetrievalMetrics
    generation_metrics: GenerationMetrics
    judge_metrics: JudgeMetrics
    total_latency: float

class MetricsCalculator:
    def __init__(self):
        self.rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    def _normalize_likert_score(self, score: float) -> float:
        """
        Normalize Likert scale (1-4) to 0-1 scale
        """
        return (score - 1) / 3  # Maps 1->0, 2->0.33, 3->0.66, 4->1
    
    def calculate_retrieval_metrics(
        self,
        retrieved_docs: List[Document],
        relevant_docs: List[str],
        retrieval_latency: float
    ) -> RetrievalMetrics:
        """
        Calculate retrieval-specific metrics.
        
        Args:
            retrieved_docs: List of retrieved documents
            relevant_docs: List of relevant document contents (ground truth)
            retrieval_latency: Time taken for retrieval
            
        Returns:
            RetrievalMetrics object with precision, recall, MRR and latency
        """
        # Convert retrieved docs to their content for comparison
        retrieved_contents = [doc.page_content for doc in retrieved_docs]
        
        # Calculate precision@k
        relevant_retrieved = sum(1 for doc in retrieved_contents if doc in relevant_docs)
        precision = relevant_retrieved / len(retrieved_docs) if retrieved_docs else 0
        
        # Calculate recall@k
        recall = relevant_retrieved / len(relevant_docs) if relevant_docs else 0
        
        # Calculate MRR (Mean Reciprocal Rank)
        mrr = 0
        for i, doc in enumerate(retrieved_contents):
            if doc in relevant_docs:
                mrr = 1 / (i + 1)
                break
                
        return RetrievalMetrics(
            precision_at_k=precision,
            recall_at_k=recall,
            mrr=mrr,
            latency=retrieval_latency
        )
    
    def calculate_generation_metrics(
        self,
        generated_answer: str,
        reference_answer: str,
        generation_latency: float
    ) -> GenerationMetrics:
        """
        Calculate generation-specific metrics.
        
        Args:
            generated_answer: Model generated answer
            reference_answer: Ground truth answer
            generation_latency: Time taken for generation
            
        Returns:
            GenerationMetrics object with ROUGE-L, BLEU and latency
        """
        # Calculate ROUGE-L score
        rouge_scores = self.rouge.score(reference_answer, generated_answer)
        rouge_l = rouge_scores['rougeL'].fmeasure
        
        # Calculate BLEU score
        reference_tokens = reference_answer.lower().split()
        candidate_tokens = generated_answer.lower().split()
        bleu = sentence_bleu([reference_tokens], candidate_tokens)
        
        return GenerationMetrics(
            rouge_l=rouge_l,
            bleu=bleu,
            latency=generation_latency
        )

    def evaluate_pipeline(
        self,
        pipeline: Any,
        test_dataset: List[Dict[str, Any]],
        judge_llm: Any
    ) -> Dict[str, Any]:
        """
        Evaluate a complete RAG pipeline using both traditional metrics and LLM judgment.
        """
        results = defaultdict(list)
        total_latency = 0
        
        for sample in test_dataset:
            # Get the query from either "question" or "user_input" key
            query = sample.get("question") or sample.get("user_input")
            if not query:
                raise ValueError("Test sample must contain either 'question' or 'user_input' key")
            
            # Measure retrieval performance
            retrieval_start = time.time()
            retrieved_docs = pipeline.retriever.get_relevant_documents(query)
            retrieval_latency = time.time() - retrieval_start
            
            # Measure generation performance
            generation_start = time.time()
            generated_answer = pipeline.run(query)
            generation_latency = time.time() - generation_start
            
            # Calculate metrics
            retrieval_metrics = self.calculate_retrieval_metrics(
                retrieved_docs=retrieved_docs,
                relevant_docs=sample["reference_contexts"],
                retrieval_latency=retrieval_latency
            )
            
            generation_metrics = self.calculate_generation_metrics(
                generated_answer=generated_answer,
                reference_answer=sample["reference"],
                generation_latency=generation_latency
            )
            
            # Get judge metrics
            judge_response = judge_llm.evaluate(
                question=query,
                candidate_answer=generated_answer,
                reference_answer=sample["reference"]
            )
            
            # Create normalized judge metrics
            judge_metrics = JudgeMetrics(
                raw_score=judge_response.score,
                normalized_score=self._normalize_likert_score(judge_response.score),
                feedback=judge_response.feedback
            )
            
            # Store results
            results["retrieval_metrics"].append(retrieval_metrics)
            results["generation_metrics"].append(generation_metrics)
            results["judge_metrics"].append(judge_metrics)
            
            total_latency += (retrieval_latency + generation_latency)
        
        # Calculate averages
        avg_results = {
            "avg_precision": np.mean([m.precision_at_k for m in results["retrieval_metrics"]]),
            "avg_recall": np.mean([m.recall_at_k for m in results["retrieval_metrics"]]),
            "avg_mrr": np.mean([m.mrr for m in results["retrieval_metrics"]]),
            "avg_rouge_l": np.mean([m.rouge_l for m in results["generation_metrics"]]),
            "avg_bleu": np.mean([m.bleu for m in results["generation_metrics"]]),
            "avg_judge_score_raw": np.mean([m.raw_score for m in results["judge_metrics"]]),
            "avg_judge_score_normalized": np.mean([m.normalized_score for m in results["judge_metrics"]]),
            "avg_latency": total_latency / len(test_dataset)
        }
        
        # Calculate composite score (example weighting)
        avg_results["composite_score"] = np.mean([
            avg_results["avg_precision"],
            avg_results["avg_recall"],
            avg_results["avg_rouge_l"],
            avg_results["avg_judge_score_normalized"]
        ])
        
        return {
            "detailed_results": results,
            "summary": avg_results
        }
