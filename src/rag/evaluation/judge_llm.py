from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
import os

class JudgeResponse(BaseModel):
    """Schema for the judge's response"""
    model_config = ConfigDict(frozen=True)
    
    feedback: str = Field(
        description="Detailed feedback on the candidate answer quality"
    )
    score: int = Field(
        description="Evaluation score on a scale of 1-4",
        ge=1,  # greater than or equal to 1
        le=4   # less than or equal to 4
    )

class JudgeLLM:
    """LLM-based judge for evaluating answer quality"""
    
    def __init__(
        self,
        model: Optional[BaseChatModel] = None,
        model_name: str = "gpt-4",
        temperature: float = 0.0
    ):
        """
        Initialize the judge LLM.
        
        Args:
            model: Optional pre-configured LLM model
            model_name: Name of the OpenAI model to use if no model provided
            temperature: Sampling temperature for the LLM
        """
        self.judge_llm = model or ChatOpenAI(
            model_name=model_name,  # Fixed: Changed model to model_name
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        ).with_structured_output(JudgeResponse)

    def evaluate(
        self,
        question: str,
        candidate_answer: str,
        reference_answer: str
    ) -> JudgeResponse:
        """
        Evaluates a single answer against a reference.
        
        Args:
            question: The question being answered
            candidate_answer: The generated answer to evaluate
            reference_answer: The ground truth answer
            
        Returns:
            JudgeResponse with score and feedback
        """
        judge_prompt = f"""
        You are an experienced evaluator. Please score the following candidate answer on a Likert scale from 1 to 4, where:
            1 = Poor: Incorrect, irrelevant, or incomplete.
            2 = Fair: Moderately accurate but missing some details.
            3 = Good: Mostly accurate and relevant.
            4 = Excellent: Completely accurate and comprehensive.

        Question: {question}

        Candidate Answer: {candidate_answer}

        Reference Answer: {reference_answer}

        Provide feedback to the candidate answer given the reference answer. Then provide the score.
        """
        
        return self.judge_llm.invoke(judge_prompt)

    def evaluate_dataset(
        self,
        dataset: List[Any],
        qa_pipeline: Any
    ) -> Dict[str, Any]:
        """
        Evaluates your retriever+LLM pipeline on each sample in the dataset.
        """
        scores = []
        detailed_results = []
        
        for _, sample in dataset.to_pandas().iterrows():
            user_input = sample["user_input"]
            reference = sample["reference"]
            
            candidate_answer = qa_pipeline.run(user_input)
            
            judge_response = self.evaluate(
                question=user_input,
                candidate_answer=candidate_answer,
                reference_answer=reference
            )
            
            scores.append(judge_response.score)
            detailed_results.append({
                "question": user_input,
                "candidate_answer": candidate_answer,
                "reference_answer": reference,
                "judge_response": judge_response,
                "score": judge_response.score
            })
        
        average_score = sum(scores) / len(scores) if scores else 0
        return {
            "scores": scores,
            "average_score": average_score,
            "detailed_results": detailed_results
        }