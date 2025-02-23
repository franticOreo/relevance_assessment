from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from langchain_community.chat_models import ChatOpenAI, init_chat_model
import os

class JudgeResponse(BaseModel):
    feedback: str = Field(description="Feedback on the candidate answer")
    score: int = Field(description="Score on a scale of 1-4")

class JudgeLLM:
    def __init__(self, model_name: str = "gpt-4o-mini", model_provider: str = "openai", temperature: float = 0):
        self.judge_llm = init_chat_model(
            model_name,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model_provider=model_provider
        ).with_structured_output(JudgeResponse)

    def evaluate(
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
            
            judge_prompt = f"""
            You are an experienced evaluator. Please score the following candidate answer on a Likert scale from 1 to 4, where:
                1 = Poor: Incorrect, irrelevant, or incomplete.
                2 = Fair: Moderately accurate but missing some details.
                3 = Good: Mostly accurate and relevant.
                4 = Excellent: Completely accurate and comprehensive.

            Question: {user_input}

            Candidate Answer: {candidate_answer}

            Reference Answer: {reference}

            Provide feedback to the candidate answer given the reference answer. Then provide the score.
            """
            
            judge_response = self.judge_llm.invoke(judge_prompt)

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