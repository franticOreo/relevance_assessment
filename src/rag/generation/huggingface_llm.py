from langchain_community.llms import HuggingFaceHub
from dotenv import load_dotenv
import os
from .base_llm import BaseLLMWrapper
from typing import Any
class HuggingFaceLLM(BaseLLMWrapper):
    """Hugging Face LLM implementation."""
    
    def __init__(
        self,
        repo_id: str = "meta-llama/Llama-3.1-70B-Instruct",
        temperature: float = 0.7,
        max_length: int = 512,
        **kwargs: Any
    ):
        self.repo_id = repo_id
        self.model_name = repo_id
        load_dotenv()
        super().__init__(temperature=temperature, max_length=max_length, **kwargs)

    def _initialize_llm(self) -> HuggingFaceHub:
        """Initialize the Hugging Face model."""
        return HuggingFaceHub(
            repo_id=self.repo_id,
            huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY"),
            model_kwargs={
                "temperature": self.temperature,
                "max_length": self.max_length,
                **self.kwargs
            }
        )