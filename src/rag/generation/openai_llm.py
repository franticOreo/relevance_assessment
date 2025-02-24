from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from .base_llm import BaseLLMWrapper
from typing import Any, Dict, Optional

class OpenAILLM(BaseLLMWrapper):
    """OpenAI ChatGPT implementation."""
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs: Any
    ):
        self.model_name = model_name
        # Convert max_tokens to max_length for consistency with base class
        super().__init__(temperature=temperature, max_length=max_tokens, **kwargs)

    def _initialize_llm(self) -> ChatOpenAI:
        """Initialize the OpenAI chat model."""
        load_dotenv()
        return ChatOpenAI(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_length,
            **self.kwargs
        )