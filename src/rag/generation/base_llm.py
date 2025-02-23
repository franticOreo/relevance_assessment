from abc import abstractmethod
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, Runnable
from typing import Any, Optional, Dict, List, Union, Mapping, Sequence, Iterator, AsyncIterator

class BaseLLMWrapper(Runnable):
    """Abstract base class for LLM implementations."""
    
    default_template: str = "Question: {question}\nAnswer: Let's think step by step."
    
    def __init__(self, temperature: float = 0.7, max_length: int = 512, **kwargs: Any):
        self.temperature = temperature
        self.max_length = max_length
        self.kwargs = kwargs
        self._setup_chain()

    @abstractmethod
    def _initialize_llm(self) -> Any:
        """Initialize the specific LLM implementation."""
        pass

    def _setup_chain(self, custom_template: Optional[str] = None) -> None:
        """Set up the processing chain with optional custom template."""
        template = custom_template or self.default_template
        prompt = PromptTemplate(template=template, input_variables=["question"])
        self.chain = prompt | self._initialize_llm() | StrOutputParser()

    def set_custom_template(self, template: str) -> None:
        """Update the prompt template."""
        self._setup_chain(custom_template=template)

    def invoke(self, input: Union[Dict[str, Any], str]) -> str:
        """Required implementation for Runnable."""
        try:
            if isinstance(input, str):
                input = {"question": input}
            return self.chain.invoke(input)
        except Exception as e:
            raise RuntimeError(f"Error generating response: {str(e)}")

    async def ainvoke(
        self,
        input: Union[Dict[str, Any], str],
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        """Required implementation for Runnable."""
        try:
            if isinstance(input, str):
                input = {"question": input}
            return await self.chain.ainvoke(input, config=config, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Error generating response: {str(e)}")

    def batch(
        self,
        inputs: Sequence[Union[Dict[str, Any], str]],
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Batch processing implementation."""
        return [self.invoke(input, config=config, **kwargs) for input in inputs]

    async def abatch(
        self,
        inputs: Sequence[Union[Dict[str, Any], str]],
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Async batch processing implementation."""
        return [await self.ainvoke(input, config=config, **kwargs) for input in inputs]

    def stream(
        self,
        input: Union[Dict[str, Any], str],
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream implementation."""
        if isinstance(input, str):
            input = {"question": input}
        yield self.invoke(input, config=config, **kwargs)

    async def astream(
        self,
        input: Union[Dict[str, Any], str],
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Async stream implementation."""
        if isinstance(input, str):
            input = {"question": input}
        yield await self.ainvoke(input, config=config, **kwargs)

    def generate(self, question: str) -> str:
        """Generate a response for the given question."""
        return self.invoke({"question": question})

    def get_chain(self) -> RunnableSequence:
        """Get the underlying chain for direct integration."""
        return self.chain