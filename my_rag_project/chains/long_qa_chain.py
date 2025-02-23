from typing import Optional, List, Sequence
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.runnables import RunnableSequence, RunnableParallel
from langchain_community.chat_models import ChatOpenAI

class LongQAChain:
    """A question-answering chain that uses map-reduce to handle long contexts efficiently."""
    
    def __init__(
        self,
        llm: Optional[BaseLanguageModel] = None,
        retriever: Optional[BaseRetriever] = None,
        map_prompt: Optional[PromptTemplate] = None,
        reduce_prompt: Optional[PromptTemplate] = None
    ):
        """Initialize the QA chain."""
        self.llm = llm or ChatOpenAI(model_name="gpt-3.5-turbo")
        self.retriever = retriever
        self.map_prompt = map_prompt or self._default_map_prompt()
        self.reduce_prompt = reduce_prompt or self._default_reduce_prompt()
        self.chain = self._create_chain()
    
    @staticmethod
    def _default_map_prompt() -> PromptTemplate:
        """Create default mapping prompt."""
        template = """The following is a chunk of an insurance document:
        {context}
        
        Based on this chunk, what information is relevant to answering: {question}
        
        Relevant information:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    @staticmethod
    def _default_reduce_prompt() -> PromptTemplate:
        """Create default reducing prompt."""
        template = """Given the following extracted information from different parts of an insurance document, 
        provide a comprehensive answer to the question: {question}

        Extracted information:
        {context}

        Final Answer:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def _create_chain(self) -> RunnableSequence:
        """Create the map-reduce chain using the LangChain v0.3 syntax."""
        
        # Define the map function to process individual documents
        map_chain = (
            self.map_prompt 
            | self.llm
            | (lambda x: x.content)  # Extract content from LLM response
        )

        # Function to combine mapped results
        def combine_docs(docs: Sequence[str]) -> str:
            return "\n\n".join(docs)
        
        # Define the reduce function
        reduce_chain = (
            self.reduce_prompt
            | self.llm
            | (lambda x: x.content)  # Extract content from LLM response
        )
        
        # Create the full chain
        chain = RunnableParallel({
            "documents": lambda x: x["documents"],
            "question": lambda x: x["question"],
        }) | {
            "mapped_results": (
                lambda x: [{"context": doc.page_content, "question": x["question"]} 
                        for doc in x["documents"]]
            ) | map_chain.map(),
            "question": lambda x: x["question"],
        } | {
            "context": lambda x: combine_docs(x["mapped_results"]),
            "question": lambda x: x["question"],
        } | reduce_chain

        return chain
    
    async def arun(
        self,
        question: str,
        documents: Optional[List[Document]] = None
    ) -> str:
        """Async run the QA chain."""
        if documents is None:
            if self.retriever is None:
                raise ValueError("Either documents or a retriever must be provided")
            documents = await self.retriever.aget_relevant_documents(question)
        
        return await self.chain.ainvoke({
            "documents": documents,
            "question": question
        })
    
    def run(
        self,
        question: str,
        documents: Optional[List[Document]] = None
    ) -> str:
        """Run the QA chain on a question and documents."""
        if documents is None:
            if self.retriever is None:
                raise ValueError("Either documents or a retriever must be provided")
            documents = self.retriever.get_relevant_documents(question)
        
        return self.chain.invoke({
            "documents": documents,
            "question": question
        })