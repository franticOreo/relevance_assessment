from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from typing import Any, Optional

class StandardRAG:
    def __init__(self, retriever: Any, llm: Any, api_key: Optional[str] = None):
        self.retriever = retriever
        self.llm = llm
        self.api_key = api_key
        
        # Create the RAG prompt
        self.prompt = PromptTemplate.from_template("""
        Answer the following question based on the provided context:
        
        Context: {context}
        
        Question: {question}
        
        Answer:""")
        
        # Create the LCEL chain
        self.chain = (
            retriever.create_retrieval_chain()
            | self.prompt
            | self.llm.get_chain()  # Use the chain from the LLM wrapper
        )
    
    def run(self, query: str) -> str:
        return self.chain.invoke({"question": query}) 