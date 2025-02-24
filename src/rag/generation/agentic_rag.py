from typing import Literal, TypedDict, Union, Any
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import PromptTemplate
from langchain.chat_models.base import BaseChatModel

class AgentState(TypedDict):
    user_query: str
    partial_answer: str
    final_answer: str
    approach: Literal["short", "long"]
    done: bool

class AgenticRAG:
    def __init__(self, retriever: BaseRetriever, llm_model: Union[str, Any] = "gpt-3.5-turbo"):
        self.retriever = retriever
        self.llm_model = llm_model
        self.setup_pipelines()
        self.build_graph()
        
    def setup_pipelines(self):
        # Create the prompts
        short_prompt = PromptTemplate.from_template("""
        Based on the provided insurance policy documents, provide a detailed comparison or answer to the following question. Include specific details and examples where relevant.
        
        Context: {context}
        Question: {question}
        
        Detailed Answer:""")
        
        long_prompt = PromptTemplate.from_template("""
        Based on the provided insurance policy documents, provide a comprehensive analysis and comparison, including specific policy details, coverage limits, and key differences where applicable.
        
        Context: {context}
        Question: {question}
        
        Comprehensive Answer:""")
        
        # Get the appropriate LLM instance
        llm = (self.llm_model if isinstance(self.llm_model, BaseChatModel) 
               else ChatOpenAI(model_name=self.llm_model if isinstance(self.llm_model, str) 
                             else getattr(self.llm_model, 'model_name', 'gpt-4o-mini')))
        
        # Create the LCEL chains with proper retrieval setup using invoke
        self.short_qa_pipeline = (
            {
                "context": lambda x: "\n\n".join(
                    doc.page_content for doc in self.retriever.invoke(
                        str(x["question"])  # Ensure string type
                    )
                ),
                "question": lambda x: str(x["question"])  # Ensure string type
            }
            | short_prompt
            | llm
        )
        
        self.long_qa_pipeline = (
            {
                "context": lambda x: "\n\n".join(
                    doc.page_content for doc in self.retriever.invoke(
                        str(x["question"])  # Ensure string type
                    )
                ),
                "question": lambda x: str(x["question"])  # Ensure string type
            }
            | long_prompt
            | llm
        )

    # 1. Decide tool/approach
    def decide_approach(self, state: AgentState):
        """
        Enhanced logic for deciding between short vs. long QA.
        """
        query = state["user_query"].lower()
        
        # Keywords that suggest need for detailed response
        detail_keywords = ["compare", "difference", "detail", "explain", "list", "versus", "vs"]
        
        if any(keyword in query for keyword in detail_keywords):
            return {"approach": "long"}
        return {"approach": "short"}

    # 2. Generate answer using chosen approach
    def generate_answer(self, state: AgentState):
        """Generate answer using chosen approach."""
        approach = state["approach"]
        query = state["user_query"]
        
        if approach == "short":
            answer = self.short_qa_pipeline.invoke({"question": query}).content
        else:
            answer = self.long_qa_pipeline.invoke({"question": query}).content
        
        return {"partial_answer": answer}

    # 3. Evaluate if we should stop or gather more context
    def evaluate_answer(self, state: AgentState):
        """
        We'll do a basic check or optionally call an LLM to see if the answer is 'complete enough'.
        For a simpler approach, we pass if partial_answer is "long enough" or
        if the user is specifically requesting more. 
        You can replace with a tool call or any custom logic you like.
        """
        partial_ans = state["partial_answer"]
        # Simple check. Replace with a real evaluator LLM if you want.
        if len(partial_ans) > 400:
            return {"done": True}
        else:
            # Possibly the user wants more details or the partial answer is found lacking.
            # For demonstration, we'll do a single loop extension for "long" approach.
            if state["approach"] == "long":
                return {"done": True}
            return {"done": True}  # Or set to False for multiple loops

    # 4. If done, finalize answer
    def finalize_answer(self, state: AgentState):
        return {"final_answer": state["partial_answer"]}
    
    # If done = True -> finalize_answer, else -> we could loop back to generate_answer
    def route_evaluation(self, state: AgentState):
        if state["done"]:
            return "Done"
        else:
            return "Repeat"

    def build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("decide_approach", self.decide_approach)
        graph.add_node("generate_answer", self.generate_answer)
        graph.add_node("evaluate_answer", self.evaluate_answer)
        graph.add_node("finalize_answer", self.finalize_answer)

        # Connect the nodes
        graph.add_edge(START, "decide_approach")
        graph.add_edge("decide_approach", "generate_answer")
        graph.add_edge("generate_answer", "evaluate_answer")

        graph.add_conditional_edges(
            "evaluate_answer",
            self.route_evaluation,
            {
                "Done": "finalize_answer",
                "Repeat": "generate_answer"  
            },
        )

        graph.add_edge("finalize_answer", END)
        self.workflow = graph.compile()

    def run(self, query: str) -> str:
        """
        Alias for run_query to maintain interface consistency with other RAG implementations.
        """
        return self.run_query(query)

    def run_query(self, user_query: str) -> str:
        initial_state = {
            "user_query": user_query,
            "partial_answer": "",
            "final_answer": "",
            "approach": "short",
            "done": False
        }
        final_state = self.workflow.invoke(initial_state)
        return final_state["final_answer"]