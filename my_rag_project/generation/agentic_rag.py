from typing import Literal, TypedDict
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.retrievers import BaseRetriever

class AgentState(TypedDict):
    user_query: str
    partial_answer: str
    final_answer: str
    approach: Literal["short", "long"]
    done: bool

class AgenticRAG:
    def __init__(self, retriever: BaseRetriever, llm_model: str = "gpt-3.5-turbo"):
        self.retriever = retriever
        self.llm_model = llm_model
        self.setup_pipelines()
        self.build_graph()
        
    def setup_pipelines(self):
        self.short_qa_pipeline = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model_name=self.llm_model),
            chain_type="stuff",
            retriever=self.retriever
        )
        
        self.long_qa_pipeline = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model_name=self.llm_model),
            chain_type="map_reduce",
            retriever=self.retriever
        )

    # 1. Decide tool/approach
    def decide_approach(self, state: AgentState):
        """
        Basic logic for deciding short vs. long QA.
        Could be replaced by a classification LLM call or more advanced router.
        """
        query = state["user_query"]
        if len(query) > 200 or "summarize" in query.lower():
            return {"approach": "long"}
        return {"approach": "short"}

    # 2. Generate answer using chosen approach
    def generate_answer(self, state: AgentState):
        approach = state["approach"]
        query = state["user_query"]
        
        if approach == "short":
            answer = self.short_qa_pipeline.run(query)
        else:
            answer = self.long_qa_pipeline.run(query)
        
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
            # For demonstration, we'll do a single loop extension for “long” approach.
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