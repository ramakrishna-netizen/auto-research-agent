from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes import planner, searcher, evaluator, summarizer

def build_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("planner", planner)
    workflow.add_node("searcher", searcher)
    workflow.add_node("evaluator", evaluator)
    workflow.add_node("summarizer", summarizer)
    
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "searcher")
    workflow.add_edge("searcher", "evaluator")
    
    def check_sufficient(state: AgentState):
        if state.get("is_sufficient", False):
            return "summarizer"
        return "planner"
        
    workflow.add_conditional_edges(
        "evaluator",
        check_sufficient,
        {
            "summarizer": "summarizer",
            "planner": "planner"
        }
    )
    
    workflow.add_edge("summarizer", END)
    
    return workflow.compile()
