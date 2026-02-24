from typing import TypedDict, List
from pydantic import BaseModel, Field

class AgentState(TypedDict):
    """The State of the Autonomous Research Agent."""
    query: str
    sub_queries: List[str]
    search_results: List[str]
    report: str
    loop_count: int
    is_sufficient: bool

class SubQueries(BaseModel):
    """Pydantic model for extracting sub-queries from LLM."""
    sub_queries: List[str] = Field(description="A list of specific search queries to execute to gather information about the topic.")

class EvaluationResult(BaseModel):
    """Pydantic model for evaluating search results."""
    is_sufficient: bool = Field(description="Whether the gathered search results contain enough detailed information to comprehensively answer the original query.")
    reasoning: str = Field(description="Reasoning for why the information is or isn't sufficient.")
