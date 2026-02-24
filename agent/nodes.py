import asyncio
from typing import Dict, Any
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import AsyncTavilyClient
from agent.state import AgentState, SubQueries, EvaluationResult
import os

# Models initialized lazily or using env vars directly
def get_planner(): return ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)
def get_evaluator(): return ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)
def get_summarizer(): return ChatGoogleGenerativeAI(model="gemini-pro-latest", temperature=0.2)
def get_tavily(): return AsyncTavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

async def send_progress(queue: asyncio.Queue, node: str, message: str):
    if queue:
        await queue.put({"node": node, "message": message})

async def planner(state: AgentState, config: RunnableConfig):
    queue = config.get("configurable", {}).get("queue")
    await send_progress(queue, "planner", "Decomposing query into sub-tasks...")
    
    # Rate limit protection: Added sleep delay to not exceed the daily rate limit for the free tier APIs
    await asyncio.sleep(4)
    
    planner_model = get_planner()
    
    # Include context if we are looping back from a failed evaluation
    context = ""
    if state.get("eval_reasoning"):
        context = f"\n\nPrevious attempt failed because: {state['eval_reasoning']}\nPlease adjust your research plan and sub-queries accordingly to find the missing information."
        
    prompt = f"You are an expert autonomous research planner. Create a step-by-step research plan and decompose this query into specific search queries (max 3).\n\nQuery: {state['query']}{context}"
    llm = planner_model.with_structured_output(SubQueries)
    try:
        result: SubQueries = await llm.ainvoke(prompt)
        await send_progress(queue, "planner", f"Plan: {result.research_plan}\nGenerated {len(result.sub_queries)} sub-queries.")
        return {"sub_queries": result.sub_queries}
    except Exception as e:
        await send_progress(queue, "planner", f"LLM parsing failed, using simple approach. Error: {e}")
        return {"sub_queries": [state["query"]]}

async def searcher(state: AgentState, config: RunnableConfig):
    queue = config.get("configurable", {}).get("queue")
    await send_progress(queue, "searcher", "Executing parallel searches using Tavily...")
    
    tavily_client = get_tavily()
    sub_queries = state.get("sub_queries", [])
    
    async def run_search(q, index):
        try:
            # Stagger concurrent searches slightly to avoid bursting the search API and exceeding daily rate limits
            await asyncio.sleep(index * 2) 
            return await tavily_client.search(q, search_depth="basic")
        except Exception as e:
            return {"results": [{"content": str(e)}]}
            
    results_lists = await asyncio.gather(*(run_search(q, i) for i, q in enumerate(sub_queries)))
    
    formatted_results = []
    for q, res in zip(sub_queries, results_lists):
        if "results" in res:
            content = "\n".join([r.get("content", "") for r in res["results"]])
            formatted_results.append(f"Query: {q}\nResults: {content}\n")
    
    await send_progress(queue, "searcher", "Parallel searches completed.")
    
    current_results = state.get("search_results", []) or []
    current_results.extend(formatted_results)
    
    return {"search_results": current_results}

async def evaluator(state: AgentState, config: RunnableConfig):
    queue = config.get("configurable", {}).get("queue")
    await send_progress(queue, "evaluator", "Evaluating if gathered information is sufficient...")
    
    # Rate limit protection: Added sleep delay to not exceed the daily rate limit for the free tier APIs
    await asyncio.sleep(4)
    
    query = state["query"]
    results = "\n\n".join(state.get("search_results", []))
    
    prompt = f"Original Query: {query}\n\nGathered Information:\n{results}\n\nDoes the gathered information sufficiently answer the query in detail? Reason step by step."
    
    evaluator_model = get_evaluator()
    llm = evaluator_model.with_structured_output(EvaluationResult)
    try:
        result: EvaluationResult = await llm.ainvoke(prompt)
        loop_count = state.get("loop_count", 0) + 1
        
        if result.is_sufficient or loop_count >= 2:
            await send_progress(queue, "evaluator", f"Information is sufficient (or max loops reached). Logic: {result.reasoning}")
            return {"is_sufficient": True, "loop_count": loop_count, "eval_reasoning": result.reasoning}
        else:
            await send_progress(queue, "evaluator", f"Information not sufficient. Logic: {result.reasoning}")
            return {"is_sufficient": False, "loop_count": loop_count, "eval_reasoning": result.reasoning}
    except Exception as e:
         # Fallback
         loop_count = state.get("loop_count", 0) + 1
         await send_progress(queue, "evaluator", "Evaluation failed, proceeding anyway.")
         return {"is_sufficient": True, "loop_count": loop_count, "eval_reasoning": "Fallback evaluation."}

async def summarizer(state: AgentState, config: RunnableConfig):
    queue = config.get("configurable", {}).get("queue")
    await send_progress(queue, "summarizer", "Synthesizing final report...")
    
    # Rate limit protection: Added sleep delay to not exceed the daily rate limit for the free tier APIs
    await asyncio.sleep(4)
    
    query = state["query"]
    results = "\n\n".join(state.get("search_results", []))
    
    prompt = f"""Original Query: {query}
    
Gathered Information:
{results}

Write a comprehensive final report to answer the query using the above information. Format nicely in Markdown.
CRITICAL INSTRUCTIONS:
1. Synthesize the findings completely. 
2. Resolve any contradictions across the provided sources. 
3. Explicitly state the confidence level of your conclusions based on the context.
4. Cite your sources inline where appropriate."""
    
    summarizer_model = get_summarizer()
    response = await summarizer_model.ainvoke(prompt)
    
    await send_progress(queue, "summarizer", "Report generated successfully.")
    
    return {"report": response.content}
