import asyncio
from typing import Dict, Any
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import AsyncTavilyClient
from agent.state import AgentState, SubQueries, EvaluationResult
import os

# Models initialized lazily or using env vars directly
def get_planner(): return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, api_key=os.environ.get("GOOGLE_API_KEY"))
def get_evaluator(): return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, api_key=os.environ.get("GOOGLE_API_KEY"))
def get_summarizer(): return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, api_key=os.environ.get("GOOGLE_API_KEY"))
def get_tavily(): return AsyncTavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

async def send_progress(queue: asyncio.Queue, node: str, message: str):
    if queue:
        await queue.put({"node": node, "message": message})

async def planner(state: AgentState, config: RunnableConfig):
    queue = config.get("configurable", {}).get("queue")
    loop_count = state.get("loop_count", 0)

    if loop_count > 0:
        await send_progress(queue, "planner", f"Re-planning (attempt {loop_count + 1}) — previous data was insufficient.")
        await send_progress(queue, "planner", f"Reason: {state.get('eval_reasoning', 'N/A')}")
    else:
        await send_progress(queue, "planner", f"Analyzing query: \"{state['query']}\"")

    await send_progress(queue, "planner", "Calling Gemini to decompose query into sub-tasks...")
    
    # Rate limit protection
    await asyncio.sleep(4)
    
    planner_model = get_planner()
    
    context = ""
    if state.get("eval_reasoning"):
        context = f"\n\nPrevious attempt failed because: {state['eval_reasoning']}\nPlease adjust your research plan and sub-queries accordingly to find the missing information."
        
    prompt = f"You are an expert autonomous research planner. Create a step-by-step research plan and decompose this query into specific search queries (max 3).\n\nQuery: {state['query']}{context}"
    llm = planner_model.with_structured_output(SubQueries)
    try:
        result: SubQueries = await llm.ainvoke(prompt)
        await send_progress(queue, "planner", f"Research plan: {result.research_plan}")
        for i, sq in enumerate(result.sub_queries, 1):
            await send_progress(queue, "planner", f"Sub-query #{i}: \"{sq}\"")
        await send_progress(queue, "planner", f"Planning complete — {len(result.sub_queries)} sub-queries ready for search.")
        return {"sub_queries": result.sub_queries}
    except Exception as e:
        await send_progress(queue, "planner", f"Structured output failed ({type(e).__name__}), falling back to raw query.")
        return {"sub_queries": [state["query"]]}

async def searcher(state: AgentState, config: RunnableConfig):
    queue = config.get("configurable", {}).get("queue")
    sub_queries = state.get("sub_queries", [])

    await send_progress(queue, "searcher", f"Starting web search — {len(sub_queries)} queries to execute.")
    
    tavily_client = get_tavily()
    
    async def run_search(q, index):
        try:
            await send_progress(queue, "searcher", f"Searching [{index+1}/{len(sub_queries)}]: \"{q}\"")
            # Stagger concurrent searches slightly
            await asyncio.sleep(index * 2) 
            result = await tavily_client.search(q, search_depth="basic")
            num_results = len(result.get("results", []))
            # Extract source URLs
            sources = [r.get("url", "") for r in result.get("results", []) if r.get("url")]
            if sources:
                top_sources = sources[:3]
                await send_progress(queue, "searcher", f"Found {num_results} results for query #{index+1}. Sources: {', '.join(top_sources)}")
            else:
                await send_progress(queue, "searcher", f"Found {num_results} results for query #{index+1}.")
            return result
        except Exception as e:
            await send_progress(queue, "searcher", f"Search #{index+1} failed: {type(e).__name__}: {e}")
            return {"results": [{"content": str(e)}]}
            
    results_lists = await asyncio.gather(*(run_search(q, i) for i, q in enumerate(sub_queries)))
    
    formatted_results = []
    total_snippets = 0
    for q, res in zip(sub_queries, results_lists):
        if "results" in res:
            snippets = res["results"]
            total_snippets += len(snippets)
            content = "\n".join([r.get("content", "") for r in snippets])
            formatted_results.append(f"Query: {q}\nResults: {content}\n")
    
    await send_progress(queue, "searcher", f"All searches complete — collected {total_snippets} content snippets from {len(sub_queries)} queries.")
    
    current_results = state.get("search_results", []) or []
    current_results.extend(formatted_results)
    
    return {"search_results": current_results}

async def evaluator(state: AgentState, config: RunnableConfig):
    queue = config.get("configurable", {}).get("queue")
    loop_count = state.get("loop_count", 0)
    num_results = len(state.get("search_results", []))

    await send_progress(queue, "evaluator", f"Evaluating {num_results} search result blocks (loop {loop_count + 1}/2)...")
    await send_progress(queue, "evaluator", "Calling Gemini to assess data quality and completeness...")
    
    # Rate limit protection
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
            verdict = "SUFFICIENT" if result.is_sufficient else "MAX LOOPS REACHED"
            await send_progress(queue, "evaluator", f"Verdict: {verdict}")
            await send_progress(queue, "evaluator", f"Reasoning: {result.reasoning}")
            await send_progress(queue, "evaluator", "Proceeding to report generation →")
            return {"is_sufficient": True, "loop_count": loop_count, "eval_reasoning": result.reasoning}
        else:
            await send_progress(queue, "evaluator", f"Verdict: INSUFFICIENT — needs more data.")
            await send_progress(queue, "evaluator", f"Reasoning: {result.reasoning}")
            await send_progress(queue, "evaluator", "Looping back to planner for refined queries →")
            return {"is_sufficient": False, "loop_count": loop_count, "eval_reasoning": result.reasoning}
    except Exception as e:
         loop_count = state.get("loop_count", 0) + 1
         await send_progress(queue, "evaluator", f"Evaluation error ({type(e).__name__}), proceeding with current data.")
         return {"is_sufficient": True, "loop_count": loop_count, "eval_reasoning": "Fallback evaluation."}

async def summarizer(state: AgentState, config: RunnableConfig):
    queue = config.get("configurable", {}).get("queue")
    num_results = len(state.get("search_results", []))

    await send_progress(queue, "summarizer", f"Synthesizing final report from {num_results} data blocks...")
    await send_progress(queue, "summarizer", "Calling Gemini to generate comprehensive markdown report...")
    
    # Rate limit protection
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

    await send_progress(queue, "summarizer", "Waiting for Gemini response...")
    response = await summarizer_model.ainvoke(prompt)

    report_len = len(response.content)
    word_count = len(response.content.split())
    await send_progress(queue, "summarizer", f"Report generated — {word_count} words, {report_len} characters.")
    await send_progress(queue, "summarizer", "Report ready ✓")
    
    return {"report": response.content}
