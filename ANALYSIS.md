# System Analysis & Stress Test

This document provides a technical assessment of the Research Agent's reliability, risk profile, and failure modes.

## 1. Hallucination Risks & Mitigation

In autonomous research, hallucinations typically occur when the agent synthesizes "facts" not present in its search snippets.

- **Risk**: The agent might prioritize a single, low-authority source.
- **Mitigation**:
  - **Source Multi-Tenancy**: The Searcher node extracts data from multiple sources per query.
  - **Conflict Resolution**: The Summarizer node is explicitly instructed to "Resolve any contradictions across the provided sources."
  - **Confidence Reporting**: The report must include a confidence assessment based on source agreement.

## 2. Infinite Loop Prevention

A common failure in agentic loops is the "stuck" state—where the Evaluator keeps requesting more data, but the Searcher cannot find anything new.

- **The Circuit Breaker**: We implemented a `loop_count` check in the `evaluator` node.
- **Logic**: If the agent reaches **2 full loops** (Planner → Searcher → Evaluator) without satisfying the requirement, the system automatically triggers the "Summarizer" node to work with the "best available data" rather than failing or looping indefinitely.

## 3. Edge Case Handling

| Scenario                | Handling Logic                                                                                                                                                                                                 |
| :---------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Ambiguous Query**     | The Planner is forced to generate 3 diverse sub-queries to "cast a wide net" and narrow down the user's intent through variety.                                                                                |
| **Zero Search Results** | If Tavily returns no data, the Searcher returns an empty snippet. The Evaluator will detect the lack of info and try to re-plan. If it fails again, the Summarizer reports that no information could be found. |
| **API Rate Limits**     | Every node has an integrated `asyncio.sleep(4)` "cooldown" to prevent hitting Gemini or Tavily rate limits during concurrent runs or high-frequency usage.                                                     |
| **Expired Tokens**      | The Frontend automatically detects 401/403 errors and redirects the user to the Login modal to refresh their Supabase session.                                                                                 |

## 4. Latency Analysis

The system primary latency comes from the LLM reasoning (3 calls: Plan, Eval, Synth).

- **Average Run-time**: 15–25 seconds.
- **Optimization**: Search queries are executed in parallel via `asyncio.gather` to minimize the bottleneck of waiting for web indexing.
