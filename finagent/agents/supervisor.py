"""
SupervisorAgent — routes queries to the correct specialist agent.

Routing logic (keyword-based, fast and explainable):
  "guard"    — detected dangerous pattern (always checked first)
  "research" — conceptual / educational questions (how, why, what is, explain)
  "analyst"  — data / market questions (price, P/E, yield, news, screen)

The graph:
    START → classify → route?
               ├── "guard"    → guard_node    → END
               ├── "research" → research_node → END
               └── "analyst"  → analyst_node  → END

Each node writes its result into state["final_answer"].
"""

from __future__ import annotations

import operator
import re
from typing import Annotated, Any, TypedDict

from finagent.agents.analyst import AnalystAgent
from finagent.agents.guard import GuardAgent
from finagent.agents.research import ResearchAgent

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class MultiAgentState(TypedDict):
    query: str
    route: str                               # "guard" | "research" | "analyst"
    messages: Annotated[list[dict], operator.add]
    rag_context: list[dict]                  # populated by research_node
    tool_results: list[dict]                 # populated by analyst_node
    final_answer: str
    metadata: dict                           # backend, sources, etc.


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

# Patterns that suggest a RESEARCH question (conceptual, educational)
_RESEARCH_PATTERNS = [
    r"\b(what is|what are|define|describe)\b",
    r"\b(how does|how do|how is|how are|explain|why is|why does|why are)\b",
    r"\b(history of|background|overview|primer|introduction to)\b",
    r"\b(difference between|compare|vs\.?|versus)\s+\w+\s+and\b",
    r"\b(strategy|strategies|approach|framework|methodology)\b",
    r"\b(risk|risks|diversif|portfolio theory|allocation|rebalancing)\b",
    r"\b(valuation method|how to value|dcf|discounted cash flow|wacc|capm)\b",
    r"\b(earnings report|income statement|balance sheet|cash flow statement)\b",
    r"\b(sector|industry|market structure|exchange|index definition)\b",
]

# Patterns that suggest an ANALYST question (live data)
_ANALYST_PATTERNS = [
    r"\b(current|today|right now|live|real.?time|latest|now)\b",
    r"\b(stock price|share price|trading at|market cap)\b",
    r"\b(p/?e ratio|price.to.earn|forward pe|trailing pe)\b",
    r"\b(yield curve|treasury yield|t.?bill|10.year)\b",
    r"\b(screen|filter stocks|find me stocks|which stocks)\b",
    r"\b(news|headlines|articles|recent|this week|this month)\b",
    r"\b(cpi|inflation rate|unemployment rate|fed funds|gdp)\b",
    r"\b(compare|analyse|analyze)\s+\w+\s+(and|vs\.?|with)\s+\w+\b",
]

_RESEARCH_RE = [re.compile(p, re.IGNORECASE) for p in _RESEARCH_PATTERNS]
_ANALYST_RE = [re.compile(p, re.IGNORECASE) for p in _ANALYST_PATTERNS]


def route_query(query: str) -> str:
    """Classify a query into 'guard', 'research', or 'analyst'.

    Guard check runs first (safety is not negotiable).
    Then score research vs analyst patterns — highest score wins.
    Default: 'analyst' (most queries want live data).
    """
    from finagent.agents.guard import classify as guard_classify

    is_dangerous, _, _ = guard_classify(query)
    if is_dangerous:
        return "guard"

    research_score = sum(1 for p in _RESEARCH_RE if p.search(query))
    analyst_score = sum(1 for p in _ANALYST_RE if p.search(query))

    if research_score > analyst_score:
        return "research"
    return "analyst"


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


def make_classify_node() -> Any:
    """Returns a node function that routes the query."""

    def classify_node(state: MultiAgentState) -> dict:
        route = route_query(state["query"])
        return {
            "route": route,
            "messages": [{"role": "system", "content": f"[Supervisor] Route: {route}"}],
        }

    return classify_node


def make_guard_node(agent: GuardAgent) -> Any:
    def guard_node(state: MultiAgentState) -> dict:
        result = agent.run(state["query"])
        answer = result["refusal"] if not result["safe"] else "Query is safe to proceed."
        return {
            "final_answer": answer,
            "metadata": {"agent": "guard", "risk_label": result.get("risk_label", "")},
            "messages": [{"role": "assistant", "content": answer}],
        }

    return guard_node


def make_research_node(agent: ResearchAgent) -> Any:
    def research_node(state: MultiAgentState) -> dict:
        result = agent.run(state["query"])
        return {
            "final_answer": result["answer"],
            "rag_context": result["context_chunks"],
            "metadata": {
                "agent": "research",
                "backend": result["backend"],
                "sources": result.get("sources", []),
            },
            "messages": [{"role": "assistant", "content": result["answer"]}],
        }

    return research_node


def make_analyst_node(agent: AnalystAgent) -> Any:
    def analyst_node(state: MultiAgentState) -> dict:
        result = agent.run(state["query"])
        return {
            "final_answer": result["answer"],
            "tool_results": result["tool_results"],
            "metadata": {
                "agent": "analyst",
                "backend": result["backend"],
                "tools_called": [tc["tool"] for tc in result["tool_calls"]],
            },
            "messages": [{"role": "assistant", "content": result["answer"]}],
        }

    return analyst_node


def _should_route(state: MultiAgentState) -> str:
    """Conditional edge: reads state['route'] and returns the node name."""
    return state["route"]


# ---------------------------------------------------------------------------
# Pure-Python fallback graph (no LangGraph dependency)
# ---------------------------------------------------------------------------


class _PythonSupervisorGraph:
    """Minimal supervisor that exposes the same invoke() interface as a compiled
    LangGraph but has no langgraph/langchain_protocol dependency.

    Used automatically when langgraph is unavailable or incompatible with the
    current Python version (langchain_protocol requires Python 3.13+ TypedDict
    extras; this fallback keeps the test suite green on Python 3.12).
    """

    def __init__(
        self,
        guard_agent: GuardAgent,
        research_agent: ResearchAgent,
        analyst_agent: AnalystAgent,
    ) -> None:
        self._nodes = {
            "guard": make_guard_node(guard_agent),
            "research": make_research_node(research_agent),
            "analyst": make_analyst_node(analyst_agent),
        }
        self._classify = make_classify_node()

    def invoke(self, state: dict) -> dict:
        # Classify first (writes route into state)
        state = {**state, **self._classify(state)}
        # Dispatch to the chosen node
        node_fn = self._nodes[state["route"]]
        state = {**state, **node_fn(state)}
        return state


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_supervisor_graph(
    rag: Any | None = None,
    model: Any | None = None,
    tokenizer: Any | None = None,
    mock_mode: bool = False,
) -> Any:
    """Build the multi-agent supervisor graph.

    Tries to use LangGraph for the full StateGraph implementation. Falls back
    to a pure-Python dispatcher (_PythonSupervisorGraph) when langgraph is
    unavailable or incompatible with the current Python version.

    Args:
        rag:        FinancialRAG instance (or None for mock_mode).
        model:      Fine-tuned model (or None for GPT-4o-mini/mock).
        tokenizer:  Paired tokenizer.
        mock_mode:  If True, all agents run in mock mode (no GPU/API).

    Returns:
        A compiled graph with an invoke(state) method.
    """
    guard_agent = GuardAgent(mock_mode=mock_mode)
    research_agent = ResearchAgent(rag=rag, model=model, tokenizer=tokenizer, mock_mode=mock_mode)
    analyst_agent = AnalystAgent(model=model, tokenizer=tokenizer, mock_mode=mock_mode)

    try:
        from langgraph.graph import END, StateGraph  # noqa: PLC0415

        graph = StateGraph(MultiAgentState)
        graph.add_node("classify", make_classify_node())
        graph.add_node("guard", make_guard_node(guard_agent))
        graph.add_node("research", make_research_node(research_agent))
        graph.add_node("analyst", make_analyst_node(analyst_agent))
        graph.set_entry_point("classify")
        graph.add_conditional_edges(
            "classify",
            _should_route,
            {"guard": "guard", "research": "research", "analyst": "analyst"},
        )
        graph.add_edge("guard", END)
        graph.add_edge("research", END)
        graph.add_edge("analyst", END)
        return graph.compile()

    except Exception:
        # langgraph not installed or incompatible (e.g. langchain_protocol requires
        # Python 3.13+ TypedDict extras; we degrade gracefully on 3.12).
        return _PythonSupervisorGraph(guard_agent, research_agent, analyst_agent)


def run_supervisor(
    query: str,
    graph: Any,
) -> dict:
    """Run the supervisor graph on a single query.

    Returns a clean result dict with answer, route, agent, and metadata.
    """
    initial_state: MultiAgentState = {
        "query": query,
        "route": "",
        "messages": [{"role": "user", "content": query}],
        "rag_context": [],
        "tool_results": [],
        "final_answer": "",
        "metadata": {},
    }
    final_state = graph.invoke(initial_state)
    return {
        "query": query,
        "route": final_state["route"],
        "answer": final_state["final_answer"],
        "agent": final_state["metadata"].get("agent", "unknown"),
        "metadata": final_state["metadata"],
    }
