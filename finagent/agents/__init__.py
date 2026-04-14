"""
Multi-agent system: SupervisorAgent → ResearchAgent | AnalystAgent | GuardAgent.

Architecture:
    User query
        ↓
    SupervisorAgent  — classifies query into one of three routes
        ├── "research"  → ResearchAgent  — RAG over financial docs + synthesis
        ├── "analyst"   → AnalystAgent   — Alpha Vantage live data + tool calls
        └── "guard"     → GuardAgent     — safety check + empathetic refusal

Imports are lazy: import from submodules directly to avoid pulling in optional
heavy dependencies (langgraph, chromadb) at package import time.
"""

__all__ = ["build_supervisor_graph", "route_query", "MultiAgentState"]
