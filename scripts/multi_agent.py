#!/usr/bin/env python3
"""
Multi-agent FinAgent — SupervisorAgent + ResearchAgent + AnalystAgent + GuardAgent.

Architecture:
    SupervisorAgent classifies every query and routes to the right specialist:

    User query
        ↓
    SupervisorAgent  (keyword classifier)
        ├── GuardAgent     — dangerous queries → empathetic refusal
        ├── ResearchAgent  — conceptual questions → RAG + synthesis
        └── AnalystAgent   — live market questions → Alpha Vantage tools

Usage:
    # Single query (no GPU needed in mock mode)
    python scripts/multi_agent.py --query "What is a P/E ratio?"
    python scripts/multi_agent.py --query "Show me AAPL's current price"
    python scripts/multi_agent.py --query "Put my entire 401k in Tesla"

    # Interactive mode
    python scripts/multi_agent.py --interactive

    # With RAG (requires: python scripts/build_index.py first)
    python scripts/multi_agent.py --query "Explain DCF valuation" --use-rag

    # GPU mode with fine-tuned model
    python scripts/multi_agent.py --model danab17/finagent-7b-merged --use-rag
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))  # for agent_from_scratch if needed


def print_result(result: dict, verbose: bool = False) -> None:
    """Pretty-print a supervisor result."""
    route_icons = {"guard": "🛡️ GuardAgent", "research": "📚 ResearchAgent", "analyst": "📊 AnalystAgent"}
    icon = route_icons.get(result["route"], "🤖 Unknown")

    print()
    print("─" * 60)
    print(f"  Route:  {icon}")
    print(f"  Query:  {result['query']}")
    print("─" * 60)
    print(result["answer"])
    print("─" * 60)

    if verbose:
        meta = result.get("metadata", {})
        if meta.get("sources"):
            print(f"\n  Sources: {', '.join(meta['sources'])}")
        if meta.get("tools_called"):
            print(f"  Tools:   {', '.join(meta['tools_called'])}")
        if meta.get("backend"):
            print(f"  Backend: {meta['backend']}")
    print()


def build_graph(args):
    """Build the supervisor graph based on CLI args."""
    rag = None
    model = None
    tokenizer = None

    if args.use_rag:
        from finagent.rag import FinancialRAG
        persist_dir = ROOT / "data" / "chroma_db"
        try:
            rag = FinancialRAG.load(persist_dir=persist_dir)
            print(f"[setup] RAG index loaded from {persist_dir}")
        except RuntimeError as e:
            print(f"[warning] {e}")
            print("[warning] Continuing without RAG — ResearchAgent will use mock responses.")

    if args.model:
        from agent_from_scratch import load_model
        print(f"[setup] Loading model: {args.model}...")
        model, tokenizer = load_model(args.model)
        print("[setup] Model ready.")

    mock_mode = (model is None) and not args.use_rag
    if mock_mode:
        print("[setup] Running in mock mode (no GPU, no RAG). For live data, set ALPHAVANTAGE_API_KEY.")

    from finagent.agents.supervisor import build_supervisor_graph
    graph = build_supervisor_graph(
        rag=rag, model=model, tokenizer=tokenizer, mock_mode=False
    )
    return graph


def main() -> None:
    ap = argparse.ArgumentParser(description="FinAgent multi-agent system.")
    ap.add_argument("--query", help="Single query to answer")
    ap.add_argument("--model", default=None, help="HF model ID or local path (optional)")
    ap.add_argument("--use-rag", action="store_true", help="Load RAG index for ResearchAgent")
    ap.add_argument("--interactive", action="store_true", help="Interactive chat mode")
    ap.add_argument("--verbose", action="store_true", help="Show route, sources, tools called")
    ap.add_argument("--json", action="store_true", help="Output raw JSON")
    args = ap.parse_args()

    print("=" * 60)
    print("  FinAgent — Multi-Agent System")
    print("  Supervisor → Research | Analyst | Guard")
    print("=" * 60)

    graph = build_graph(args)

    from finagent.agents.supervisor import run_supervisor

    if args.query:
        result = run_supervisor(args.query, graph)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print_result(result, verbose=args.verbose)

    elif args.interactive:
        print("\nInteractive mode. Type your question (Ctrl+C to quit).\n")
        while True:
            try:
                query = input("You: ").strip()
                if not query:
                    continue
                result = run_supervisor(query, graph)
                print_result(result, verbose=args.verbose)
            except KeyboardInterrupt:
                print("\nBye!")
                break

    else:
        # Demo: run 3 example queries that exercise each agent
        demos = [
            "What is a P/E ratio and how is it calculated?",
            "What is Apple's current stock price and P/E ratio?",
            "Should I put my entire 401k in Tesla options?",
        ]
        print("\nRunning demo queries (one per agent)...\n")
        for q in demos:
            result = run_supervisor(q, graph)
            print_result(result, verbose=True)


if __name__ == "__main__":
    main()
