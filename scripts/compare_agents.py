#!/usr/bin/env python3
"""
Side-by-side comparison: from-scratch ReAct loop vs LangGraph.

Both agents run the same set of queries against the same fine-tuned model.
We measure:
  - lines of orchestration code (smaller → simpler mental model)
  - wall-clock latency per query
  - identical outputs? (a sanity check that the LangGraph version preserves semantics)

Output: results/compare_report.md

Run:
    python scripts/compare_agents.py                 # uses default model
    python scripts/compare_agents.py --queries 3     # short run
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

DEFAULT_QUERIES = [
    "What is Apple's current price?",
    "Is the US yield curve inverted right now?",
    "Give me Microsoft's P/E ratio and ROE.",
    "What's the macro picture — CPI, unemployment, Fed rate?",
    "Show me healthcare stocks above 100B market cap.",
]


def count_orchestration_lines(path: Path, anchors: list[str]) -> int:
    """Count non-empty non-comment lines between the anchor markers."""
    text = path.read_text()
    total = 0
    for start_marker in anchors:
        if start_marker not in text:
            continue
        block = text.split(start_marker, 1)[1]
        # Stop at the next === STEP marker or main()
        for stop in ["# =====", "def main(", "if __name__"]:
            if stop in block:
                block = block.split(stop, 1)[0]
                break
        for ln in block.splitlines():
            stripped = ln.strip()
            if stripped and not stripped.startswith("#"):
                total += 1
    return total


def time_from_scratch(query: str) -> tuple[float, str]:
    from agent_from_scratch import react_loop  # type: ignore

    t0 = time.time()
    out = react_loop(MODEL, TOKENIZER, query, max_iterations=5)
    return time.time() - t0, out


def time_langgraph(query: str) -> tuple[float, str]:
    from agent_langgraph import build_graph, run_agent  # type: ignore

    global GRAPH
    if GRAPH is None:
        GRAPH = build_graph(MODEL, TOKENIZER)
    t0 = time.time()
    out = run_agent(GRAPH, query)
    return time.time() - t0, out


MODEL = None
TOKENIZER = None
GRAPH = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="danab17/finagent-7b-merged")
    ap.add_argument("--queries", type=int, default=len(DEFAULT_QUERIES))
    ap.add_argument("--out", default="results/compare_report.md")
    args = ap.parse_args()

    global MODEL, TOKENIZER
    from agent_from_scratch import load_model  # shared between both agents

    print(f"Loading {args.model}...")
    MODEL, TOKENIZER = load_model(args.model)
    print("Loaded.\n")

    queries = DEFAULT_QUERIES[: args.queries]
    rows = []
    for q in queries:
        print(f"  query: {q}")
        t_scratch, out_scratch = time_from_scratch(q)
        t_graph, out_graph = time_langgraph(q)
        rows.append({
            "query": q,
            "scratch_s": round(t_scratch, 2),
            "langgraph_s": round(t_graph, 2),
            "scratch_out": out_scratch[:300],
            "langgraph_out": out_graph[:300],
        })

    loc_scratch = count_orchestration_lines(
        ROOT / "scripts/agent_from_scratch.py",
        ["def react_loop("],
    )
    loc_graph = count_orchestration_lines(
        ROOT / "scripts/agent_langgraph.py",
        ["def make_generate_node(", "def execute_tools_node(", "def should_continue(", "def build_graph("],
    )

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# FinAgent — from-scratch vs LangGraph",
        "",
        "Same model, same queries. Numbers below are wall-clock for one greedy pass each.",
        "",
        "## Orchestration code size",
        "",
        "| Implementation | Lines of orchestration code |",
        "|---|---|",
        f"| from-scratch `react_loop` | {loc_scratch} |",
        f"| LangGraph nodes + graph builder | {loc_graph} |",
        "",
        "## Latency per query",
        "",
        "| Query | from-scratch (s) | LangGraph (s) |",
        "|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['query']} | {r['scratch_s']} | {r['langgraph_s']} |")

    avg_s = round(sum(r["scratch_s"] for r in rows) / len(rows), 2)
    avg_g = round(sum(r["langgraph_s"] for r in rows) / len(rows), 2)
    lines += ["", f"**Mean latency**: from-scratch {avg_s}s · LangGraph {avg_g}s", ""]
    lines += ["## Raw outputs (truncated to 300 chars)", ""]
    for r in rows:
        lines += [
            f"### {r['query']}",
            "",
            "**from-scratch:**",
            "```",
            r["scratch_out"],
            "```",
            "**LangGraph:**",
            "```",
            r["langgraph_out"],
            "```",
            "",
        ]

    out_path.write_text("\n".join(lines))
    print(f"\nWrote {out_path}")
    print(f"  orchestration LOC: scratch={loc_scratch} · langgraph={loc_graph}")
    print(f"  mean latency: scratch={avg_s}s · langgraph={avg_g}s")


if __name__ == "__main__":
    main()
