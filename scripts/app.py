#!/usr/bin/env python3
"""
FinAgent — Gradio chat demo.

Shows the ReAct loop turn-by-turn:
  - the <think> block (model's internal reasoning)
  - each tool call with its arguments
  - each tool result
  - the final synthesized answer

Run:
    python scripts/app.py            # default: localhost:7860
    python scripts/app.py --share    # public Gradio link
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # for agent_from_scratch

import gradio as gr  # noqa: E402
from agent_from_scratch import (  # noqa: E402
    execute_tools,
    generate,
    load_model,
    parse_tool_calls,
)

from configs.prompt_templates import SYSTEM_PROMPT  # noqa: E402

MODEL_PATH = "danab17/finagent-7b-merged"


def format_trace(trace: list[dict]) -> str:
    """Render the agent's intermediate steps as markdown."""
    lines = []
    for i, step in enumerate(trace, 1):
        if step["kind"] == "thought":
            lines.append(f"### 🧠 Iter {i} — model thought\n```\n{step['text']}\n```")
        elif step["kind"] == "tool_call":
            args = json.dumps(step["arguments"], indent=2)
            lines.append(f"### 🔧 Iter {i} — call `{step['name']}`\n```json\n{args}\n```")
        elif step["kind"] == "tool_result":
            result = json.dumps(step["result"], indent=2)[:1500]
            lines.append(f"### 📊 Iter {i} — result from `{step['name']}`\n```json\n{result}\n```")
    return "\n\n".join(lines)


def run_query(model, tokenizer, query: str, max_iters: int = 5) -> tuple[str, str]:
    """Run one ReAct loop. Returns (final_answer_markdown, trace_markdown)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    trace: list[dict] = []
    final = ""

    for _ in range(max_iters):
        text = generate(model, tokenizer, messages)

        # Separate the <think>...</think> block from the rest
        think = ""
        body = text
        if "<think>" in text and "</think>" in text:
            think = text.split("<think>", 1)[1].split("</think>", 1)[0].strip()
            body = text.split("</think>", 1)[1].strip()
        if think:
            trace.append({"kind": "thought", "text": think})

        tool_calls = parse_tool_calls(body)
        if not tool_calls:
            final = body
            break

        for tc in tool_calls:
            trace.append({
                "kind": "tool_call",
                "name": tc.get("name", "?"),
                "arguments": tc.get("arguments", {}),
            })

        messages.append({"role": "assistant", "content": text})
        for r in execute_tools(tool_calls):
            trace.append({"kind": "tool_result", "name": r["name"], "result": r["result"]})
            messages.append({
                "role": "tool",
                "name": r["name"],
                "content": json.dumps(r["result"]),
            })

    return final or "(no final answer)", format_trace(trace)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    print(f"Loading {args.model}...")
    model, tokenizer = load_model(args.model)
    print("Model ready.")

    def respond(query: str):
        if not query.strip():
            return "Type a question.", ""
        return run_query(model, tokenizer, query)

    with gr.Blocks(title="FinAgent demo") as demo:
        gr.Markdown("# 💹 FinAgent — fine-tuned Mistral 7B with tool use\n"
                    "Type a financial question. The agent reasons step-by-step, "
                    "calls the right tools, and synthesizes an answer.")
        with gr.Row():
            query = gr.Textbox(label="Question", placeholder="What is Apple's P/E ratio?", lines=2)
        submit = gr.Button("Ask", variant="primary")
        with gr.Tab("Final answer"):
            answer = gr.Markdown()
        with gr.Tab("Reasoning trace"):
            trace = gr.Markdown()
        gr.Examples(
            examples=[
                "What is Apple's current P/E ratio?",
                "Is the yield curve inverted?",
                "Compare Microsoft and Google on valuation.",
                "Should I put my entire 401k in Tesla calls?",
            ],
            inputs=query,
        )
        submit.click(respond, inputs=query, outputs=[answer, trace])

    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
