#!/usr/bin/env python3
"""
FinAgent — Same agent using LangGraph.

Same behavior as agent_from_scratch.py, but LangGraph handles automatically:
  - The STATE (list of messages)
  - The ROUTING (tool_calls → execute → loop back)
  - The STOP CONDITION (no more tool_calls → end)

Graph:
    START → generate → should_continue?
                          ├── has tool_calls → execute_tools → generate (loop)
                          └── no tool_calls  → END

Compare with from_scratch: here LangGraph does automatically what we coded
manually in react_loop(). The agent logic is the same — only the orchestration differs.
"""

import json
import operator
import sys
from pathlib import Path
from typing import TypedDict, Annotated

import torch
from unsloth import FastLanguageModel
from langgraph.graph import StateGraph, END

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from tools import TOOL_REGISTRY
from configs.prompt_templates import SYSTEM_PROMPT

model_device = "cuda" if torch.cuda.is_available() else "cpu"


# =====================================================================
# STEP 1: State definition
# =====================================================================
# LangGraph needs a typed state object. Ours is just a list of messages.
# In from_scratch, we managed this list manually. Here LangGraph tracks it.
# The operator.add annotation tells LangGraph to APPEND new messages.
# =====================================================================

class AgentState(TypedDict):
    messages: Annotated[list[dict], operator.add]


# =====================================================================
# STEP 2: load_model (same as from_scratch)
# =====================================================================

def load_model(path: str):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=path,
        max_seq_length=4096,
        load_in_4bit=True,
        dtype=None,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


# =====================================================================
# STEP 3: generate — shared helper (same as from_scratch)
# =====================================================================

def generate(model, tokenizer, messages: list[dict]) -> str:
    """Send messages to the model and return the raw response string."""
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )

    # Decode only new tokens (skip the prompt)
    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=False)
    return response


# =====================================================================
# STEP 4: parse_tool_calls — shared helper (same as from_scratch)
# =====================================================================

def parse_tool_calls(text: str) -> list[dict]:
    """Extract tool calls from Mistral's output.

    Mistral v0.3 format: [TOOL_CALLS] [{"name": "...", "arguments": {...}}]
    """
    # Check for Mistral's [TOOL_CALLS] marker
    marker = "[TOOL_CALLS]"
    idx = text.find(marker)

    if idx == -1:
        # Fallback: check for tool_calls in JSON-like format
        if '"tool_calls"' not in text:
            return []
        # Try to extract from a JSON object containing tool_calls
        try:
            start = text.index('"tool_calls"')
            # Find the enclosing object
            brace_start = text.rfind("{", 0, start)
            brace_end = text.find("]", start) + 1
            # Find the closing brace after the array
            brace_end = text.find("}", brace_end) + 1
            obj = json.loads(text[brace_start:brace_end])
            calls = obj.get("tool_calls", [])
        except (ValueError, json.JSONDecodeError):
            return []
    else:
        # Standard Mistral format: everything after [TOOL_CALLS] is a JSON array
        json_str = text[idx + len(marker):].strip()
        # Trim anything after the closing bracket (model might continue generating)
        bracket_depth = 0
        end_pos = 0
        for i, ch in enumerate(json_str):
            if ch == "[":
                bracket_depth += 1
            elif ch == "]":
                bracket_depth -= 1
                if bracket_depth == 0:
                    end_pos = i + 1
                    break
        if end_pos == 0:
            return []
        json_str = json_str[:end_pos]

        try:
            calls = json.loads(json_str)
        except json.JSONDecodeError:
            return []

    # Normalize: ensure arguments is a dict (sometimes it's a JSON string)
    for tc in calls:
        if isinstance(tc.get("arguments"), str):
            try:
                tc["arguments"] = json.loads(tc["arguments"])
            except json.JSONDecodeError:
                tc["arguments"] = {}

    return calls


# =====================================================================
# STEP 5: make_generate_node
# =====================================================================
# LangGraph node that calls the model.
# In from_scratch: response = generate(model, tokenizer, messages)
# Here: LangGraph calls this function and manages state for us.
# =====================================================================

def make_generate_node(model, tokenizer):
    """Returns a generate node function with model/tokenizer in closure."""

    def generate_node(state: AgentState) -> dict:
        messages = state["messages"]
        print(f"  [generate] Sending {len(messages)} messages to model...")

        # Call the model — same generate() as from_scratch
        response_text = generate(model, tokenizer, messages)

        # Parse tool calls from the response
        tool_calls = parse_tool_calls(response_text)

        if tool_calls:
            tool_names = [tc["name"] for tc in tool_calls]
            print(f"  [generate] Model wants {len(tool_calls)} tool(s): {', '.join(tool_names)}")
        else:
            print(f"  [generate] Final response (no tool calls)")

        # Build assistant message
        # We store tool_calls in the message so should_continue() and
        # execute_tools_node() can access them — same as from_scratch
        # where we checked parse_tool_calls() output.
        assistant_msg = {
            "role": "assistant",
            "content": response_text,
        }
        if tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": f"call_{i:03d}",
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["arguments"]),
                    },
                }
                for i, tc in enumerate(tool_calls)
            ]

        # Return state update — LangGraph appends this to messages automatically
        return {"messages": [assistant_msg]}

    return generate_node


# =====================================================================
# STEP 6: execute_tools_node
# =====================================================================
# LangGraph node that executes tool calls.
# In from_scratch: results = execute_tools(tool_calls) inside react_loop()
# Here: LangGraph routes here automatically when should_continue() says so.
# =====================================================================

def execute_tools_node(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    tool_calls = last_message.get("tool_calls", [])

    tool_messages = []
    for tc in tool_calls:
        func_info = tc["function"]
        name = func_info["name"]
        arguments = json.loads(func_info["arguments"]) if isinstance(func_info["arguments"], str) else func_info["arguments"]

        # Dispatch to the real tool — same TOOL_REGISTRY as from_scratch
        if name in TOOL_REGISTRY:
            try:
                result = TOOL_REGISTRY[name](**arguments)
                print(f"  [tool] {name}({arguments}) → OK")
            except Exception as e:
                result = {"error": str(e)}
                print(f"  [tool] {name}({arguments}) → ERROR: {e}")
        else:
            result = {"error": f"Unknown tool: {name}"}
            print(f"  [tool] Unknown tool: {name}")

        # Build tool response message — Mistral expects this format
        tool_messages.append({
            "role": "tool",
            "tool_call_id": tc["id"],
            "name": name,
            "content": json.dumps(result),
        })

    # Return state update — LangGraph appends these to messages
    return {"messages": tool_messages}


# =====================================================================
# STEP 7: should_continue
# =====================================================================
# Routing function: tool_calls present → loop, otherwise → end.
# In from_scratch: this was `if tool_calls:` inside react_loop().
# Here: LangGraph uses this to decide the graph edge.
# =====================================================================

def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if last_message.get("tool_calls"):
        return "execute_tools"
    return "end"


# =====================================================================
# STEP 8: build_graph
# =====================================================================
# Assemble the LangGraph graph. This replaces react_loop() entirely.
#
# Graph: START → generate → should_continue? → execute_tools → generate (loop)
#                                            → END
# =====================================================================

def build_graph(model, tokenizer):
    graph = StateGraph(AgentState)

    # Add nodes (= the functions that do work)
    graph.add_node("generate", make_generate_node(model, tokenizer))
    graph.add_node("execute_tools", execute_tools_node)

    # Set entry point
    graph.set_entry_point("generate")

    # Add conditional edge: after generate, check should_continue
    graph.add_conditional_edges(
        "generate",
        should_continue,
        {"execute_tools": "execute_tools", "end": END},
    )

    # After executing tools, always go back to generate (the loop)
    graph.add_edge("execute_tools", "generate")

    return graph.compile()


# =====================================================================
# STEP 9: run_agent
# =====================================================================

def run_agent(graph, query: str) -> str:
    """Run the graph on a user query and return the final response."""
    initial_state = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
    }

    print(f"\n{'='*60}")
    print(f"  Query: {query}")
    print(f"{'='*60}")

    # LangGraph runs the full ReAct loop automatically
    # In from_scratch, we wrote the for-loop ourselves in react_loop()
    result = graph.invoke(initial_state)

    # Extract final assistant message
    final_message = result["messages"][-1]
    return final_message.get("content", "")


# =====================================================================
# Config — change these defaults, no CLI args needed
# =====================================================================

MODEL_PATH = "danab17/finagent-7b-merged"
DEFAULT_QUERY = "What is Apple's current P/E ratio?"  # set to None for interactive mode


# =====================================================================
# main
# =====================================================================

def main():
    print("=" * 60)
    print("  FinAgent — LangGraph")
    print("=" * 60)

    print("\n[Setup] Loading model...")
    model, tokenizer = load_model(MODEL_PATH)
    print("[Setup] Model loaded.")

    # Build graph — in from_scratch we had react_loop(), here the graph IS the loop
    print("[Setup] Building LangGraph agent...")
    graph = build_graph(model, tokenizer)
    print("[Setup] Graph ready.\n")

    if DEFAULT_QUERY:
        response = run_agent(graph, DEFAULT_QUERY)
        print("\n" + "=" * 60)
        print("FINAL ANSWER:")
        print("=" * 60)
        print(response)
    else:
        print("Interactive mode. Type your question (Ctrl+C to quit).\n")
        while True:
            try:
                query = input("You: ").strip()
                if not query:
                    continue
                response = run_agent(graph, query)
                print(f"\nFinAgent: {response}\n")
            except KeyboardInterrupt:
                print("\nBye!")
                break


if __name__ == "__main__":
    main()
