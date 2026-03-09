#!/usr/bin/env python3
"""
FinAgent — ReAct loop from scratch (no framework).

This is the pedagogical version. It shows EXACTLY what happens inside an LLM agent:

    LOOP:
    1. Build messages = [system, user, ...]
    2. Send to model → it generates a response
    3. Parse the response:
       a. If tool_calls detected → execute tools → add results to messages → back to 2
       b. If final text (no tool_calls) → print response → END
    4. Safety: max 5 iterations to avoid infinite loops

Usage:
    python scripts/agent_from_scratch.py --model results/finagent-7b-merged --query "What is Apple's P/E ratio?"
    python scripts/agent_from_scratch.py --model results/finagent-7b-merged   # interactive mode
"""

import json
import sys
from pathlib import Path

import torch.cuda

# Add project root to path so we can import tools and configs
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from tools import TOOL_REGISTRY
from configs.prompt_templates import SYSTEM_PROMPT
from unsloth import FastLanguageModel

model_device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================================
# STEP 1: load_model
# =====================================================================
# Load the fine-tuned merged model from disk using Unsloth.
#
# Input:  load_model("results/finagent-7b-merged")
# Output: (model, tokenizer)  — ready for inference
#
# Steps:
#   1. from unsloth import FastLanguageModel
#   2. model, tokenizer = FastLanguageModel.from_pretrained(
#          model_name=path, max_seq_length=4096,
#          load_in_4bit=True, dtype=None)
#   3. FastLanguageModel.for_inference(model)  — switch to inference mode
#   4. return model, tokenizer
#
# Note: load_in_4bit=True loads the model quantized (~4 GB instead of ~14 GB)
# =====================================================================

def load_model(path: str):
    # TODO: implement
    model , tokenizer = FastLanguageModel.from_pretrained(model_name=path,
                                                          device=model_device,
                                                          max_seq_length=4096,
                                                          load_in_4bit=True,
                                                          dtype=None)
    FastLanguageModel.for_inference(model)
    return model,tokenizer


# =====================================================================
# STEP 2: generate
# =====================================================================
# Send messages to the model and get a response string.
#
# Input:  generate(model, tokenizer, [
#             {"role": "system", "content": "You are FinAgent..."},
#             {"role": "user", "content": "What is AAPL's P/E?"}
#         ])
# Output: "<think>\nI need to look up AAPL's P/E ratio...\n</think>\n\n
#          [TOOL_CALLS] [{\"name\": \"get_financial_ratios\", ...}]"
#
# Steps:
#   1. tokenizer.apply_chat_template(messages, tokenize=True,
#          add_generation_prompt=True, return_tensors="pt")
#      → this converts messages to Mistral's token format and adds
#        the assistant turn prefix so the model starts generating
#   2. model.generate(input_ids=inputs, max_new_tokens=1024,
#          temperature=0.7, top_p=0.9, do_sample=True)
#   3. Decode only the NEW tokens (skip the prompt):
#      tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=False)
#      → skip_special_tokens=False because we need to see [TOOL_CALLS] tokens
#   4. Return the decoded string
# =====================================================================

def generate(model, tokenizer, messages: list[dict]) -> str:
    # TODO: implement
    chat_template = tokenizer.apply_chat_generation_template(messages,tokenize=True,add_gemeration_prompt=True,return_tensors="pt").to(device=model_device)
    response = model.generate(chat_template,
                   max_new_tokens=1024,
                   temperature=0.7,
                   top_p=0.9,
                   do_sample=True)
    decoded = tokenizer.decode(response[0][chat_template.shape[1]:], skip_special_tokens=False)
    return decoded


# =====================================================================
# STEP 3: parse_tool_calls
# =====================================================================
# Extract tool calls from the model's response text.
#
# Mistral v0.3 format for tool calls:
#   "[TOOL_CALLS] [{"name": "get_stock_quote", "arguments": {"ticker": "AAPL"}}]"
#
# Input:  parse_tool_calls("<think>...</think>\n[TOOL_CALLS] [{\"name\": \"get_stock_quote\", \"arguments\": {\"ticker\": \"AAPL\"}}]")
# Output: [{"name": "get_stock_quote", "arguments": {"ticker": "AAPL"}}]
#
# Input:  parse_tool_calls("Apple's P/E ratio is 28.5...")  # no tool calls
# Output: []
#
# Steps:
#   1. Look for "[TOOL_CALLS]" in the text
#   2. If not found → return []
#   3. If found → extract everything after "[TOOL_CALLS]"
#   4. Parse the JSON array: json.loads(...)
#   5. If arguments is a string (not dict), parse it: json.loads(tc["arguments"])
#   6. Return list of tool call dicts
#   7. Wrap in try/except — if JSON parsing fails, return []
#
# Tip: the model sometimes generates tool_calls in other formats.
#   Also check for '"tool_calls"' in the text as a fallback.
# =====================================================================

def parse_tool_calls(text: str) -> list[dict]:
    # TODO: implement
    pass


# =====================================================================
# STEP 4: execute_tools
# =====================================================================
# Run the actual tool functions and return results.
#
# Input:  execute_tools([
#             {"name": "get_stock_quote", "arguments": {"ticker": "AAPL"}},
#             {"name": "get_financial_ratios", "arguments": {"ticker": "AAPL"}}
#         ])
# Output: [
#             {"name": "get_stock_quote", "result": {"price": 264.72, ...}},
#             {"name": "get_financial_ratios", "result": {"ratios": {...}, ...}}
#         ]
#
# Steps:
#   1. For each tool_call in the list:
#      a. Get the function from TOOL_REGISTRY[tool_call["name"]]
#      b. Call it with **tool_call["arguments"]
#      c. Collect {"name": ..., "result": ...}
#   2. If tool name not in TOOL_REGISTRY → result = {"error": "Unknown tool: ..."}
#   3. If function raises → result = {"error": str(e)}
#   4. Print pedagogical logs:
#      "[Iter N] Executing get_stock_quote(ticker='AAPL') → {"price": 264.72, ...}"
# =====================================================================

def execute_tools(tool_calls: list[dict]) -> list[dict]:
    # TODO: implement
    pass


# =====================================================================
# STEP 5: react_loop
# =====================================================================
# Orchestrate the full ReAct (Reasoning + Acting) loop.
# THIS IS THE CORE OF AN LLM AGENT.
#
# Input:  react_loop(model, tokenizer, "What is Apple's current P/E ratio?")
# Output: "Apple's trailing P/E ratio is 28.5, based on..."  (final text response)
#
# The loop:
#   messages = [
#       {"role": "system", "content": SYSTEM_PROMPT},
#       {"role": "user", "content": user_query}
#   ]
#
#   for i in range(max_iterations):
#       1. response = generate(model, tokenizer, messages)
#       2. tool_calls = parse_tool_calls(response)
#       3. if no tool_calls → return response (DONE)
#       4. if tool_calls:
#          a. Add assistant message with tool_calls to messages
#          b. results = execute_tools(tool_calls)
#          c. For each result, add a tool response message to messages:
#             {"role": "tool", "name": "get_stock_quote",
#              "content": json.dumps(result)}
#          d. Continue loop (model will see tool results and respond)
#
#   If max_iterations reached → return last response + warning
#
# Pedagogical logs at each step:
#   [Iter 1] Sending 2 messages to model...
#   [Iter 1] Model wants 2 tools: get_stock_quote, get_financial_ratios
#   [Iter 1] Executing get_stock_quote(ticker="AAPL") → {"price": 264.72}
#   [Iter 2] Sending 6 messages (added 2 tool results)...
#   [Iter 2] Final response (no more tool calls)
# =====================================================================

def react_loop(model, tokenizer, user_query: str, max_iterations: int = 5) -> str:
    # TODO: implement
    pass


# =====================================================================
# Config — change these defaults, no CLI args needed
# =====================================================================

MODEL_PATH = "danab17/finagent-7b-merged"       # HuggingFace ID or local path
DEFAULT_QUERY = "What is Apple's current P/E ratio?"  # set to None for interactive mode
MAX_ITERATIONS = 5


# =====================================================================
# STEP 6: main
# =====================================================================
# Just run: python agent_from_scratch.py
# - If DEFAULT_QUERY is set → runs that query and exits
# - If DEFAULT_QUERY is None → interactive mode (type questions, Ctrl+C to quit)
# =====================================================================

def main():
    print("=" * 60)
    print("  FinAgent — From Scratch (no framework)")
    print("=" * 60)

    print("\n[Setup] Loading model...")
    model, tokenizer = load_model(MODEL_PATH)
    print("[Setup] Model loaded.\n")

    if DEFAULT_QUERY:
        print(f"Query: {DEFAULT_QUERY}\n")
        response = react_loop(model, tokenizer, DEFAULT_QUERY, MAX_ITERATIONS)
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
                response = react_loop(model, tokenizer, query, MAX_ITERATIONS)
                print(f"\nFinAgent: {response}\n")
            except KeyboardInterrupt:
                print("\nBye!")
                break


if __name__ == "__main__":
    main()
