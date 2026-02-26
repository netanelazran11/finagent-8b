"""
FinAgent Synthetic Data Generation Pipeline
============================================

Generates three types of training data using GPT-4o-mini (~$0.50 total).

Usage:
  conda activate finagent
  python -u scripts/generate_dataset.py
"""

import json
import os
import sys
import time
from pathlib import Path
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.prompt_templates import (
    SYSTEM_PROMPT,
    COT_GENERATION_PROMPT,
    TOOL_TRAJECTORY_PROMPT,
    GUARDRAIL_GENERATION_PROMPT,
)

# ---------------------------------------------------------------------------
# Single LLM Client — GPT-4o-mini for everything (~$0.50 total)
# ---------------------------------------------------------------------------
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL = "gpt-4o-mini"


def generate(prompt, temperature=0.7, max_tokens=2048, retries=3):
    """Call GPT-4o-mini with retry on rate limit."""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate_limit" in error_str:
                wait = 30 * (attempt + 1)
                print(f"    [RATE LIMIT] Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    [ERROR] {e}")
                return None
    return None


# ===========================================================================
# Load and classify seeds + resume support
# ===========================================================================
def load_seeds():
    curated_path = PROJECT_ROOT / "data" / "seeds" / "expanded_curated.json"
    original_path = PROJECT_ROOT / "data" / "seeds" / "financial_scenarios.json"

    seeds_path = curated_path if curated_path.exists() else original_path
    print(f"Using seeds: {seeds_path}")

    with open(seeds_path) as f:
        seeds = json.load(f)

    classified = {"cot": [], "tool": [], "guardrail": []}
    for seed in seeds:
        if seed["type"] == "guardrail":
            classified["guardrail"].append(seed)
        elif seed.get("tools_expected"):
            classified["tool"].append(seed)
        else:
            classified["cot"].append(seed)

    print(f"Seeds — CoT: {len(classified['cot'])}, "
          f"Tool: {len(classified['tool'])}, "
          f"Guardrail: {len(classified['guardrail'])}")
    return classified


def load_existing_seed_ids(jsonl_path):
    """Load seed IDs already generated — for resume support."""
    ids = set()
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    seed_id = data.get("metadata", {}).get("seed_id")
                    if seed_id:
                        ids.add(seed_id)
    return ids


# ===========================================================================
# Format helpers
# ===========================================================================
def format_cot_or_guardrail(query, generation):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
        {"role": "assistant", "content": generation.strip()},
    ]


def format_tool_trajectory(query, raw_generation):
    cleaned = raw_generation.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]
        cleaned = cleaned.rsplit("```", 1)[0]

    trajectory = json.loads(cleaned)
    if not isinstance(trajectory, list):
        raise TypeError(f"Expected list, got {type(trajectory)}")

    for msg in trajectory:
        if msg["role"] not in ("assistant", "tool"):
            raise ValueError(f"Unexpected role: {msg['role']}")

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ] + trajectory


# ===========================================================================
# Quality validation
# ===========================================================================
VALID_TOOL_NAMES = {
    "get_stock_quote", "get_financial_ratios", "search_financial_news",
    "get_market_overview", "get_treasury_yields", "screen_stocks",
    "get_economic_indicators",
}


def validate_quality(messages, gen_type):
    score = 0
    assistant_msgs = [m for m in messages if m["role"] == "assistant"]

    if any("<think>" in m.get("content", "") for m in assistant_msgs):
        score += 1

    if gen_type == "tool":
        call_ids, response_ids, tool_names = set(), set(), set()
        for msg in messages:
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    call_ids.add(tc["id"])
                    tool_names.add(tc["function"]["name"])
            if msg["role"] == "tool":
                response_ids.add(msg["tool_call_id"])
        if call_ids == response_ids and len(call_ids) > 0:
            score += 1
        if tool_names.issubset(VALID_TOOL_NAMES):
            score += 1

    total_len = sum(len(m.get("content", "")) for m in assistant_msgs)
    if total_len > 200:
        score += 1

    return score


# ===========================================================================
# Generation — all using GPT-4o-mini
# ===========================================================================
def generate_cot_examples(seeds, output_path):
    """Type A: Chain-of-Thought. Appends to file for resume support."""
    existing_ids = load_existing_seed_ids(output_path)
    seeds_todo = [s for s in seeds if s["id"] not in existing_ids]

    if existing_ids:
        print(f"  Resuming: {len(existing_ids)} already done, {len(seeds_todo)} remaining")

    count = 0
    with open(output_path, "a") as f:
        for i, seed in enumerate(seeds_todo):
            prompt = COT_GENERATION_PROMPT.format(
                query=seed["query"],
                reasoning_axes=", ".join(seed["reasoning_axes"]),
            )
            print(f"  [{i+1}/{len(seeds_todo)}] {seed['id']}: {seed['query'][:60]}...")

            raw = generate(prompt)
            if not raw:
                continue

            messages = format_cot_or_guardrail(seed["query"], raw)
            score = validate_quality(messages, "cot")

            if score >= 2:
                record = {
                    "messages": messages,
                    "metadata": {"seed_id": seed["id"], "generation_type": "cot", "quality_score": score},
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                count += 1

    total = len(existing_ids) + count
    print(f"  → {total} total CoT examples ({count} new)")
    return total


def generate_tool_examples(seeds, output_path):
    """Type B: Tool trajectories. 3 candidates per seed, keep all valid."""
    existing_ids = load_existing_seed_ids(output_path)
    seeds_todo = [s for s in seeds if s["id"] not in existing_ids]

    if existing_ids:
        print(f"  Resuming: {len(existing_ids)} seeds already done, {len(seeds_todo)} remaining")

    tool_defs_path = PROJECT_ROOT / "configs" / "tool_definitions.json"
    with open(tool_defs_path) as f:
        all_tools = json.load(f)

    count = 0
    with open(output_path, "a") as f:
        for i, seed in enumerate(seeds_todo):
            relevant_tools = [t for t in all_tools if t["function"]["name"] in seed["tools_expected"]]

            prompt = TOOL_TRAJECTORY_PROMPT.format(
                query=seed["query"],
                tool_definitions=json.dumps(relevant_tools, indent=2),
                tools_expected=", ".join(seed["tools_expected"]),
                reasoning_axes=", ".join(seed["reasoning_axes"]),
            )
            print(f"  [{i+1}/{len(seeds_todo)}] {seed['id']}: {seed['query'][:60]}...")

            for gen_idx in range(3):
                raw = generate(prompt, temperature=0.5)
                if not raw:
                    continue

                try:
                    messages = format_tool_trajectory(seed["query"], raw)
                    score = validate_quality(messages, "tool")
                    if score >= 2:
                        record = {
                            "messages": messages,
                            "metadata": {"seed_id": seed["id"], "generation_type": "tool", "quality_score": score},
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        f.flush()
                        count += 1
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                    print(f"    [WARN] gen_{gen_idx} parse failed: {e}")

    total_lines = sum(1 for _ in open(output_path)) if output_path.exists() else 0
    print(f"  → {total_lines} total Tool examples ({count} new)")
    return total_lines


def generate_guardrail_examples(seeds, output_path):
    """Type C: Guardrails. Appends to file for resume support."""
    existing_ids = load_existing_seed_ids(output_path)
    seeds_todo = [s for s in seeds if s["id"] not in existing_ids]

    if existing_ids:
        print(f"  Resuming: {len(existing_ids)} already done, {len(seeds_todo)} remaining")

    count = 0
    with open(output_path, "a") as f:
        for i, seed in enumerate(seeds_todo):
            prompt = GUARDRAIL_GENERATION_PROMPT.format(
                query=seed["query"],
                reasoning_axes=", ".join(seed["reasoning_axes"]),
            )
            print(f"  [{i+1}/{len(seeds_todo)}] {seed['id']}: {seed['query'][:60]}...")

            raw = generate(prompt)
            if not raw:
                continue

            messages = format_cot_or_guardrail(seed["query"], raw)
            score = validate_quality(messages, "guardrail")

            if score >= 1:
                record = {
                    "messages": messages,
                    "metadata": {"seed_id": seed["id"], "generation_type": "guardrail", "quality_score": score},
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                count += 1

    total = len(existing_ids) + count
    print(f"  → {total} total Guardrail examples ({count} new)")
    return total


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("=" * 60)
    print("FinAgent — Synthetic Data Generation (GPT-4o-mini, ~$0.50)")
    print("Supports resume — safe to Ctrl+C and re-run")
    print("=" * 60)

    classified = load_seeds()
    output_dir = PROJECT_ROOT / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Type A: CoT ---
    print(f"\n[A] CoT examples ({len(classified['cot'])} seeds)...")
    n_cot = generate_cot_examples(classified["cot"], output_dir / "cot_examples.jsonl")

    # --- Type B: Tool trajectories ---
    print(f"\n[B] Tool examples ({len(classified['tool'])} seeds × 3 candidates)...")
    n_tool = generate_tool_examples(classified["tool"], output_dir / "tool_examples.jsonl")

    # --- Type C: Guardrails ---
    print(f"\n[C] Guardrail examples ({len(classified['guardrail'])} seeds)...")
    n_guard = generate_guardrail_examples(classified["guardrail"], output_dir / "guardrail_examples.jsonl")

    total = n_cot + n_tool + n_guard
    print(f"\n{'=' * 60}")
    print(f"DONE — {total} total examples")
    print(f"  CoT:       {n_cot}")
    print(f"  Tool:      {n_tool}")
    print(f"  Guardrail: {n_guard}")
    print(f"\nNext: python scripts/prepare_dataset.py")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
