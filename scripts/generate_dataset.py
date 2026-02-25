"""
FinAgent Synthetic Data Generation Pipeline
============================================

This script uses Distilabel to generate the three types of training data:
  - Type A: Chain-of-Thought reasoning (pure reasoning, no tools)
  - Type B: Tool-calling trajectories (ReAct multi-turn)
  - Type C: Guardrail/refusal examples

Hybrid Cost Strategy:
  - Type A (CoT) + Type C (Guardrails) → Groq (FREE, Llama 3.3 70B)
  - Type B (Tool trajectories)         → GPT-4o-mini (~$2-3 total)
  Total cost: ~$2-3 instead of ~$100

Usage:
  export OPENAI_API_KEY=your_key_here
  export GROQ_API_KEY=your_key_here    # Free at https://console.groq.com
  python scripts/generate_dataset.py
"""

import json
import os
import sys
import hashlib
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from distilabel.llms import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts, StepInput, step
from distilabel.steps.tasks import TextGeneration

# Groq import — uses the OpenAI-compatible API
# Distilabel supports Groq natively but we can also use OpenAILLM
# with Groq's OpenAI-compatible endpoint, which is more reliable
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# ---------------------------------------------------------------------------
# Add project root so we can import configs
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.prompt_templates import (
    SYSTEM_PROMPT,
    COT_GENERATION_PROMPT,
    TOOL_TRAJECTORY_PROMPT,
    GUARDRAIL_GENERATION_PROMPT,
)


# ===========================================================================
# STEP 1: Load and classify seeds
# ===========================================================================
def load_seeds() -> dict[str, list[dict]]:
    """
    Load seeds from JSON and split them by generation type.

    Returns three lists:
      - cot_seeds: Type A (no tool use, pure reasoning)
      - tool_seeds: Type B (requires tool-calling trajectories)
      - guardrail_seeds: Type C (refusal/safety examples)

    The classification logic:
      - If seed.type == "guardrail" → Type C
      - If seed.tools_expected is non-empty → Type B
      - Otherwise → Type A
    """
    # Use expanded curated seeds if available, otherwise fall back to originals
    curated_path = PROJECT_ROOT / "data" / "seeds" / "expanded_curated.json"
    original_path = PROJECT_ROOT / "data" / "seeds" / "financial_scenarios.json"

    if curated_path.exists():
        seeds_path = curated_path
        print(f"Using expanded seeds: {curated_path}")
    else:
        seeds_path = original_path
        print(f"Using original seeds (run expand_seeds.py first for full dataset): {original_path}")

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

    print(f"Seeds loaded — CoT: {len(classified['cot'])}, "
          f"Tool: {len(classified['tool'])}, "
          f"Guardrail: {len(classified['guardrail'])}")

    return classified


# ===========================================================================
# STEP 2: Build prompts from seeds
# ===========================================================================
def build_cot_inputs(seeds: list[dict]) -> list[dict]:
    """Build Distilabel input dicts for Type A (CoT) generation."""
    inputs = []
    for seed in seeds:
        prompt = COT_GENERATION_PROMPT.format(
            query=seed["query"],
            reasoning_axes=", ".join(seed["reasoning_axes"]),
        )
        inputs.append({
            "instruction": prompt,
            "seed_id": seed["id"],
            "seed_query": seed["query"],
            "generation_type": "cot",
        })
    return inputs


def build_tool_inputs(seeds: list[dict]) -> list[dict]:
    """Build Distilabel input dicts for Type B (tool trajectory) generation."""
    tool_defs_path = PROJECT_ROOT / "configs" / "tool_definitions.json"
    with open(tool_defs_path) as f:
        all_tools = json.load(f)

    # Filter tool definitions to only those expected by the seed
    inputs = []
    for seed in seeds:
        relevant_tools = [
            t for t in all_tools
            if t["function"]["name"] in seed["tools_expected"]
        ]
        prompt = TOOL_TRAJECTORY_PROMPT.format(
            query=seed["query"],
            tool_definitions=json.dumps(relevant_tools, indent=2),
            tools_expected=", ".join(seed["tools_expected"]),
            reasoning_axes=", ".join(seed["reasoning_axes"]),
        )
        inputs.append({
            "instruction": prompt,
            "seed_id": seed["id"],
            "seed_query": seed["query"],
            "generation_type": "tool",
        })
    return inputs


def build_guardrail_inputs(seeds: list[dict]) -> list[dict]:
    """Build Distilabel input dicts for Type C (guardrail) generation."""
    inputs = []
    for seed in seeds:
        prompt = GUARDRAIL_GENERATION_PROMPT.format(
            query=seed["query"],
            reasoning_axes=", ".join(seed["reasoning_axes"]),
        )
        inputs.append({
            "instruction": prompt,
            "seed_id": seed["id"],
            "seed_query": seed["query"],
            "generation_type": "guardrail",
        })
    return inputs


# ===========================================================================
# STEP 3: Post-processing — convert raw generations to chat-ml JSONL
# ===========================================================================
@step(inputs=["seed_query", "generation_type", "generation"], outputs=["messages"])
def FormatToChatML(instructions: StepInput):
    """
    Convert raw teacher generations into the final chat-ml message format.

    This step handles:
      - Type A/C: Wraps the generation in system + user + assistant messages
      - Type B: Parses the JSON trajectory and wraps with system + user messages

    Architecture Note:
    -----------------
    With num_generations=3, each input produces a LIST of generations.
    We "explode" them: one input seed → N output records (one per valid generation).
    This is the keep-all-valid strategy — every generation that passes
    validation becomes a separate training example, maximizing dataset
    diversity from the same set of seeds.

    Distilabel stores multiple generations as a list in the "generation"
    field. We iterate over each candidate and produce a separate record.
    """
    exploded = []

    for item in instructions:
        gen_type = item["generation_type"]
        query = item["seed_query"]
        raw_generations = item["generation"]

        # Handle both single generation (str) and multi-generation (list)
        if isinstance(raw_generations, str):
            raw_generations = [raw_generations]

        for i, raw in enumerate(raw_generations):
            record = {
                **item,
                "generation_idx": i,
            }
            try:
                if gen_type == "tool":
                    messages = _format_tool_trajectory(query, raw)
                else:
                    messages = _format_cot_or_guardrail(query, raw)

                record["messages"] = messages
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"  [WARN] Failed to parse {item.get('seed_id', '?')} gen_{i}: {e}")
                record["messages"] = None

            exploded.append(record)

    yield exploded


def _format_cot_or_guardrail(query: str, generation: str) -> list[dict]:
    """Wrap a CoT or guardrail generation in the chat-ml format."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
        {"role": "assistant", "content": generation.strip()},
    ]


def _format_tool_trajectory(query: str, raw_generation: str) -> list[dict]:
    """
    Parse a tool trajectory JSON and prepend system + user messages.

    The teacher model generates a JSON array of assistant/tool turns.
    We validate the structure and prepend the system prompt + user query.
    """
    # The teacher sometimes wraps JSON in markdown code fences — strip them
    cleaned = raw_generation.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]  # remove first line (```json)
        cleaned = cleaned.rsplit("```", 1)[0]  # remove last fence

    trajectory = json.loads(cleaned)

    if not isinstance(trajectory, list):
        raise TypeError(f"Expected list, got {type(trajectory)}")

    # Validate the trajectory structure
    for msg in trajectory:
        if msg["role"] not in ("assistant", "tool"):
            raise ValueError(f"Unexpected role in trajectory: {msg['role']}")

    # Prepend system + user, then append the trajectory
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ] + trajectory

    return messages


# ===========================================================================
# STEP 4: Quality validation
# ===========================================================================
@step(inputs=["messages", "generation_type"], outputs=["messages", "quality_score"])
def ValidateQuality(instructions: StepInput):
    """
    Validate generated examples meet our quality bar.

    Checks:
      1. messages is not None (parsing succeeded)
      2. Assistant responses contain <think> blocks
      3. Tool trajectories have matching call_id ↔ response pairs
      4. Minimum length thresholds (filters out degenerate short responses)
      5. No hallucinated tool names (only our defined tools)
    """
    valid_tool_names = {
        "get_stock_quote", "get_financial_ratios", "search_financial_news",
        "get_market_overview", "get_treasury_yields", "screen_stocks",
        "get_economic_indicators",
    }

    for item in instructions:
        score = 0
        messages = item["messages"]

        # Check 1: Parsing succeeded
        if messages is None:
            item["quality_score"] = 0
            continue

        # Check 2: Has <think> block in at least one assistant message
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        has_think = any("<think>" in m.get("content", "") for m in assistant_msgs)
        if has_think:
            score += 1

        # Check 3: Tool call/response consistency
        if item["generation_type"] == "tool":
            call_ids = set()
            response_ids = set()
            tool_names_used = set()

            for msg in messages:
                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        call_ids.add(tc["id"])
                        tool_names_used.add(tc["function"]["name"])
                if msg["role"] == "tool":
                    response_ids.add(msg["tool_call_id"])

            # Every call must have a response
            if call_ids == response_ids and len(call_ids) > 0:
                score += 1

            # No hallucinated tool names
            if tool_names_used.issubset(valid_tool_names):
                score += 1
            else:
                print(f"  [WARN] Invalid tools: {tool_names_used - valid_tool_names}")

        # Check 4: Minimum content length
        total_assistant_content = sum(
            len(m.get("content", "")) for m in assistant_msgs
        )
        if total_assistant_content > 200:
            score += 1

        item["quality_score"] = score

    yield instructions


# ===========================================================================
# STEP 5: Assemble and run the pipeline
# ===========================================================================
def _make_llm(provider: str, model_name: str, temperature: float):
    """
    Factory for creating the right LLM backend.

    Two providers:
      - "openai"  → OpenAI API (GPT-4o-mini). Used for tool trajectories
                     because structured JSON output is critical.
      - "groq"    → Groq API (Llama 3.3 70B, FREE tier). Used for CoT and
                     guardrails where the output is plain text, not JSON.

    Why Groq works via OpenAILLM:
      Groq exposes an OpenAI-compatible API. Distilabel's OpenAILLM accepts
      a base_url parameter, so we just point it to Groq's endpoint.
      This means zero extra dependencies — same code, different endpoint.
    """
    if provider == "groq":
        import os
        return OpenAILLM(
            model=model_name,
            base_url=GROQ_BASE_URL,
            api_key=os.environ["GROQ_API_KEY"],
            generation_kwargs={
                "temperature": temperature,
                "max_tokens": 2048,
            },
        )
    else:
        return OpenAILLM(
            model=model_name,
            generation_kwargs={
                "temperature": temperature,
                "max_new_tokens": 2048,
            },
        )


def create_pipeline(
    inputs: list[dict],
    pipeline_name: str,
    provider: str = "openai",
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.7,
    num_generations: int = 3,
) -> Pipeline:
    """
    Create a Distilabel pipeline for a batch of inputs.

    Architecture:
      LoadData → TextGeneration (teacher LLM) → FormatToChatML → ValidateQuality

    Parameters:
    ----------
    provider : str
        "openai" for GPT-4o-mini (paid, ~$2-3 for tool trajectories)
        "groq" for Llama 3.3 70B (free, for CoT and guardrails)
    temperature : float
        0.7 is the sweet spot for synthetic data:
        - Too low (0.1-0.3): Repetitive, cookie-cutter responses
        - Too high (0.9-1.0): Hallucinated data, broken JSON in tool calls
        - 0.7: Diverse but coherent, valid JSON structure maintained
    num_generations : int
        Candidates per seed. 3 = keep-all-valid strategy.
        For Groq free tier, use 1 to stay within rate limits.
    """
    with Pipeline(name=pipeline_name) as pipeline:
        load_data = LoadDataFromDicts(data=inputs)

        generate = TextGeneration(
            llm=_make_llm(provider, model_name, temperature),
            num_generations=num_generations,
        )

        format_step = FormatToChatML()
        validate_step = ValidateQuality()

        # Chain the steps: load → generate → format → validate
        load_data >> generate >> format_step >> validate_step

    return pipeline


def save_results(results, output_path: Path, min_quality: int = 2):
    """
    Save validated results as JSONL, filtering by quality score.

    Each line is a complete training example with the "messages" field
    ready for consumption by Unsloth/TRL in Module 2.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0

    with open(output_path, "w") as f:
        for batch in results:
            for item in batch:
                total += 1
                if item.get("messages") and item.get("quality_score", 0) >= min_quality:
                    record = {
                        "messages": item["messages"],
                        "metadata": {
                            "seed_id": item.get("seed_id"),
                            "generation_type": item.get("generation_type"),
                            "quality_score": item.get("quality_score"),
                        },
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    kept += 1

    print(f"Saved {kept}/{total} examples to {output_path}")
    return kept


# ===========================================================================
# MAIN — Orchestrate all three pipelines
# ===========================================================================
def main():
    print("=" * 60)
    print("FinAgent — Synthetic Data Generation Pipeline")
    print("=" * 60)
    print("Cost strategy: Groq (FREE) for CoT/Guardrails, GPT-4o-mini (~$2-3) for Tools")
    print("=" * 60)

    # Load and classify seeds
    classified = load_seeds()

    output_dir = PROJECT_ROOT / "data" / "raw"
    total_kept = 0

    # --- Pipeline A: Chain-of-Thought → Groq FREE ---
    if classified["cot"]:
        print("\n[Pipeline A] Generating Chain-of-Thought examples (Groq — FREE)...")
        cot_inputs = build_cot_inputs(classified["cot"])
        pipeline_a = create_pipeline(
            cot_inputs,
            "finagent-cot",
            provider="groq",
            model_name="llama-3.3-70b-versatile",
            temperature=0.7,
            num_generations=1,  # 1 per call due to Groq rate limits, run script 3x for diversity
        )
        results_a = pipeline_a.run(use_cache=False)
        total_kept += save_results(
            results_a["ValidateQuality_0"].values(),
            output_dir / "cot_examples.jsonl",
        )

    # --- Pipeline B: Tool Trajectories → GPT-4o-mini (paid, ~$2-3) ---
    if classified["tool"]:
        print("\n[Pipeline B] Generating Tool Trajectory examples (GPT-4o-mini)...")
        tool_inputs = build_tool_inputs(classified["tool"])
        pipeline_b = create_pipeline(
            tool_inputs,
            "finagent-tools",
            provider="openai",              # Paid — JSON quality matters here
            model_name="gpt-4o-mini",
            temperature=0.5,                # Lower temp for valid JSON structure
            num_generations=3,              # 3 candidates, keep all valid
        )
        results_b = pipeline_b.run(use_cache=False)
        total_kept += save_results(
            results_b["ValidateQuality_0"].values(),
            output_dir / "tool_examples.jsonl",
        )

    # --- Pipeline C: Guardrails → Groq FREE ---
    if classified["guardrail"]:
        print("\n[Pipeline C] Generating Guardrail examples (Groq — FREE)...")
        guard_inputs = build_guardrail_inputs(classified["guardrail"])
        pipeline_c = create_pipeline(
            guard_inputs,
            "finagent-guardrails",
            provider="groq",
            model_name="llama-3.3-70b-versatile",
            temperature=0.7,
            num_generations=1,
        )
        results_c = pipeline_c.run(use_cache=False)
        total_kept += save_results(
            results_c["ValidateQuality_0"].values(),
            output_dir / "guardrail_examples.jsonl",
        )

    print(f"\n{'=' * 60}")
    print(f"Total examples generated: {total_kept}")
    print(f"Output directory: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
