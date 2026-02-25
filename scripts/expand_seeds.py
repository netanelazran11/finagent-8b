"""
Seed Expansion Script
======================

Takes the 12 hand-crafted seeds and uses GPT-4o to generate ~300 diverse
seeds across all query types, asset classes, and complexity levels.

Architecture:
  12 curated seeds → GPT-4o meta-generation → ~300 expanded seeds → human review

Why not hand-craft 300 seeds?
  - 300 seeds would take 10-15 hours of manual work
  - More importantly, human-written seeds suffer from "imagination bottleneck":
    you unconsciously repeat the same patterns, phrasing, and scenarios
  - LLM-generated seeds explore the combinatorial space more uniformly
  - We still validate the output — the LLM drafts, you curate

Production flow:
  1. Run this script → generates candidate seeds to data/seeds/expanded_candidates.json
  2. You manually review: delete bad seeds, edit mediocre ones, keep good ones
  3. Move curated seeds to data/seeds/expanded_curated.json
  4. Run generate_dataset.py with the expanded seed file
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# The combinatorial space we want to cover
# ---------------------------------------------------------------------------
QUERY_TYPES = [
    "comparison",       # "Compare X vs Y"
    "analysis",         # "Analyze the risk/reward of X"
    "screening",        # "Find stocks matching criteria"
    "allocation",       # "How should I allocate between..."
    "explanation",      # "Explain how X affects Y"
    "risk_assessment",  # "What are the risks of..."
    "timing",           # "Is now a good time to..."
    "income_strategy",  # "How do I generate income from..."
    "guardrail",        # Irresponsible/dangerous requests the model should refuse
]

ASSET_CLASSES = [
    "us_equities",
    "international_equities",
    "etfs",
    "fixed_income",
    "commodities",
    "crypto",
    "options",
    "reits",
]

USER_PERSONAS = [
    "college student with $5k to invest for the first time",
    "30-year-old software engineer with $200k portfolio",
    "55-year-old pre-retiree shifting to income focus",
    "day trader looking for short-term momentum plays",
    "risk-averse parent saving for children's college fund",
    "small business owner with irregular income",
]

# Tools available (from tool_definitions.json)
AVAILABLE_TOOLS = [
    "get_stock_quote",
    "get_financial_ratios",
    "search_financial_news",
    "get_market_overview",
    "get_treasury_yields",
    "screen_stocks",
    "get_economic_indicators",
]

# Reasoning axes the model should learn to cover
REASONING_AXES_POOL = [
    "valuation", "growth", "momentum", "volatility", "concentration_risk",
    "diversification", "tax_implications", "inflation_correlation",
    "interest_rate_sensitivity", "currency_risk", "liquidity_risk",
    "sector_rotation", "earnings_quality", "debt_sustainability",
    "dividend_safety", "macro_environment", "geopolitical_risk",
    "behavioral_bias", "time_horizon", "risk_tolerance",
    "sequence_of_returns_risk", "mean_reversion", "correlation_analysis",
    "options_greeks", "yield_curve_analysis", "credit_risk",
]


# ---------------------------------------------------------------------------
# Meta-prompt: instructs GPT-4o to generate seed batches
# ---------------------------------------------------------------------------
SEED_GENERATION_PROMPT = """You are helping build a training dataset for a financial AI assistant.

Generate exactly {batch_size} DIVERSE financial question seeds. Each seed must be a JSON object.

CONSTRAINTS FOR THIS BATCH:
- Query types to cover: {query_types}
- Asset classes to cover: {asset_classes}
- User persona for this batch: {persona}

Each seed must have this exact schema:
{{
  "id": "seed_XXX",         // unique ID, start numbering from {start_id}
  "query": "...",            // the user's financial question (natural, conversational tone)
  "type": "...",             // one of: {all_query_types}
  "asset_class": "...",      // one of: {all_asset_classes}
  "complexity": "...",       // "single_step" (no tools needed) or "multi_tool" (requires tool calls)
  "tools_expected": [...],   // list of tool names from: {tools}. Empty list if complexity is single_step
  "reasoning_axes": [...]    // 2-4 axes from: {axes}
}}

QUALITY RULES:
1. Each query must sound like a REAL person asking a financial question — not a textbook exercise
2. Vary the specificity: some queries name specific tickers, others are general
3. Mix simple questions (explanation, single analysis) with complex ones (multi-asset comparison, screening with filters)
4. At least 2 seeds should be "guardrail" type where the user asks something irresponsible or dangerous
5. tools_expected must ONLY contain tools that are logically needed to answer the query
6. For "multi_tool" complexity, require 2-3 tools maximum

Output ONLY a valid JSON array of seed objects. No commentary."""


def generate_seed_batch(
    client: OpenAI,
    batch_size: int,
    query_types: list[str],
    asset_classes: list[str],
    persona: str,
    start_id: int,
) -> list[dict]:
    """Generate one batch of seeds using GPT-4o."""

    prompt = SEED_GENERATION_PROMPT.format(
        batch_size=batch_size,
        query_types=", ".join(query_types),
        asset_classes=", ".join(asset_classes),
        persona=persona,
        start_id=start_id,
        all_query_types=", ".join(QUERY_TYPES),
        all_asset_classes=", ".join(ASSET_CLASSES),
        tools=", ".join(AVAILABLE_TOOLS),
        axes=", ".join(REASONING_AXES_POOL),
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,  # Higher temp for seed diversity
        max_tokens=4096,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    parsed = json.loads(raw)

    # Handle GPT-4o wrapping array in an object (e.g., {"seeds": [...]})
    if isinstance(parsed, dict):
        for key in ("seeds", "data", "results", "items"):
            if key in parsed and isinstance(parsed[key], list):
                return parsed[key]
        raise ValueError(f"Unexpected JSON structure: {list(parsed.keys())}")

    return parsed


def validate_seed(seed: dict) -> list[str]:
    """Return list of issues with a seed. Empty list = valid."""
    issues = []

    required_fields = ["id", "query", "type", "asset_class", "complexity",
                        "tools_expected", "reasoning_axes"]
    for field in required_fields:
        if field not in seed:
            issues.append(f"missing field: {field}")

    if seed.get("type") not in QUERY_TYPES:
        issues.append(f"invalid type: {seed.get('type')}")

    if seed.get("complexity") not in ("single_step", "multi_tool", "refusal"):
        issues.append(f"invalid complexity: {seed.get('complexity')}")

    # Tool consistency checks
    tools = seed.get("tools_expected", [])
    for t in tools:
        if t not in AVAILABLE_TOOLS:
            issues.append(f"unknown tool: {t}")

    if seed.get("complexity") == "multi_tool" and not tools:
        issues.append("multi_tool complexity but no tools_expected")

    if seed.get("complexity") == "single_step" and tools:
        issues.append("single_step complexity but has tools_expected")

    if len(seed.get("reasoning_axes", [])) < 2:
        issues.append("needs at least 2 reasoning_axes")

    if len(seed.get("query", "")) < 20:
        issues.append("query too short")

    return issues


def main():
    print("=" * 60)
    print("FinAgent — Seed Expansion Pipeline")
    print("=" * 60)

    client = OpenAI()  # Uses OPENAI_API_KEY env var

    all_seeds = []
    seed_counter = 13  # Start after the 12 hand-crafted seeds

    # -----------------------------------------------------------------------
    # Strategy: Generate seeds in focused batches
    #
    # Each batch targets a specific (persona × query_type_subset × asset_class_subset)
    # combination. This ensures COVERAGE of the combinatorial space rather than
    # letting GPT-4o default to "compare AAPL vs MSFT" 200 times.
    #
    # 6 personas × 8 batches × 8 seeds per batch = 384 candidate seeds
    # After validation and dedup: ~280-320 curated seeds
    # -----------------------------------------------------------------------

    batches_config = [
        # (query_types_subset, asset_classes_subset, seeds_per_batch)
        (["comparison", "analysis"],       ["us_equities", "etfs"],       8),
        (["screening", "allocation"],      ["us_equities", "fixed_income"], 8),
        (["explanation", "risk_assessment"], ["fixed_income", "commodities"], 8),
        (["timing", "income_strategy"],    ["etfs", "reits"],              8),
        (["analysis", "comparison"],       ["crypto", "options"],          8),
        (["allocation", "risk_assessment"], ["international_equities", "commodities"], 8),
        (["explanation", "timing"],        ["us_equities", "crypto"],      8),
        (["screening", "income_strategy"], ["reits", "fixed_income"],      8),
    ]

    total_batches = len(USER_PERSONAS) * len(batches_config)
    batch_num = 0

    for persona in USER_PERSONAS:
        for query_types, asset_classes, batch_size in batches_config:
            batch_num += 1
            print(f"\n[Batch {batch_num}/{total_batches}] "
                  f"Persona: {persona[:30]}... | "
                  f"Types: {query_types} | Assets: {asset_classes}")

            try:
                seeds = generate_seed_batch(
                    client=client,
                    batch_size=batch_size,
                    query_types=query_types,
                    asset_classes=asset_classes,
                    persona=persona,
                    start_id=seed_counter,
                )

                # Validate each seed
                valid_count = 0
                for seed in seeds:
                    issues = validate_seed(seed)
                    if issues:
                        print(f"  [SKIP] {seed.get('id', '?')}: {issues}")
                    else:
                        seed["id"] = f"seed_{seed_counter:04d}"
                        seed["source_persona"] = persona
                        all_seeds.append(seed)
                        seed_counter += 1
                        valid_count += 1

                print(f"  → {valid_count}/{len(seeds)} valid")

            except Exception as e:
                print(f"  [ERROR] Batch failed: {e}")
                continue

    # -----------------------------------------------------------------------
    # Save candidates for human review
    # -----------------------------------------------------------------------
    output_path = PROJECT_ROOT / "data" / "seeds" / "expanded_candidates.json"
    with open(output_path, "w") as f:
        json.dump(all_seeds, f, indent=2, ensure_ascii=False)

    # Print coverage stats
    type_dist = {}
    asset_dist = {}
    for s in all_seeds:
        type_dist[s["type"]] = type_dist.get(s["type"], 0) + 1
        asset_dist[s["asset_class"]] = asset_dist.get(s["asset_class"], 0) + 1

    print(f"\n{'=' * 60}")
    print(f"Generated {len(all_seeds)} valid seeds → {output_path}")
    print(f"\nType distribution:")
    for t, c in sorted(type_dist.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}")
    print(f"\nAsset class distribution:")
    for a, c in sorted(asset_dist.items(), key=lambda x: -x[1]):
        print(f"  {a}: {c}")
    print(f"{'=' * 60}")

    print(f"\nNEXT STEP: Review {output_path}")
    print("  Delete low-quality seeds, edit mediocre ones, then save as:")
    print(f"  {PROJECT_ROOT / 'data' / 'seeds' / 'expanded_curated.json'}")


if __name__ == "__main__":
    main()
