"""Quick debug script to inspect what Distilabel actually returns."""
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from distilabel.models import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts
from distilabel.steps.tasks import TextGeneration

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Use just 2 seeds to debug
test_inputs = [
    {
        "instruction": "Explain in 3 sentences why diversification matters in investing.",
        "seed_id": "debug_001",
    },
    {
        "instruction": "Explain in 3 sentences the concept of dollar-cost averaging.",
        "seed_id": "debug_002",
    },
]

import os

with Pipeline(name="debug") as pipeline:
    load_data = LoadDataFromDicts(data=test_inputs)
    generate = TextGeneration(
        llm=OpenAILLM(
            model="llama-3.3-70b-versatile",
            base_url=GROQ_BASE_URL,
            api_key=os.environ["GROQ_API_KEY"],
            generation_kwargs={"temperature": 0.7, "max_tokens": 512},
        ),
        num_generations=1,
    )
    load_data >> generate

results = pipeline.run(use_cache=False)

# Inspect the structure
print("\n" + "=" * 60)
print("DISTISET KEYS:", list(results.keys()))

ds = results["default"]["train"]
print(f"DATASET COLUMNS: {ds.column_names}")
print(f"NUM ROWS: {len(ds)}")

print("\n--- FIRST ROW ---")
row = ds[0]
for key, val in row.items():
    val_str = str(val)[:200]
    print(f"  {key}: {val_str}")
