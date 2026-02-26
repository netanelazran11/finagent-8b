"""
Dataset Preparation Script
===========================

Merges raw JSONL files from the three pipelines into a single
training-ready dataset with train/validation split.

Architecture Note:
-----------------
Why a separate validation set matters for fine-tuning:
  - Training loss alone can be misleading (it always goes down)
  - Validation loss tells you when the model starts MEMORIZING instead of LEARNING
  - For QLoRA with 2,500 examples, we use a 90/10 split
  - The val set must contain ALL three data types proportionally
    (otherwise you can't detect if tool-calling quality degrades while CoT improves)

Output format:
  data/processed/train.jsonl  — 90% of data, shuffled
  data/processed/val.jsonl    — 10% of data, stratified by type
  data/processed/stats.json   — Dataset statistics for documentation
"""

import json
import random
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def split_parallel_tool_calls(messages: list[dict]) -> list[dict]:
    """Split parallel tool_calls into sequential single-call messages.

    Llama 3.1's chat template only supports one tool_call per assistant message.
    If an assistant message has N tool_calls, we split it into N assistant messages,
    each followed by its corresponding tool response.

    Before:  [assistant(think + 2 tool_calls), tool_1, tool_2, assistant(final)]
    After:   [assistant(think + tool_call_1), tool_1, assistant(tool_call_2), tool_2, assistant(final)]
    """
    result = []
    # Build a lookup: tool_call_id -> tool response message
    tool_responses = {}
    for msg in messages:
        if msg["role"] == "tool" and "tool_call_id" in msg:
            tool_responses[msg["tool_call_id"]] = msg

    i = 0
    while i < len(messages):
        msg = messages[i]

        if msg.get("tool_calls") and len(msg["tool_calls"]) > 1:
            # Split: first call keeps the <think> content, rest get empty content
            for j, tc in enumerate(msg["tool_calls"]):
                assistant_msg = {
                    "role": "assistant",
                    "content": msg.get("content", "") if j == 0 else "",
                    "tool_calls": [tc],
                }
                result.append(assistant_msg)
                # Insert corresponding tool response right after
                if tc["id"] in tool_responses:
                    result.append(tool_responses[tc["id"]])

            # Skip the original tool response messages (we already inserted them)
            i += 1
            while i < len(messages) and messages[i]["role"] == "tool":
                i += 1
        else:
            result.append(msg)
            i += 1

    return result


def load_all_examples() -> list[dict]:
    """Load all JSONL files from the raw directory."""
    examples = []
    for jsonl_file in RAW_DIR.glob("*.jsonl"):
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))

    print(f"Loaded {len(examples)} total examples")

    # Split parallel tool_calls for Llama 3.1 compatibility
    split_count = 0
    for ex in examples:
        old_len = len(ex["messages"])
        ex["messages"] = split_parallel_tool_calls(ex["messages"])
        if len(ex["messages"]) != old_len:
            split_count += 1

    if split_count:
        print(f"Split parallel tool_calls in {split_count} examples")

    return examples


def stratified_split(
    examples: list[dict],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Split examples into train/val sets, stratified by generation_type.

    Stratification ensures the validation set has proportional representation
    of each data type. Without this, you might end up with a val set that's
    100% CoT and 0% tool examples — making it useless for detecting tool-calling
    regression during training.
    """
    random.seed(seed)

    # Group by type
    by_type: dict[str, list[dict]] = {}
    for ex in examples:
        gen_type = ex.get("metadata", {}).get("generation_type", "unknown")
        by_type.setdefault(gen_type, []).append(ex)

    train, val = [], []

    for gen_type, type_examples in by_type.items():
        random.shuffle(type_examples)
        n_val = max(1, int(len(type_examples) * val_ratio))
        val.extend(type_examples[:n_val])
        train.extend(type_examples[n_val:])

    # Final shuffle (so training doesn't see all CoT first, then all tool, etc.)
    random.shuffle(train)
    random.shuffle(val)

    return train, val


def compute_stats(train: list[dict], val: list[dict]) -> dict:
    """Compute dataset statistics for documentation and sanity checking."""
    def type_counts(examples):
        return Counter(
            ex.get("metadata", {}).get("generation_type", "unknown")
            for ex in examples
        )

    def avg_messages(examples):
        lengths = [len(ex["messages"]) for ex in examples if ex.get("messages")]
        return round(sum(lengths) / len(lengths), 1) if lengths else 0

    def avg_assistant_tokens(examples):
        """Rough token estimate (words * 1.3) for assistant content."""
        total_words = 0
        count = 0
        for ex in examples:
            for msg in ex.get("messages", []):
                if msg["role"] == "assistant":
                    total_words += len(msg.get("content", "").split())
                    count += 1
        return round((total_words / count * 1.3) if count else 0)

    return {
        "train_total": len(train),
        "val_total": len(val),
        "train_by_type": dict(type_counts(train)),
        "val_by_type": dict(type_counts(val)),
        "avg_messages_per_example": avg_messages(train + val),
        "approx_avg_assistant_tokens": avg_assistant_tokens(train + val),
    }


def save_jsonl(examples: list[dict], path: Path):
    """Save examples as JSONL (one JSON object per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            # For training, we only need the "messages" field
            # Metadata is stripped to keep the training file clean
            record = {"messages": ex["messages"]}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Saved {len(examples)} examples → {path}")


def main():
    print("=" * 60)
    print("FinAgent — Dataset Preparation")
    print("=" * 60)

    examples = load_all_examples()

    if not examples:
        print("No examples found in data/raw/. Run generate_dataset.py first.")
        return

    train, val = stratified_split(examples)
    stats = compute_stats(train, val)

    # Save
    save_jsonl(train, PROCESSED_DIR / "train.jsonl")
    save_jsonl(val, PROCESSED_DIR / "val.jsonl")

    with open(PROCESSED_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Print summary
    print(f"\n{'─' * 40}")
    print(f"  Train: {stats['train_total']} examples")
    for t, c in stats["train_by_type"].items():
        print(f"    {t}: {c}")
    print(f"  Val:   {stats['val_total']} examples")
    for t, c in stats["val_by_type"].items():
        print(f"    {t}: {c}")
    print(f"  Avg messages/example: {stats['avg_messages_per_example']}")
    print(f"  Avg assistant tokens: {stats['approx_avg_assistant_tokens']}")
    print(f"{'─' * 40}")


if __name__ == "__main__":
    main()
