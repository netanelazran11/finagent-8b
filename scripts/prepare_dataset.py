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
import string
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def fix_tool_call_ids(messages: list[dict]) -> list[dict]:
    """Rewrite tool_call IDs to be alphanumeric, 9 chars (Mistral requirement).

    Mistral v0.3's chat template enforces: alphanumeric, exactly 9 characters.
    Our generated data uses 'call_001' style IDs which don't comply.
    """
    rng = random.Random(42)
    id_map = {}  # old_id -> new_id

    def make_id():
        chars = string.ascii_letters + string.digits
        return "".join(rng.choice(chars) for _ in range(9))

    for msg in messages:
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                old_id = tc["id"]
                if old_id not in id_map:
                    id_map[old_id] = make_id()
                tc["id"] = id_map[old_id]
        if msg.get("role") == "tool" and msg.get("tool_call_id"):
            old_id = msg["tool_call_id"]
            if old_id in id_map:
                msg["tool_call_id"] = id_map[old_id]

    return messages


def fix_tool_messages(messages: list[dict]) -> list[dict]:
    """Fix tool response messages: add 'name' field from matching tool_call.

    Mistral v0.3 expects tool messages to have a 'name' field with the function name.
    """
    # Build lookup: tool_call_id -> function name
    id_to_name = {}
    for msg in messages:
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                id_to_name[tc["id"]] = tc["function"]["name"]

    for msg in messages:
        if msg["role"] == "tool" and "name" not in msg:
            tcid = msg.get("tool_call_id", "")
            if tcid in id_to_name:
                msg["name"] = id_to_name[tcid]

    return messages


def clean_empty_tool_calls(messages: list[dict]) -> list[dict]:
    """Remove empty tool_calls fields from assistant messages.

    Mistral's template checks 'tool_calls is not none' but an empty list []
    is not none — so the message is NOT skipped during alternation checks,
    breaking the user/assistant alternation validation.
    """
    for msg in messages:
        if msg.get("role") == "assistant" and "tool_calls" in msg:
            if not msg["tool_calls"]:  # empty list or None
                del msg["tool_calls"]
    return messages


def merge_consecutive_assistants(messages: list[dict]) -> list[dict]:
    """Merge consecutive assistant messages into one.

    Mistral requires strict alternation: user/assistant/user/assistant.
    Our generated data sometimes has two assistant messages in a row
    (e.g., a <think> block followed by a final response). Merge them.
    """
    result = []
    for msg in messages:
        if (result
                and msg["role"] == "assistant"
                and result[-1]["role"] == "assistant"
                and not result[-1].get("tool_calls")):
            # Merge content into previous assistant message
            prev_content = result[-1].get("content", "") or ""
            new_content = msg.get("content", "") or ""
            result[-1]["content"] = (prev_content + "\n" + new_content).strip()
            # If the new msg has tool_calls, carry them over
            if msg.get("tool_calls"):
                result[-1]["tool_calls"] = msg["tool_calls"]
        else:
            result.append(msg)
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

    # Fix for Mistral v0.3 compatibility
    id_fixed = 0
    merge_fixed = 0
    for ex in examples:
        msgs = ex["messages"]

        # 1. Fix tool_call IDs (9-char alphanumeric)
        for msg in msgs:
            if msg.get("tool_calls"):
                fix_tool_call_ids(msgs)
                id_fixed += 1
                break

        # 2. Add 'name' field to tool responses
        fix_tool_messages(msgs)

        # 3. Remove empty tool_calls fields
        clean_empty_tool_calls(msgs)

        # 4. Merge consecutive assistant messages
        old_len = len(msgs)
        ex["messages"] = merge_consecutive_assistants(msgs)
        if len(ex["messages"]) != old_len:
            merge_fixed += 1

    if id_fixed:
        print(f"Fixed tool_call IDs in {id_fixed} examples")
    if merge_fixed:
        print(f"Merged consecutive assistants in {merge_fixed} examples")

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
