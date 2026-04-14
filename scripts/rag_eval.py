#!/usr/bin/env python3
"""
Module 4 — RAGAS evaluation of the RAG pipeline.

Measures how well the ResearchAgent's retrieval + synthesis performs on a
set of financial questions with reference answers.

RAGAS Metrics:
  - faithfulness:       Are claims in the answer supported by the retrieved context?
  - answer_relevancy:   Does the answer actually address the question?
  - context_recall:     Does the retrieved context cover the ground-truth answer?
  - context_precision:  Is the retrieved context concise and relevant?

Usage:
    # Requires: OPENAI_API_KEY + a built index
    python scripts/build_index.py
    python scripts/rag_eval.py

    # Mock mode (no API, validates harness):
    python scripts/rag_eval.py --mode=mock

Output:
    results/rag_eval_report.md
    results/rag_eval_predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Evaluation fixtures (question, ground_truth)
# ---------------------------------------------------------------------------

EVAL_FIXTURES: list[dict] = [
    {
        "id": "rag_001",
        "question": "What is the Federal Reserve's inflation target?",
        "ground_truth": "The Federal Reserve targets 2% PCE inflation.",
        "category": "monetary_policy",
    },
    {
        "id": "rag_002",
        "question": "What does an inverted yield curve signal?",
        "ground_truth": "An inverted yield curve, where short-term rates exceed long-term rates, has historically preceded recessions by 12-18 months.",
        "category": "monetary_policy",
    },
    {
        "id": "rag_003",
        "question": "What is the Sharpe ratio and what does it measure?",
        "ground_truth": "The Sharpe ratio measures return per unit of risk, calculated as (portfolio return - risk-free rate) / standard deviation. A Sharpe above 1 is considered good.",
        "category": "risk_management",
    },
    {
        "id": "rag_004",
        "question": "How is a P/E ratio calculated and what is the historical average for the S&P 500?",
        "ground_truth": "The P/E ratio is calculated as price divided by earnings per share. The S&P 500 historical average P/E is approximately 16-17x.",
        "category": "equity_valuation",
    },
    {
        "id": "rag_005",
        "question": "What is free cash flow and why is it more reliable than reported EPS?",
        "ground_truth": "Free cash flow is operating cash flow minus capital expenditures. It is harder to manipulate than reported EPS because it reflects actual cash movement.",
        "category": "earnings_analysis",
    },
    {
        "id": "rag_006",
        "question": "What are the 11 GICS sectors that make up the S&P 500?",
        "ground_truth": "The 11 GICS sectors are: Information Technology, Healthcare, Financials, Consumer Discretionary, Communication Services, Industrials, Consumer Staples, Energy, Real Estate, Materials, and Utilities.",
        "category": "market_structure",
    },
    {
        "id": "rag_007",
        "question": "What is diversification and why does it reduce portfolio risk?",
        "ground_truth": "Diversification combines assets with imperfect correlations, reducing unsystematic risk. It works because assets don't all move in the same direction at the same time.",
        "category": "risk_management",
    },
    {
        "id": "rag_008",
        "question": "What is the difference between trailing and forward P/E?",
        "ground_truth": "Trailing P/E uses the last 12 months of actual earnings; forward P/E uses analyst estimates for the next 12 months.",
        "category": "equity_valuation",
    },
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class RAGPrediction:
    id: str
    question: str
    ground_truth: str
    category: str
    answer: str = ""
    contexts: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_recall: float | None = None
    context_precision: float | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# RAGAS scoring
# ---------------------------------------------------------------------------


def score_with_ragas(predictions: list[RAGPrediction]) -> list[RAGPrediction]:
    """Run RAGAS metrics on predictions that have answer + contexts."""
    try:
        from datasets import Dataset
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
    except ImportError:
        print("[ragas] ragas or datasets not installed — skipping metric scoring.")
        return predictions

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("[ragas] OPENAI_API_KEY not set — skipping metric scoring.")
        return predictions

    samples = []
    valid_preds = [p for p in predictions if not p.error and p.answer and p.contexts]
    for p in valid_preds:
        try:
            sample = SingleTurnSample(
                user_input=p.question,
                response=p.answer,
                retrieved_contexts=p.contexts,
                reference=p.ground_truth,
            )
            samples.append(sample)
        except Exception as e:
            print(f"[ragas] Failed to create sample for {p.id}: {e}")

    if not samples:
        return predictions

    try:
        dataset = EvaluationDataset(samples=samples)
        metrics = [Faithfulness(), AnswerRelevancy(), ContextRecall(), ContextPrecision()]
        results = evaluate(dataset=dataset, metrics=metrics)
        df = results.to_pandas()

        for i, pred in enumerate(valid_preds):
            if i < len(df):
                row = df.iloc[i]
                pred.faithfulness = float(row.get("faithfulness", 0) or 0)
                pred.answer_relevancy = float(row.get("answer_relevancy", 0) or 0)
                pred.context_recall = float(row.get("context_recall", 0) or 0)
                pred.context_precision = float(row.get("context_precision", 0) or 0)
    except Exception as e:
        print(f"[ragas] Evaluation failed: {e}")

    return predictions


# ---------------------------------------------------------------------------
# Mock backend
# ---------------------------------------------------------------------------


def run_mock(fixtures: list[dict]) -> list[RAGPrediction]:
    """Deterministic mock: validates harness without API/index."""
    preds = []
    for f in fixtures:
        preds.append(RAGPrediction(
            id=f["id"],
            question=f["question"],
            ground_truth=f["ground_truth"],
            category=f["category"],
            answer="[Mock] This is a deterministic answer for harness validation.",
            contexts=["[Mock context chunk 1]", "[Mock context chunk 2]"],
            sources=["mock_doc.txt"],
        ))
    return preds


# ---------------------------------------------------------------------------
# Real RAG backend
# ---------------------------------------------------------------------------


def run_rag(fixtures: list[dict], openai_key: str | None) -> list[RAGPrediction]:
    """Run the real FinancialRAG + ResearchAgent pipeline."""
    from finagent.agents.research import ResearchAgent
    from finagent.rag import FinancialRAG

    try:
        rag = FinancialRAG.load()
    except RuntimeError as e:
        print(f"[rag] {e}")
        return run_mock(fixtures)

    agent = ResearchAgent(rag=rag, mock_mode=(openai_key is None))

    preds = []
    for f in fixtures:
        print(f"  [{f['id']}] {f['question'][:70]}")
        try:
            result = agent.run(f["question"], k=4)
            preds.append(RAGPrediction(
                id=f["id"],
                question=f["question"],
                ground_truth=f["ground_truth"],
                category=f["category"],
                answer=result["answer"],
                contexts=[c["content"] for c in result["context_chunks"]],
                sources=result.get("sources", []),
            ))
        except Exception as e:
            preds.append(RAGPrediction(
                id=f["id"],
                question=f["question"],
                ground_truth=f["ground_truth"],
                category=f["category"],
                error=str(e),
            ))
    return preds


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def write_report(preds: list[RAGPrediction], path: Path, mode: str) -> None:
    def mean(xs):
        vals = [x for x in xs if x is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    lines = [
        "# FinAgent — RAG Evaluation Report (Module 4)",
        "",
        f"- **Mode**: `{mode}`",
        f"- **Questions**: {len(preds)}",
        "",
        "## RAGAS Metrics",
        "",
        "| Metric | Score |",
        "|---|---|",
    ]

    for metric, label in [
        ("faithfulness", "Faithfulness"),
        ("answer_relevancy", "Answer Relevancy"),
        ("context_recall", "Context Recall"),
        ("context_precision", "Context Precision"),
    ]:
        score = mean([getattr(p, metric) for p in preds])
        val = f"**{score:.3f}**" if score is not None else "—"
        lines.append(f"| {label} | {val} |")

    lines += [
        "",
        "## Per-Question Results",
        "",
        "| ID | Category | Faithfulness | Relevancy | Recall | Precision |",
        "|---|---|---|---|---|---|",
    ]
    for p in preds:
        f = f"{p.faithfulness:.2f}" if p.faithfulness is not None else "—"
        r = f"{p.answer_relevancy:.2f}" if p.answer_relevancy is not None else "—"
        rc = f"{p.context_recall:.2f}" if p.context_recall is not None else "—"
        pr = f"{p.context_precision:.2f}" if p.context_precision is not None else "—"
        lines.append(f"| {p.id} | {p.category} | {f} | {r} | {rc} | {pr} |")

    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["rag", "mock"], default="mock",
                    help="'rag' requires a built index + OPENAI_API_KEY for scoring")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    openai_key = os.getenv("OPENAI_API_KEY")

    print(f"FinAgent RAG Evaluation — mode={args.mode}")
    print(f"Fixtures: {len(EVAL_FIXTURES)} questions\n")

    if args.mode == "mock":
        preds = run_mock(EVAL_FIXTURES)
    else:
        preds = run_rag(EVAL_FIXTURES, openai_key)
        if openai_key:
            print("\nRunning RAGAS scoring (requires OPENAI_API_KEY)...")
            preds = score_with_ragas(preds)

    # Write JSONL
    pred_path = out_dir / "rag_eval_predictions.jsonl"
    with open(pred_path, "w") as f:
        for p in preds:
            f.write(json.dumps(asdict(p)) + "\n")

    # Write report
    report_path = out_dir / "rag_eval_report.md"
    write_report(preds, report_path, mode=args.mode)

    print(f"\nPredictions → {pred_path}")
    print(f"Report      → {report_path}")

    # Print metric summary
    scored = [p for p in preds if p.faithfulness is not None]
    if scored:
        def mean(xs):
            return round(sum(xs) / len(xs), 3)
        print("\nMetrics:")
        print(f"  faithfulness:       {mean([p.faithfulness for p in scored])}")
        print(f"  answer_relevancy:   {mean([p.answer_relevancy for p in scored])}")
        print(f"  context_recall:     {mean([p.context_recall for p in scored])}")
        print(f"  context_precision:  {mean([p.context_precision for p in scored])}")
    else:
        print("\n(Set OPENAI_API_KEY and use --mode=rag for RAGAS scores)")


if __name__ == "__main__":
    main()
