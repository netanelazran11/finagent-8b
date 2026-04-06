# FinAgent — a fine-tuned financial reasoning agent

End-to-end project: I generated a synthetic dataset, QLoRA-fine-tuned **Mistral 7B v0.3** into `finagent-7b`, wired it into a **ReAct tool-use loop** against the Alpha Vantage market-data API, and built an eval harness comparing it against the base model.

> **Why this matters**: a 7B model fine-tuned on the right data can match much larger generalist models on a focused domain. This repo shows the full loop — data, training, agent, evaluation — without hand-waving any step.

[![CI](https://github.com/netanelazran11/finagent-8b/actions/workflows/ci.yml/badge.svg)](https://github.com/netanelazran11/finagent-8b/actions/workflows/ci.yml)
[![Model on HF](https://img.shields.io/badge/🤗_model-danab17/finagent--7b--merged-yellow)](https://huggingface.co/danab17/finagent-7b-merged)

---

## Architecture

```
                  ┌─────────────────────────┐
  Module 1 ─────► │  Synthetic dataset      │   ~2400 examples, 3 types:
                  │  (GPT-4o as teacher)    │   CoT reasoning · tool trajectories · guardrails
                  └────────────┬────────────┘
                               │  train.jsonl / val.jsonl
                               ▼
                  ┌─────────────────────────┐
  Module 2 ─────► │  QLoRA fine-tune        │   Mistral 7B v0.3 → finagent-7b
                  │  (Unsloth on L40S)      │   4-bit base · LoRA r=16 · ~160M trainable params
                  └────────────┬────────────┘
                               │  merged weights → HF Hub
                               ▼
                  ┌─────────────────────────┐
  Module 3 ─────► │  ReAct agent            │   from-scratch loop AND LangGraph variant
                  │  + 7 financial tools    │   Alpha Vantage quotes / ratios / news / yields
                  └────────────┬────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
  Module 5 ─────► │  Evaluation harness     │   tool-call accuracy · args validity · LLM-judge
                  │  (mock + GPU modes)     │   compares fine-tuned vs base
                  └─────────────────────────┘
```

## Quick start

```bash
git clone https://github.com/netanelazran11/finagent-8b.git
cd finagent
pip install -r requirements.txt
cp .env.example .env       # add ALPHAVANTAGE_API_KEY + OPENAI_API_KEY

# Run the tests (no GPU needed — unsloth is mocked)
make test

# Run the agent (requires GPU)
make agent

# Run the eval harness in mock mode (no GPU, no API)
make eval

# Run the eval harness on the real fine-tuned model (requires GPU)
python scripts/eval.py --mode=gpu --model=danab17/finagent-7b-merged --judge
```

## Modules

### 1 — Synthetic data engineering (`scripts/generate_dataset.py`)

Distilabel pipeline with GPT-4o as the teacher. Three example types, each with its own template in `configs/prompt_templates.py`:

- **CoT reasoning**: `<think>`-block decomposition + structured advisor answer.
- **Tool trajectories**: multi-turn `assistant → tool → assistant` sequences with valid `[TOOL_CALLS]` JSON and grounded final answers.
- **Guardrails**: refusal examples (concentration risk, unrealistic returns, gambling-style behavior) with empathetic redirection.

Output: `data/processed/{train,val}.jsonl` in Mistral chat format.

### 2 — QLoRA fine-tuning (`scripts/train_qlora.py`, `notebooks/module2_qlora_finetune.ipynb`)

- Base: `mistralai/Mistral-7B-Instruct-v0.3` (chosen for native parallel `tool_calls` support).
- Unsloth + 4-bit NF4 + LoRA `r=16, α=32` on `q/k/v/o/gate/up/down_proj`.
- `bf16` on L40S, batch 8 × grad-accum 2 = effective 16.
- Cosine LR schedule, `2e-4` peak, ~3 epochs, eval every 50 steps.

The SLURM script (`scripts/slurm_train.sh`) reproduces the run on a single L40S in ~45 min.

### 3 — Agent (`scripts/agent_from_scratch.py` + `scripts/agent_langgraph.py`)

Two implementations of the **same** ReAct loop, kept side by side on purpose:

| | `agent_from_scratch.py` | `agent_langgraph.py` |
|---|---|---|
| Orchestration | hand-written `for _ in range(max_iters)` | LangGraph state machine |
| Loop control | `if not tool_calls: break` | `conditional_edges` |
| What it shows | every byte of the agent contract | how to scale to a real product |

Both call the **same** `TOOL_REGISTRY` (`scripts/tools.py`) — 7 finance tools backed by Alpha Vantage with a 60-minute file cache.

Try it interactively in `notebooks/module3_agent_demo.ipynb`, or live via Gradio:

```bash
python scripts/app.py            # launches at localhost:7860
```

### 5 — Evaluation (`scripts/eval.py`)

20 fixture questions across 4 categories (`data/eval/questions.jsonl`):

| Category | What it tests |
|---|---|
| `single_tool` | does the model pick the right tool for a focused question? |
| `parallel_tools` | does it batch independent tools in one assistant turn? |
| `multi_turn` | does it sequence tools when an answer depends on a prior call? |
| `cot_only` | does it answer reasoning questions without unnecessary tool calls? |
| `guardrail` | does it refuse dangerous asks without calling tools? |

Metrics: tool recall/precision, exact-set match, args JSON validity, guardrail pass rate, optional GPT-4o-mini judge on a 1–5 rubric.

```bash
# Mock backend — for CI, validates harness logic without a model
python scripts/eval.py --mode=mock

# Real model — on GPU
python scripts/eval.py --mode=gpu --judge
```

Output: `results/eval_report.md` (markdown summary) + `results/eval_predictions.jsonl` (raw).

## Design decisions worth flagging

- **Mistral 7B v0.3, not Llama 3.1**: parallel `tool_calls` in a single assistant turn are native — critical for "fetch price AND ratios" patterns.
- **From-scratch agent AND LangGraph**: keeping both forces the framework to earn its keep. The from-scratch loop fits on one screen; LangGraph is justified the moment you want streaming, branching, or human-in-the-loop. See `scripts/compare_agents.py`.
- **File-based cache on Alpha Vantage**: the free tier is 25 req/day. Without cache, three test runs burn the daily quota. TTL is 60 min — enough for a development session, short enough that intraday-changing data stays fresh.
- **Mock backend for the eval**: lets CI run the full eval pipeline (loading questions, scoring, writing the report) without GPU or API keys. Catches harness bugs early.
- **No `pyproject.toml`-style package install**: scripts use `sys.path` hacks because the project is a learning artifact, not a library. Promoting to a `finagent/` package is one rename away.

## Repo layout

```
configs/
  prompt_templates.py        teacher-model templates for the three data types
  tool_definitions.json      JSON schema for the 7 tools (Mistral chat format)
scripts/
  generate_dataset.py        Distilabel pipeline (GPT-4o teacher)
  expand_seeds.py            seed-question expansion via paraphrase + variation
  prepare_dataset.py         shuffle, split, tokenize-check
  train_qlora.py             Unsloth QLoRA training loop (also as notebook)
  slurm_train.sh             one-node SLURM submission for the lab cluster
  tools.py                   7 financial tools + TOOL_REGISTRY + AV cache
  agent_from_scratch.py      bare-metal ReAct loop
  agent_langgraph.py         same loop via LangGraph
  app.py                     Gradio demo
  eval.py                    eval harness (mock + GPU modes)
  compare_agents.py          benchmark from-scratch vs LangGraph
tests/
  test_tools.py              43 unit tests on the 7 tools (mocked AV API)
  test_parse_tool_calls.py   9 tests on Mistral [TOOL_CALLS] parsing
  test_execute_tools.py      6 tests on the dispatch layer
notebooks/
  module2_qlora_finetune.ipynb
  module3_agent_demo.ipynb   step-by-step walk through the ReAct loop
data/
  seeds/                     seed questions for the teacher
  processed/                 train.jsonl / val.jsonl after splitting
  eval/questions.jsonl       20 eval fixtures with expected tool calls
```

## Status

- [x] Module 1 — synthetic data pipeline
- [x] Module 2 — QLoRA fine-tune (model on the HF Hub)
- [x] Module 3 — ReAct agent (both implementations) + 7 tools
- [x] Module 5 — evaluation harness
- [ ] Module 4 — real-time RAG (not started; out of scope for the portfolio cut)

## License

MIT — see `LICENSE`.
