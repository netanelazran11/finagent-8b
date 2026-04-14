.PHONY: help install test lint format agent agent-langgraph eval app multi-agent build-index rag-eval clean

PYTHON ?= python

help:
	@echo "Available targets:"
	@echo "  install         Install dev + runtime requirements"
	@echo "  test            Run pytest (mocks unsloth, no GPU needed)"
	@echo "  lint            Run ruff check"
	@echo "  format          Run ruff format + fix"
	@echo "  agent           Run the from-scratch ReAct agent (requires GPU)"
	@echo "  agent-langgraph Run the LangGraph agent (requires GPU)"
	@echo "  eval            Run the eval harness against the fine-tuned model"
	@echo "  app             Launch the Gradio demo"
	@echo "  build-index     Build Chroma RAG index from data/financial_docs/"
	@echo "  rag-eval        Run RAGAS evaluation on the RAG pipeline"
	@echo "  multi-agent     Run the multi-agent system (demo mode)"
	@echo "  clean           Remove caches and pyc files"

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install ruff pytest
	$(PYTHON) -m pip install -e .

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m ruff check scripts tests configs finagent

format:
	$(PYTHON) -m ruff format scripts tests configs finagent
	$(PYTHON) -m ruff check --fix scripts tests configs finagent

agent:
	$(PYTHON) scripts/agent_from_scratch.py

agent-langgraph:
	$(PYTHON) scripts/agent_langgraph.py

eval:
	$(PYTHON) scripts/eval.py

app:
	$(PYTHON) scripts/app.py

build-index:
	$(PYTHON) scripts/build_index.py

rag-eval:
	$(PYTHON) scripts/rag_eval.py

multi-agent:
	$(PYTHON) scripts/multi_agent.py

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +
	rm -f .av_cache.json
