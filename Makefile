.PHONY: help install test lint format agent agent-langgraph eval app clean

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
	@echo "  clean           Remove caches and pyc files"

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install ruff pytest

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m ruff check scripts tests configs

format:
	$(PYTHON) -m ruff format scripts tests configs
	$(PYTHON) -m ruff check --fix scripts tests configs

agent:
	$(PYTHON) scripts/agent_from_scratch.py

agent-langgraph:
	$(PYTHON) scripts/agent_langgraph.py

eval:
	$(PYTHON) scripts/eval.py

app:
	$(PYTHON) scripts/app.py

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +
	rm -f .av_cache.json
