"""
ResearchAgent — answers conceptual financial questions via RAG over local documents.

Flow:
    query
      → FinancialRAG.retrieve(query, k=4)     # find relevant chunks
      → format_context(chunks)                 # build context block
      → LLM synthesis prompt                   # generate grounded answer
      → response with citations

LLM backend (in priority order):
  1. GPT-4o-mini (if OPENAI_API_KEY is set)     — fast, cheap, great at synthesis
  2. Fine-tuned model (if model/tokenizer given) — showcases the fine-tune
  3. Mock (if mock_mode=True)                   — deterministic, for CI

The ResearchAgent does NOT call Alpha Vantage — live market data is the AnalystAgent's job.
"""

from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger("finagent.agents.research")

SYNTHESIS_SYSTEM = """You are FinAgent, a careful financial research assistant.
You will be given a question and relevant excerpts from financial documents.
Your job is to synthesize a clear, accurate answer GROUNDED IN THE CONTEXT.

Rules:
- Only state facts that are supported by the provided context.
- If the context doesn't contain enough information, say so honestly.
- Cite the source document when referencing specific data.
- Flag any risks or important caveats.
- Keep the answer concise (3–5 sentences for most questions).
"""

SYNTHESIS_USER = """CONTEXT:
{context}

QUESTION:
{query}

Answer based on the context above:"""

_MOCK_ANSWERS = {
    "default": (
        "Based on the financial knowledge base: this is a complex topic that depends on "
        "your specific situation. Generally, diversification and a long time horizon are "
        "the most reliable paths to financial stability. Consider consulting a licensed "
        "financial advisor for personalized guidance."
    )
}


class ResearchAgent:
    """RAG-powered research agent for conceptual financial questions.

    Args:
        rag: A FinancialRAG instance (or None for mock_mode).
        model: Loaded fine-tuned model object, OR None to use GPT-4o-mini.
        tokenizer: Tokenizer paired with `model`.
        mock_mode: If True, skip all LLM/RAG calls and return canned responses.
    """

    def __init__(
        self,
        rag: Any | None = None,
        model: Any | None = None,
        tokenizer: Any | None = None,
        mock_mode: bool = False,
    ) -> None:
        self.rag = rag
        self.model = model
        self.tokenizer = tokenizer
        self.mock_mode = mock_mode
        self._openai_key = os.getenv("OPENAI_API_KEY")

    def run(self, query: str, k: int = 4) -> dict:
        """Run the research pipeline.

        Returns:
            {
                "answer": str,
                "context_chunks": list[dict],   # retrieved docs
                "context_str": str,             # formatted for prompt
                "sources": list[str],           # unique source filenames
                "backend": str,                 # "mock"|"gpt-4o-mini"|"finagent"
            }
        """
        if self.mock_mode or self.rag is None:
            return self._mock_response(query)

        chunks = self.rag.retrieve(query, k=k)
        context_str = self.rag.format_context(chunks)
        sources = list({c["source"] for c in chunks})

        answer = self._synthesize(query, context_str)
        return {
            "answer": answer,
            "context_chunks": chunks,
            "context_str": context_str,
            "sources": sources,
            "backend": self._backend_name(),
        }

    # ------------------------------------------------------------------
    # Internal synthesis methods
    # ------------------------------------------------------------------

    def _synthesize(self, query: str, context: str) -> str:
        """Pick the best available LLM backend and generate an answer."""
        if self._openai_key:
            return self._synthesize_openai(query, context)
        if self.model is not None:
            return self._synthesize_finagent(query, context)
        return _MOCK_ANSWERS["default"]

    def _synthesize_openai(self, query: str, context: str) -> str:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self._openai_key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYNTHESIS_SYSTEM},
                    {
                        "role": "user",
                        "content": SYNTHESIS_USER.format(context=context, query=query),
                    },
                ],
                temperature=0.2,
                max_tokens=512,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            log.warning("OpenAI synthesis failed: %s", e)
            return f"Synthesis failed: {e}"

    def _synthesize_finagent(self, query: str, context: str) -> str:
        """Use the fine-tuned model for synthesis (requires GPU)."""
        messages = [
            {"role": "system", "content": SYNTHESIS_SYSTEM},
            {
                "role": "user",
                "content": SYNTHESIS_USER.format(context=context, query=query),
            },
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        outputs = self.model.generate(
            input_ids=inputs,
            max_new_tokens=512,
            temperature=0.3,
            do_sample=True,
        )
        return self.tokenizer.decode(
            outputs[0][inputs.shape[1] :], skip_special_tokens=True
        ).strip()

    def _backend_name(self) -> str:
        if self._openai_key:
            return "gpt-4o-mini"
        if self.model is not None:
            return "finagent"
        return "mock"

    def _mock_response(self, query: str) -> dict:
        return {
            "answer": _MOCK_ANSWERS["default"],
            "context_chunks": [],
            "context_str": "",
            "sources": [],
            "backend": "mock",
        }
