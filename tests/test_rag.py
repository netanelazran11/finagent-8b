"""
Unit tests for finagent/rag.py — Module 4 RAG pipeline.

Tests use mocked Chroma/embeddings so they run without network access, GPU, or
downloading the embedding model. Run: python -m pytest tests/test_rag.py -v
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch


class TestFinancialRAGInterface:
    """Test the FinancialRAG public interface with a mocked vector store."""

    def _make_rag(self, docs=None):
        """Create a FinancialRAG with mocked embeddings and Chroma store."""
        if docs is None:
            docs = [
                MagicMock(page_content="The federal funds rate is 4.25–4.50%.",
                          metadata={"source": "monetary_policy.txt"}),
                MagicMock(page_content="The S&P 500 historical average P/E is 16–17x.",
                          metadata={"source": "equity_valuation.txt"}),
                MagicMock(page_content="The Sharpe ratio measures return per unit of risk.",
                          metadata={"source": "risk_management.txt"}),
            ]

        mock_embeddings = MagicMock()
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = docs[:2]
        mock_store.similarity_search_with_score.return_value = [(d, 0.1 * i) for i, d in enumerate(docs[:2])]
        mock_store._collection.count.return_value = len(docs)

        with (
            patch("finagent.rag.Chroma", return_value=mock_store),  # noqa: F841
            patch("finagent.rag.HuggingFaceEmbeddings", return_value=mock_embeddings),
        ):
            from finagent.rag import FinancialRAG
            rag = FinancialRAG.__new__(FinancialRAG)
            rag._embeddings = mock_embeddings
            rag._store = mock_store
        return rag

    def test_retrieve_returns_list_of_dicts(self):
        rag = self._make_rag()
        results = rag.retrieve("What is the fed funds rate?", k=2)
        assert isinstance(results, list)
        assert len(results) == 2
        for chunk in results:
            assert "content" in chunk
            assert "source" in chunk

    def test_retrieve_content_and_source(self):
        rag = self._make_rag()
        results = rag.retrieve("fed funds rate", k=2)
        assert "federal funds rate" in results[0]["content"]
        assert results[0]["source"] == "monetary_policy.txt"

    def test_retrieve_with_scores_includes_distance(self):
        rag = self._make_rag()
        results = rag.retrieve_with_scores("P/E ratio", k=2)
        assert len(results) == 2
        for r in results:
            assert "distance" in r
            assert isinstance(r["distance"], float)

    def test_format_context_returns_string(self):
        rag = self._make_rag()
        chunks = [
            {"content": "The Fed targets 2% inflation.", "source": "monetary_policy.txt"},
            {"content": "The P/E ratio divides price by earnings.", "source": "equity_valuation.txt"},
        ]
        context = rag.format_context(chunks)
        assert isinstance(context, str)
        assert "Source 1: monetary_policy.txt" in context
        assert "Source 2: equity_valuation.txt" in context
        assert "---" in context

    def test_format_context_empty(self):
        rag = self._make_rag()
        context = rag.format_context([])
        assert "No relevant context" in context

    def test_is_empty_false_when_indexed(self):
        rag = self._make_rag()
        assert rag.is_empty() is False

    def test_is_empty_true_when_no_docs(self):
        rag = self._make_rag(docs=[])
        rag._store._collection.count.return_value = 0
        assert rag.is_empty() is True

    def test_retrieve_calls_similarity_search(self):
        rag = self._make_rag()
        rag.retrieve("test query", k=3)
        rag._store.similarity_search.assert_called_once_with("test query", k=3)


class TestBuildIndex:
    """Test the build_index method."""

    def test_build_index_no_docs_returns_zero(self, tmp_path):
        mock_embeddings = MagicMock()
        mock_store = MagicMock()
        mock_store._collection.count.return_value = 0

        with (
            patch("finagent.rag.Chroma", return_value=mock_store),
            patch("finagent.rag.HuggingFaceEmbeddings", return_value=mock_embeddings),
            patch("finagent.rag.DirectoryLoader") as mock_loader_cls,
        ):
            mock_loader = MagicMock()
            mock_loader.load.return_value = []
            mock_loader_cls.return_value = mock_loader

            from finagent.rag import FinancialRAG
            rag = FinancialRAG.__new__(FinancialRAG)
            rag._embeddings = mock_embeddings
            rag._store = mock_store

            n = rag.build_index(tmp_path)
            assert n == 0

    def test_build_index_adds_documents(self, tmp_path):
        mock_embeddings = MagicMock()
        mock_store = MagicMock()
        mock_store._collection.count.return_value = 5

        fake_doc = MagicMock()
        fake_doc.page_content = "The federal funds rate is 4.50%."
        fake_doc.metadata = {"source": str(tmp_path / "test.txt")}

        with (
            patch("finagent.rag.Chroma", return_value=mock_store),
            patch("finagent.rag.HuggingFaceEmbeddings", return_value=mock_embeddings),
            patch("finagent.rag.DirectoryLoader") as mock_loader_cls,
            patch("finagent.rag.RecursiveCharacterTextSplitter") as mock_splitter_cls,
        ):
            mock_loader = MagicMock()
            mock_loader.load.return_value = [fake_doc]
            mock_loader_cls.return_value = mock_loader

            mock_splitter = MagicMock()
            mock_splitter.split_documents.return_value = [fake_doc] * 3
            mock_splitter_cls.return_value = mock_splitter

            from finagent.rag import FinancialRAG
            rag = FinancialRAG.__new__(FinancialRAG)
            rag._embeddings = mock_embeddings
            rag._store = mock_store

            n = rag.build_index(tmp_path)
            assert n == 3
            mock_store.add_documents.assert_called_once()


class TestFormatContext:
    """Focused tests for context formatting."""

    def test_single_chunk(self):
        # Import without constructing (avoids Chroma)
        from finagent.rag import FinancialRAG
        rag = object.__new__(FinancialRAG)
        chunks = [{"content": "Test content.", "source": "doc.txt"}]
        ctx = rag.format_context(chunks)
        assert "Source 1: doc.txt" in ctx
        assert "Test content." in ctx

    def test_multiple_chunks_separated(self):
        from finagent.rag import FinancialRAG
        rag = object.__new__(FinancialRAG)
        chunks = [
            {"content": "Content A.", "source": "a.txt"},
            {"content": "Content B.", "source": "b.txt"},
        ]
        ctx = rag.format_context(chunks)
        assert "Source 1: a.txt" in ctx
        assert "Source 2: b.txt" in ctx
        # Chunks are separated
        assert ctx.index("Source 1") < ctx.index("Source 2")
