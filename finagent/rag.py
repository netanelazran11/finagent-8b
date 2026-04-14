"""
Module 4 — RAG pipeline over local financial documents.

Architecture:
    data/financial_docs/*.txt
        → chunk (512 tokens, 64 overlap)
        → embed  (MiniLM-L6-v2, local, no API key)
        → store  (Chroma, persisted to data/chroma_db/)
        → retrieve (cosine similarity, top-k)
        → format_context (for the agent prompt)

Usage:
    # Build the index once
    rag = FinancialRAG.from_docs()

    # Retrieve for any query
    chunks = rag.retrieve("What is the current Fed funds rate?")
    context = rag.format_context(chunks)
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("finagent.rag")

# Paths
PERSIST_DIR = Path(__file__).parent.parent / "data" / "chroma_db"
DOCS_DIR = Path(__file__).parent.parent / "data" / "financial_docs"

# Embedding model — runs locally on CPU; no API key required
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking parameters
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
TOP_K = 4

# Module-level imports so patch("finagent.rag.Chroma", ...) works in tests.
# The try/except lets the module be imported even when langchain_chroma is
# not installed — unit tests mock these names before any real call.
try:
    from langchain_chroma import Chroma  # noqa: F401
    from langchain_community.document_loaders import DirectoryLoader, TextLoader  # noqa: F401
    from langchain_huggingface import HuggingFaceEmbeddings  # noqa: F401
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # noqa: F401
except ImportError:
    Chroma = None  # type: ignore[assignment,misc]
    HuggingFaceEmbeddings = None  # type: ignore[assignment,misc]
    DirectoryLoader = None  # type: ignore[assignment,misc]
    TextLoader = None  # type: ignore[assignment,misc]
    RecursiveCharacterTextSplitter = None  # type: ignore[assignment,misc]


class FinancialRAG:
    """Vector-search retrieval over a Chroma index of financial documents.

    The index is built once from data/financial_docs/ and persisted to
    data/chroma_db/. Subsequent runs load the persisted index — no re-embedding.

    Retrieval uses cosine similarity over sentence-transformers/all-MiniLM-L6-v2
    embeddings (90 MB download on first run; cached afterwards).
    """

    def __init__(self, persist_dir: Path = PERSIST_DIR) -> None:
        self._embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self._store = Chroma(
            collection_name="financial_docs",
            embedding_function=self._embeddings,
            persist_directory=str(persist_dir),
        )
        log.info("FinancialRAG ready (persist_dir=%s)", persist_dir)

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def build_index(self, docs_dir: Path = DOCS_DIR) -> int:
        """Chunk and embed all .txt documents in docs_dir.

        Returns the number of chunks added to the index.
        Safe to call multiple times — Chroma deduplicates by document ID.
        """
        loader = DirectoryLoader(
            str(docs_dir),
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=False,
        )
        raw_docs = loader.load()
        if not raw_docs:
            log.warning("No documents found in %s", docs_dir)
            return 0

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " "],
        )
        chunks = splitter.split_documents(raw_docs)

        # Store only the filename, not the full path, for readable citations
        for chunk in chunks:
            chunk.metadata["source"] = Path(chunk.metadata.get("source", "unknown")).name

        self._store.add_documents(chunks)
        log.info("Indexed %d chunks from %d documents", len(chunks), len(raw_docs))
        return len(chunks)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, k: int = TOP_K) -> list[dict[str, str]]:
        """Return top-k relevant chunks for a query.

        Returns a list of dicts: [{"content": str, "source": str}, ...]
        """
        docs = self._store.similarity_search(query, k=k)
        return [
            {"content": doc.page_content, "source": doc.metadata.get("source", "unknown")}
            for doc in docs
        ]

    def retrieve_with_scores(self, query: str, k: int = TOP_K) -> list[dict]:
        """Like retrieve() but includes cosine distance (lower = more similar)."""
        results = self._store.similarity_search_with_score(query, k=k)
        return [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "distance": round(float(score), 4),
            }
            for doc, score in results
        ]

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------

    def format_context(self, chunks: list[dict[str, str]]) -> str:
        """Format retrieved chunks into a context block for injection into a prompt.

        Example output:
            [Source 1: monetary_policy.txt]
            The federal funds rate is the interest rate...

            ---

            [Source 2: equity_valuation.txt]
            The P/E ratio is the most widely used...
        """
        if not chunks:
            return "No relevant context found in the knowledge base."
        parts = []
        for i, chunk in enumerate(chunks, 1):
            parts.append(f"[Source {i}: {chunk['source']}]\n{chunk['content']}")
        return "\n\n---\n\n".join(parts)

    def is_empty(self) -> bool:
        """True if the vector store has no documents yet."""
        return self._store._collection.count() == 0

    # ------------------------------------------------------------------
    # Convenience constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_docs(
        cls,
        docs_dir: Path = DOCS_DIR,
        persist_dir: Path = PERSIST_DIR,
    ) -> FinancialRAG:
        """Build a fresh index from docs_dir and return the FinancialRAG instance."""
        rag = cls(persist_dir=persist_dir)
        n = rag.build_index(docs_dir)
        log.info("from_docs: indexed %d chunks", n)
        return rag

    @classmethod
    def load(cls, persist_dir: Path = PERSIST_DIR) -> FinancialRAG:
        """Load a previously built index (faster — no re-embedding)."""
        rag = cls(persist_dir=persist_dir)
        if rag.is_empty():
            raise RuntimeError(
                f"No index found at {persist_dir}. Run `python scripts/build_index.py` first."
            )
        return rag
