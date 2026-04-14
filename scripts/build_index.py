#!/usr/bin/env python3
"""
Module 4 — Build the Chroma vector index from financial documents.

Run once before using the RAG agent or multi-agent system:
    python scripts/build_index.py

The index is persisted to data/chroma_db/ and reloaded automatically on subsequent runs.
Re-running this script rebuilds the index from scratch.

Documents indexed: data/financial_docs/*.txt
Embedding model:   sentence-transformers/all-MiniLM-L6-v2  (local, no API key)
Vector store:      Chroma (persisted locally)
"""

from __future__ import annotations

import argparse
import logging
import shutil
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_index")

ROOT = Path(__file__).parent.parent
DOCS_DIR = ROOT / "data" / "financial_docs"
PERSIST_DIR = ROOT / "data" / "chroma_db"


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Chroma RAG index from financial documents.")
    ap.add_argument("--docs-dir", default=str(DOCS_DIR), help="Directory of .txt documents")
    ap.add_argument("--persist-dir", default=str(PERSIST_DIR), help="Chroma persist directory")
    ap.add_argument("--reset", action="store_true", help="Delete existing index and rebuild")
    args = ap.parse_args()

    docs_dir = Path(args.docs_dir)
    persist_dir = Path(args.persist_dir)

    if not docs_dir.exists():
        log.error("Documents directory not found: %s", docs_dir)
        return

    doc_files = list(docs_dir.glob("**/*.txt"))
    if not doc_files:
        log.error("No .txt files found in %s", docs_dir)
        return

    log.info("Found %d document(s) to index:", len(doc_files))
    for f in doc_files:
        log.info("  %s", f.name)

    if args.reset and persist_dir.exists():
        log.info("Resetting index at %s", persist_dir)
        shutil.rmtree(persist_dir)

    persist_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading embedding model (first run downloads ~90 MB)...")
    from finagent.rag import FinancialRAG

    t0 = time.time()
    rag = FinancialRAG(persist_dir=persist_dir)
    n_chunks = rag.build_index(docs_dir=docs_dir)
    elapsed = round(time.time() - t0, 1)

    print()
    print("=" * 55)
    print(f"  Index built successfully")
    print(f"  Documents:    {len(doc_files)}")
    print(f"  Chunks:       {n_chunks}")
    print(f"  Persist dir:  {persist_dir}")
    print(f"  Time:         {elapsed}s")
    print("=" * 55)
    print()
    print("Test a retrieval query:")

    test_query = "What is the current federal funds rate?"
    chunks = rag.retrieve(test_query, k=2)
    print(f"  Query: {test_query!r}")
    for i, chunk in enumerate(chunks, 1):
        preview = chunk["content"][:120].replace("\n", " ")
        print(f"  [{i}] ({chunk['source']}) {preview}...")


if __name__ == "__main__":
    main()
