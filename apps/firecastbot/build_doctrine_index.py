from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent
APPS_ROOT = PACKAGE_ROOT.parent
if str(APPS_ROOT) not in sys.path:
    sys.path.insert(0, str(APPS_ROOT))

from firecastbot.config import Settings
from firecastbot.services.embedder_service import EmbedderService

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")


TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9+/#.&'-]*")


def load_chunks(jsonl_path: Path) -> list[dict[str, object]]:
    chunks: list[dict[str, object]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def normalize_token(token: str) -> str:
    return token.casefold().strip("._-")


def tokenize(text: str) -> list[str]:
    return [
        normalized
        for raw in TOKEN_PATTERN.findall(text)
        if (normalized := normalize_token(raw))
    ]


def build_keyword_index(chunks: list[dict[str, object]]) -> dict[str, object]:
    postings: dict[str, list[dict[str, object]]] = defaultdict(list)
    documents: list[dict[str, object]] = []
    doc_lengths: list[int] = []

    for chunk in chunks:
        text = str(chunk["text"])
        tokens = tokenize(text)
        frequencies = Counter(tokens)
        doc_length = len(tokens)
        doc_lengths.append(doc_length)
        documents.append(
            {
                "chunk_id": chunk["chunk_id"],
                "doc_title": chunk["doc_title"],
                "section_path": chunk["section_path"],
                "text": text,
                "metadata": {
                    key: value
                    for key, value in chunk.items()
                    if key not in {"text"}
                },
                "length": doc_length,
            }
        )
        for term, tf in frequencies.items():
            postings[term].append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "tf": tf,
                }
            )

    doc_count = len(chunks)
    avgdl = (sum(doc_lengths) / doc_count) if doc_count else 0.0
    vocabulary = {
        term: {
            "df": len(term_postings),
            "idf": math.log(1 + ((doc_count - len(term_postings) + 0.5) / (len(term_postings) + 0.5))) if doc_count else 0.0,
            "postings": term_postings,
        }
        for term, term_postings in postings.items()
    }
    return {
        "index_type": "bm25",
        "tokenizer": "regex_casefold_acronym_preserving",
        "bm25": {
            "k1": 1.5,
            "b": 0.75,
            "avgdl": avgdl,
        },
        "document_count": doc_count,
        "documents": documents,
        "vocabulary": vocabulary,
    }


def build_embeddings(
    chunks: list[dict[str, object]],
    embeddings_path: Path,
    metadata_path: Path,
    provider: str,
    model_name: str,
) -> dict[str, object]:
    texts = [str(chunk["text"]) for chunk in chunks]
    embedder = EmbedderService(
        Settings(embedding_provider=provider, embedding_model=model_name),
        provider=provider,
        model=model_name,
    )
    vectors = embedder.embed_texts(texts).astype(np.float32)
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, vectors)

    with metadata_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            record = {
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "metadata": {
                    key: value
                    for key, value in chunk.items()
                    if key not in {"text", "chunk_id"}
                },
            }
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    return {
        "embedding_provider": provider,
        "embedding_model": model_name,
        "embedding_dtype": "float32",
        "vector_count": int(vectors.shape[0]),
        "vector_dim": int(vectors.shape[1]) if vectors.ndim == 2 else 0,
        "vectors_path": str(embeddings_path),
        "metadata_path": str(metadata_path),
    }


def write_json(data: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline dense and keyword retrieval artifacts for doctrine chunks.")
    parser.add_argument(
        "--input",
        default="apps/firecastbot/incident_response_docs/doctrine_kb.jsonl",
        help="Path to doctrine KB JSONL.",
    )
    parser.add_argument(
        "--keyword-output",
        default="apps/firecastbot/incident_response_docs/doctrine_keyword_index.json",
        help="Path to keyword/BM25 index JSON.",
    )
    parser.add_argument(
        "--embeddings-output",
        default="apps/firecastbot/incident_response_docs/doctrine_embeddings.npy",
        help="Path to dense embedding matrix (.npy).",
    )
    parser.add_argument(
        "--embedding-metadata-output",
        default="apps/firecastbot/incident_response_docs/doctrine_embeddings.meta.jsonl",
        help="Path to embedding metadata JSONL.",
    )
    parser.add_argument(
        "--manifest-output",
        default="apps/firecastbot/incident_response_docs/doctrine_retrieval_manifest.json",
        help="Path to retrieval manifest JSON.",
    )
    parser.add_argument(
        "--embedding-provider",
        default="sentence-transformers",
        help="Embedding provider to use for dense retrieval.",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model to use for dense retrieval.",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Build only the keyword index and manifest.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    chunks = load_chunks(input_path)

    keyword_index = build_keyword_index(chunks)
    keyword_output = Path(args.keyword_output)
    write_json(keyword_index, keyword_output)

    manifest: dict[str, object] = {
        "source_kb": str(input_path),
        "chunk_count": len(chunks),
        "keyword_index": str(keyword_output),
        "dense_index": None,
    }

    if not args.skip_embeddings:
        embedding_info = build_embeddings(
            chunks,
            Path(args.embeddings_output),
            Path(args.embedding_metadata_output),
            args.embedding_provider,
            args.embedding_model,
        )
        manifest["dense_index"] = embedding_info

    write_json(manifest, Path(args.manifest_output))
    print(f"Built keyword index for {len(chunks)} doctrine chunks.")
    if args.skip_embeddings:
        print("Skipped dense embeddings.")


if __name__ == "__main__":
    main()
