from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent
APP_ROOT = PACKAGE_ROOT.parent.parent
REPO_ROOT = APP_ROOT.parent.parent
APPS_ROOT = REPO_ROOT / "apps"

if str(APPS_ROOT) not in sys.path:
    sys.path.insert(0, str(APPS_ROOT))

from inferno_api.firecastbot_runtime import classify_query, load_doctrine_assets, retrieve_chunks
from inferno_api.rag_model_compare import load_json_records, write_csv


def _norm(text: str) -> str:
    normalized = text.casefold()
    for source, target in {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
    }.items():
        normalized = normalized.replace(source, target)
    return " ".join(normalized.split())


def _matches(chunk: dict[str, Any], matcher: dict[str, Any]) -> bool:
    if matcher.get("doc_title") and str(chunk.get("doc_title") or "") != str(matcher["doc_title"]):
        return False
    if matcher.get("section_contains") and _norm(str(matcher["section_contains"])) not in _norm(
        str(chunk.get("section_path") or chunk.get("section") or "")
    ):
        return False
    if matcher.get("text_contains") and _norm(str(matcher["text_contains"])) not in _norm(str(chunk.get("text") or "")):
        return False
    return True


def _relevant_chunk_ids(chunks: list[dict[str, Any]], matchers: list[dict[str, Any]]) -> set[str]:
    relevant: set[str] = set()
    for chunk in chunks:
        for matcher in matchers:
            if _matches(chunk, matcher):
                relevant.add(str(chunk["chunk_id"]))
                break
    return relevant


def run_retrieval_eval(*, doctrine_manifest: Path, case_records: list[dict[str, Any]], k: int) -> dict[str, Any]:
    doctrine_store = load_doctrine_assets(doctrine_manifest)
    rows: list[dict[str, Any]] = []
    for case in case_records:
        query = str(case["query"]).strip()
        query_class = str(case.get("query_class") or classify_query(query))
        relevant_ids = _relevant_chunk_ids(doctrine_store["chunks"], list(case["expected_matchers"]))
        retrieved = retrieve_chunks(
            query,
            chunks=doctrine_store["chunks"],
            embeddings=doctrine_store["embeddings"],
            keyword_index=doctrine_store["keyword_index"],
            embedding_provider=doctrine_store["embedding_provider"],
            model_name=doctrine_store["embedding_model"],
            limit=k,
        )
        retrieved_ids = [str(item["chunk_id"]) for item in retrieved]
        relevant_retrieved = [chunk_id for chunk_id in retrieved_ids if chunk_id in relevant_ids]
        precision_at_k = round(len(relevant_retrieved) / max(len(retrieved_ids), 1), 4)
        recall_at_k = round(len(relevant_retrieved) / max(len(relevant_ids), 1), 4)
        hit_at_k = 1 if relevant_retrieved else 0
        reciprocal_rank = 0.0
        for idx, chunk_id in enumerate(retrieved_ids, start=1):
            if chunk_id in relevant_ids:
                reciprocal_rank = round(1.0 / idx, 4)
                break
        rows.append(
            {
                "case_id": str(case["id"]),
                "query": query,
                "query_class": query_class,
                "k": k,
                "relevant_chunk_count": len(relevant_ids),
                "retrieved_count": len(retrieved_ids),
                "relevant_retrieved_count": len(relevant_retrieved),
                "precision_at_k": precision_at_k,
                "recall_at_k": recall_at_k,
                "hit_at_k": hit_at_k,
                "reciprocal_rank": reciprocal_rank,
                "expected_matchers": json.dumps(case["expected_matchers"], ensure_ascii=True),
                "relevant_chunk_ids": json.dumps(sorted(relevant_ids), ensure_ascii=True),
                "retrieved_chunk_ids": json.dumps(retrieved_ids, ensure_ascii=True),
                "retrieved_sections": json.dumps(
                    [
                        {
                            "chunk_id": item["chunk_id"],
                            "doc_title": item.get("doc_title"),
                            "section": item.get("section_path") or item.get("section"),
                            "retrieval_score": item.get("retrieval_score"),
                        }
                        for item in retrieved
                    ],
                    ensure_ascii=True,
                ),
            }
        )
    summary = {
        "case_count": len(rows),
        "precision_at_k_avg": round(sum(float(row["precision_at_k"]) for row in rows) / max(len(rows), 1), 4),
        "recall_at_k_avg": round(sum(float(row["recall_at_k"]) for row in rows) / max(len(rows), 1), 4),
        "hit_at_k_avg": round(sum(int(row["hit_at_k"]) for row in rows) / max(len(rows), 1), 4),
        "mrr": round(sum(float(row["reciprocal_rank"]) for row in rows) / max(len(rows), 1), 4),
        "k": k,
        "embedding_model": doctrine_store["embedding_model"],
    }
    return {"summary": summary, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate doctrine retrieval precision and recall against curated handbook targets.")
    parser.add_argument("--cases", default="apps/api/evals/retrieval_cases.doctrine.json")
    parser.add_argument("--doctrine-manifest", default="apps/firecastbot/incident_response_docs/doctrine_retrieval_manifest.bge_base.json")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output-json", default="apps/api/evals/retrieval_eval_bge_base.json")
    parser.add_argument("--output-csv", default="apps/api/evals/retrieval_eval_bge_base.csv")
    args = parser.parse_args()

    case_records = load_json_records(Path(args.cases))
    report = run_retrieval_eval(doctrine_manifest=Path(args.doctrine_manifest), case_records=case_records, k=args.k)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(report["rows"], Path(args.output_csv))
    print(f"Wrote {len(report['rows'])} retrieval eval rows to {output_json} and {args.output_csv}.")


if __name__ == "__main__":
    main()
