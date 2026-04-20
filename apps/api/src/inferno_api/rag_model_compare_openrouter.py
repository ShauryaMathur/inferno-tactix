from __future__ import annotations

import argparse
import json
from pathlib import Path

from firecastbot.config import get_settings
from inferno_api.rag_model_compare import load_json_records, run_evaluation, write_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare multiple free OpenRouter models on the same FireCastBot RAG prompts."
    )
    parser.add_argument(
        "--prompts",
        default="apps/api/evals/prompts.sample.json",
        help="Path to prompt records JSON/JSONL. Each item needs at least a query field.",
    )
    parser.add_argument(
        "--models",
        default="apps/api/evals/models.openrouter.free.json",
        help="Path to model records JSON/JSONL. Each item needs provider and model.",
    )
    parser.add_argument(
        "--incident-report",
        help="Optional path to an incident report PDF to include runtime incident retrieval.",
    )
    parser.add_argument(
        "--doctrine-manifest",
        default="apps/firecastbot/incident_response_docs/doctrine_retrieval_manifest.json",
        help="Path to doctrine retrieval manifest.",
    )
    parser.add_argument(
        "--output-json",
        default="apps/api/evals/rag_model_compare_openrouter.json",
        help="Path to write the evaluation JSON report.",
    )
    parser.add_argument(
        "--output-csv",
        default="apps/api/evals/rag_model_compare_openrouter.csv",
        help="Path to write the flattened evaluation CSV report.",
    )
    args = parser.parse_args()

    settings = get_settings()
    prompt_records = load_json_records(Path(args.prompts))
    model_records = load_json_records(Path(args.models))
    report = run_evaluation(
        settings=settings,
        doctrine_manifest=Path(args.doctrine_manifest),
        prompt_records=prompt_records,
        model_records=model_records,
        incident_report_path=Path(args.incident_report) if args.incident_report else None,
    )

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(report["results"], Path(args.output_csv))
    print(f"Wrote {report['result_count']} evaluation rows to {output_json} and {args.output_csv}.")


if __name__ == "__main__":
    main()
