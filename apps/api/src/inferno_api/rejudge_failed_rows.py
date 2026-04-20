from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from firecastbot.config import get_settings
from firecastbot.services.llm_service import LLMService
from inferno_api.rag_model_compare import (
    evaluate_correctness,
    evaluate_with_judge,
    load_json_records,
    write_csv,
)


def _load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _compact_text(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2].rstrip()
    tail = text[-(max_chars // 2) :].lstrip()
    return f"{head}\n\n[... truncated for compact re-judge ...]\n\n{tail}"


def _prompt_map(prompt_records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for record in prompt_records:
        key = str(record.get("id") or str(record["query"]).strip()[:40])
        mapping[key] = record
    return mapping


def _rejudge_rows(
    *,
    report: dict[str, Any],
    prompt_records: list[dict[str, Any]],
    judge_provider: str,
    judge_model: str,
    judge_temperature: float,
    judge_max_tokens: int,
    compact_mode: bool,
) -> dict[str, Any]:
    settings = get_settings()
    prompts_by_id = _prompt_map(prompt_records)
    judge_llm = None
    if judge_provider.strip().casefold() != "gemini":
        judge_llm = LLMService(
            settings,
            provider=judge_provider,
            model=judge_model,
            temperature=judge_temperature,
            max_tokens=judge_max_tokens,
        )

    for row in report.get("results", []):
        judge_error = str(row.get("judge_error") or "").strip()
        if not judge_error:
            continue

        prompt_id = str(row.get("prompt_id") or "")
        prompt_record = prompts_by_id.get(prompt_id)
        if prompt_record is None:
            raise KeyError(f"Could not find prompt record for prompt_id={prompt_id!r}")

        rag_system_prompt = str(row.get("rag_system_prompt") or "")
        rag_user_prompt = str(row.get("rag_user_prompt") or "")
        reply = str(row.get("reply") or "")
        if compact_mode:
            rag_system_prompt = _compact_text(rag_system_prompt, 600)
            rag_user_prompt = _compact_text(rag_user_prompt, 1800)
            reply = _compact_text(reply, 2200)
        rag_messages = [
            {"role": "system", "content": rag_system_prompt},
            {"role": "user", "content": rag_user_prompt},
        ]
        retrieval_sources = json.loads(str(row.get("retrieval_sources") or "[]"))

        judge_result = evaluate_with_judge(
            settings=settings,
            judge_llm=judge_llm,
            judge_provider=judge_provider,
            judge_model=judge_model,
            judge_temperature=judge_temperature,
            judge_max_tokens=judge_max_tokens,
            query=str(row.get("query") or ""),
            prompt_record=prompt_record,
            rag_messages=rag_messages,
            rag_reply=reply,
            retrieval_sources=retrieval_sources,
        )

        keyword_correctness = evaluate_correctness(reply, prompt_record)
        matched_expected = (
            judge_result["matched_expected"] or keyword_correctness["matched_expected"]
        )

        row["correctness_score"] = judge_result["correctness_score"]
        row["groundedness_score"] = judge_result["groundedness_score"]
        row["instruction_following_score"] = judge_result["instruction_following_score"]
        row["matched_expected_count"] = len(matched_expected)
        row["matched_expected"] = json.dumps(matched_expected, ensure_ascii=True)
        row["judge_latency_ms"] = judge_result["judge_latency_ms"]
        row["judge_error"] = judge_result["judge_error"]
        row["judge_rationale"] = judge_result["judge_rationale"]
        row["judge_system_prompt"] = judge_result["judge_system_prompt"]
        row["judge_user_prompt"] = judge_result["judge_user_prompt"]
        row["judge_reply"] = judge_result["judge_reply"]
        row["judge_reply_char_count"] = len(judge_result["judge_reply"])
        row["judge_reply_word_count"] = len(judge_result["judge_reply"].split())
        row["judge_response_id"] = judge_result["judge_response_id"]
        row["judge_finish_reason"] = judge_result["judge_finish_reason"]
        row["judge_prompt_tokens"] = judge_result["judge_prompt_tokens"]
        row["judge_completion_tokens"] = judge_result["judge_completion_tokens"]
        row["judge_total_tokens"] = judge_result["judge_total_tokens"]
        row["judge_reasoning_tokens"] = judge_result["judge_reasoning_tokens"]
        row["judge_cached_tokens"] = judge_result["judge_cached_tokens"]
        row["judge_response_metadata"] = judge_result["judge_response_metadata"]

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-run Gemini judging only for rows that currently have judge errors."
    )
    parser.add_argument(
        "--report-json",
        default="apps/api/evals/rag_model_compare_openrouter_paid_gemini_judge.json",
        help="Path to the evaluation JSON report to patch in place.",
    )
    parser.add_argument(
        "--report-csv",
        default="apps/api/evals/rag_model_compare_openrouter_paid_gemini_judge.csv",
        help="Path to the evaluation CSV report to patch in place.",
    )
    parser.add_argument(
        "--prompts",
        default="apps/api/evals/prompts.sample.json",
        help="Path to the prompt records JSON/JSONL used to generate the report.",
    )
    parser.add_argument(
        "--judge-provider",
        default="gemini",
        help="Judge provider.",
    )
    parser.add_argument(
        "--judge-model",
        default="gemini-2.5-flash",
        help="Judge model.",
    )
    parser.add_argument(
        "--judge-temperature",
        type=float,
        default=0.0,
        help="Judge temperature.",
    )
    parser.add_argument(
        "--judge-max-tokens",
        type=int,
        default=1200,
        help="Judge max tokens.",
    )
    parser.add_argument(
        "--compact-mode",
        action="store_true",
        default=True,
        help="Use compact judge inputs for failed-row retries.",
    )
    args = parser.parse_args()

    report_json = Path(args.report_json)
    report_csv = Path(args.report_csv)
    prompt_records = load_json_records(Path(args.prompts))
    report = _load_report(report_json)
    patched = _rejudge_rows(
        report=report,
        prompt_records=prompt_records,
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
        judge_temperature=args.judge_temperature,
        judge_max_tokens=args.judge_max_tokens,
        compact_mode=args.compact_mode,
    )

    report_json.write_text(json.dumps(patched, indent=2), encoding="utf-8")
    write_csv(patched["results"], report_csv)
    print(f"Patched judge failures in {report_json} and {report_csv}.")


if __name__ == "__main__":
    main()
