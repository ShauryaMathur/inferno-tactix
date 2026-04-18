from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from firecastbot.config import get_settings
from firecastbot.services.llm_service import LLMService
from inferno_api.rag_model_compare import _result_metrics
from inferno_api.rag_model_compare import write_csv
from inferno_api.rag_quality_eval import build_quality_judge_messages, evaluate_with_groq_quality_judge, _extract_json_object
from inferno_api.rag_model_compare import evaluate_with_gemini_judge_rest


def _load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _case_map(case_records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(record["id"]): record for record in case_records}


def _load_cases(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return data


def _rejudge_rows(
    *,
    report: dict[str, Any],
    case_records: list[dict[str, Any]],
    judge_provider: str,
    judge_model: str,
    judge_temperature: float,
    judge_max_tokens: int,
) -> dict[str, Any]:
    settings = get_settings()
    judge_llm = None
    if judge_provider.strip().casefold() != "gemini":
        judge_llm = LLMService(
            settings,
            provider=judge_provider,
            model=judge_model,
            temperature=judge_temperature,
            max_tokens=judge_max_tokens,
        )
    cases_by_id = _case_map(case_records)

    for row in report.get("rows", []):
        if not str(row.get("judge_error") or "").strip():
            continue

        case_id = str(row.get("case_id") or "")
        case_record = cases_by_id.get(case_id)
        if case_record is None:
            raise KeyError(f"Could not find case record for case_id={case_id!r}")

        rag_messages = [
            {"role": "system", "content": str(row.get("rag_system_prompt") or "")},
            {"role": "user", "content": str(row.get("rag_user_prompt") or "")},
        ]
        rag_reply = str(row.get("reply") or "")
        retrieval_sources = json.loads(str(row.get("retrieval_sources") or "[]"))
        if judge_provider.strip().casefold() == "gemini":
            judge_messages = build_quality_judge_messages(
                case_record=case_record,
                rag_messages=rag_messages,
                rag_reply=rag_reply,
                retrieval_sources=retrieval_sources,
            )
            try:
                completion, parsed = evaluate_with_gemini_judge_rest(
                    api_key=settings.require_api_key("gemini"),
                    judge_model=judge_model,
                    judge_temperature=judge_temperature,
                    judge_max_tokens=judge_max_tokens,
                    judge_messages=judge_messages,
                )
                judge_result = {
                    "judge_system_prompt": judge_messages[0]["content"],
                    "judge_user_prompt": judge_messages[1]["content"],
                    "judge_reply": completion.content,
                    "judge_error": str(parsed.get("error") or ""),
                    "judge_latency_ms": None,
                    "correctness_score": float(parsed.get("correctness_score")) if parsed.get("correctness_score") is not None else None,
                    "faithfulness_score": float(parsed.get("faithfulness_score")) if parsed.get("faithfulness_score") is not None else None,
                    "refusal_compliance_score": float(parsed.get("refusal_compliance_score")) if parsed.get("refusal_compliance_score") is not None else None,
                    "multi_step_reasoning_score": float(parsed.get("multi_step_reasoning_score")) if parsed.get("multi_step_reasoning_score") is not None else None,
                    "judge_rationale": str(parsed.get("rationale") or ""),
                    **_result_metrics("judge", completion),
                }
            except Exception as exc:
                judge_result = {
                    "judge_system_prompt": judge_messages[0]["content"],
                    "judge_user_prompt": judge_messages[1]["content"],
                    "judge_reply": "",
                    "judge_error": f"{type(exc).__name__}: {exc}",
                    "judge_latency_ms": None,
                    "correctness_score": None,
                    "faithfulness_score": None,
                    "refusal_compliance_score": None,
                    "multi_step_reasoning_score": None,
                    "judge_rationale": "",
                    **_result_metrics("judge", None),
                }
        else:
            judge_result = evaluate_with_groq_quality_judge(
                judge_llm=judge_llm,
                case_record=case_record,
                rag_messages=rag_messages,
                rag_reply=rag_reply,
                retrieval_sources=retrieval_sources,
            )

        row["judge_provider"] = judge_provider
        row["judge_model"] = judge_model
        row["judge_latency_ms"] = judge_result["judge_latency_ms"]
        row["judge_error"] = judge_result["judge_error"]
        row["judge_rationale"] = judge_result["judge_rationale"]
        row["judge_system_prompt"] = judge_result["judge_system_prompt"]
        row["judge_user_prompt"] = judge_result["judge_user_prompt"]
        row["judge_reply"] = judge_result["judge_reply"]
        row["correctness_score"] = judge_result["correctness_score"]
        row["faithfulness_score"] = judge_result["faithfulness_score"]
        row["refusal_compliance_score"] = judge_result["refusal_compliance_score"]
        row["multi_step_reasoning_score"] = judge_result["multi_step_reasoning_score"]
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
    parser = argparse.ArgumentParser(description="Rejudge only failed rows in the strict quality eval artifacts.")
    parser.add_argument(
        "--report-json",
        default="apps/api/evals/rag_quality_eval_top4_groq_judge.json",
        help="Path to the quality-eval JSON report to patch in place.",
    )
    parser.add_argument(
        "--report-csv",
        default="apps/api/evals/rag_quality_eval_top4_groq_judge.csv",
        help="Path to the quality-eval CSV report to patch in place.",
    )
    parser.add_argument(
        "--cases",
        default="apps/api/evals/quality_cases.strict.json",
        help="Path to the quality case definitions.",
    )
    parser.add_argument("--judge-provider", default="groq")
    parser.add_argument("--judge-model", default="llama-3.3-70b-versatile")
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-max-tokens", type=int, default=900)
    args = parser.parse_args()

    report_json = Path(args.report_json)
    report_csv = Path(args.report_csv)
    report = _load_report(report_json)
    case_records = _load_cases(Path(args.cases))
    patched = _rejudge_rows(
        report=report,
        case_records=case_records,
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
        judge_temperature=args.judge_temperature,
        judge_max_tokens=args.judge_max_tokens,
    )

    report_json.write_text(json.dumps(patched, indent=2), encoding="utf-8")
    write_csv(patched["rows"], report_csv)
    print(f"Patched quality-eval judge failures in {report_json} and {report_csv}.")


if __name__ == "__main__":
    main()
