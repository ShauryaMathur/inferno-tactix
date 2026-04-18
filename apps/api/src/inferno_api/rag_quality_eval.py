from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent
APP_ROOT = PACKAGE_ROOT.parent.parent
REPO_ROOT = APP_ROOT.parent.parent
APPS_ROOT = REPO_ROOT / "apps"

if str(APPS_ROOT) not in sys.path:
    sys.path.insert(0, str(APPS_ROOT))

from firecastbot.config import get_settings
from firecastbot.services.llm_service import ChatCompletionResult, LLMService
from inferno_api.firecastbot_runtime import build_runtime_incident_report, classify_query, load_doctrine_assets, retrieve_firecast_context
from inferno_api.rag_model_compare import _is_rate_limit_error, _result_metrics, _text_metrics, load_json_records, normalize_settings, write_csv
from inferno_api.rag_model_compare_upgraded import build_messages_upgraded


def _extract_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        candidate = candidate.replace("json\n", "", 1).strip()
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Judge response did not contain a JSON object.")
    payload = json.loads(candidate[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Judge response JSON was not an object.")
    return payload


def build_quality_judge_messages(
    *,
    case_record: dict[str, Any],
    rag_messages: list[dict[str, str]],
    rag_reply: str,
    retrieval_sources: list[dict[str, Any]],
) -> list[dict[str, str]]:
    schema = {
        "correctness_score": "number in [0,1]",
        "faithfulness_score": "number in [0,1]",
        "refusal_compliance_score": "number in [0,1]",
        "multi_step_reasoning_score": "number in [0,1]",
        "error": "empty string unless evaluation cannot be completed",
        "rationale": "one short sentence",
    }
    system = (
        "You are a strict judge for wildfire RAG evaluation. "
        "Penalize unsupported claims, weak refusals, and shallow reasoning. Return JSON only."
    )
    user = (
        f"Query:\n{case_record['query']}\n\n"
        f"Rubric:\n{case_record['rubric']}\n\n"
        f"Reference answer:\n{case_record['reference_answer']}\n\n"
        f"Expected key phrases:\n{json.dumps(case_record.get('expected_contains', []), ensure_ascii=True)}\n\n"
        f"Refusal expected:\n{bool(case_record.get('refusal_expected'))}\n\n"
        f"Multi-step reasoning expected:\n{bool(case_record.get('multi_step_expected'))}\n\n"
        f"Retrieved sources:\n{json.dumps(retrieval_sources, ensure_ascii=True)}\n\n"
        f"RAG system prompt:\n{rag_messages[0]['content'][:1800]}\n\n"
        f"RAG user prompt:\n{rag_messages[1]['content'][:3200]}\n\n"
        f"Assistant answer:\n{rag_reply[:3200]}\n\n"
        "Scoring guidance:\n"
        "- correctness_score: does the answer satisfy the task and rubric?\n"
        "- faithfulness_score: does it stay within retrieved evidence and refuse unsupported leaps?\n"
        "- refusal_compliance_score: when refusal is expected, does it clearly refuse to guess?\n"
        "- multi_step_reasoning_score: when multi-step reasoning is expected, does it combine evidence into a coherent, cautious chain of reasoning?\n"
        f"Return JSON matching this schema:\n{json.dumps(schema, ensure_ascii=True)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def evaluate_with_groq_quality_judge(
    *,
    judge_llm: LLMService,
    case_record: dict[str, Any],
    rag_messages: list[dict[str, str]],
    rag_reply: str,
    retrieval_sources: list[dict[str, Any]],
) -> dict[str, Any]:
    judge_messages = build_quality_judge_messages(
        case_record=case_record,
        rag_messages=rag_messages,
        rag_reply=rag_reply,
        retrieval_sources=retrieval_sources,
    )
    completion: ChatCompletionResult | None = None
    reply = ""
    error = ""
    parsed: dict[str, Any] = {}
    started = time.perf_counter()
    try:
        completion = judge_llm.chat_completion_result(judge_messages)
        reply = completion.content
        parsed = _extract_json_object(reply)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    latency_ms = round((time.perf_counter() - started) * 1000, 2)

    def _score(key: str) -> float | None:
        value = parsed.get(key)
        if value in {None, ""}:
            return None
        try:
            return round(float(value), 4)
        except (TypeError, ValueError):
            return None

    return {
        "judge_system_prompt": judge_messages[0]["content"],
        "judge_user_prompt": judge_messages[1]["content"],
        "judge_reply": reply,
        "judge_error": error or str(parsed.get("error") or ""),
        "judge_latency_ms": latency_ms,
        "correctness_score": _score("correctness_score"),
        "faithfulness_score": _score("faithfulness_score"),
        "refusal_compliance_score": _score("refusal_compliance_score"),
        "multi_step_reasoning_score": _score("multi_step_reasoning_score"),
        "judge_rationale": str(parsed.get("rationale") or ""),
        **_result_metrics("judge", completion),
    }


def run_quality_eval(
    *,
    case_records: list[dict[str, Any]],
    model_records: list[dict[str, Any]],
    doctrine_manifest: Path,
    incident_report: Path,
    judge_provider: str,
    judge_model: str,
    judge_temperature: float,
    judge_max_tokens: int,
    retrieval_k: int,
    fail_on_rate_limit: bool,
) -> dict[str, Any]:
    settings = get_settings()
    doctrine_store = load_doctrine_assets(doctrine_manifest)
    incident_bundle = build_runtime_incident_report(
        incident_report.read_bytes(),
        incident_report.name,
        settings.embedding_provider,
        settings.embedding_model,
    )
    judge_llm = LLMService(
        settings,
        provider=judge_provider,
        model=judge_model,
        temperature=judge_temperature,
        max_tokens=judge_max_tokens,
    )
    rows: list[dict[str, Any]] = []
    started_at = time.time()

    for model_record in model_records:
        effective_settings = normalize_settings(
            settings,
            {
                "llm_provider": model_record["provider"],
                "llm_model": model_record["model"],
                "llm_temperature": model_record.get("temperature", settings.llm_temperature),
                "llm_max_tokens": model_record.get("max_tokens", settings.llm_max_tokens),
            },
        )
        llm = LLMService(
            effective_settings,
            provider=effective_settings.llm_provider,
            model=effective_settings.llm_model,
            temperature=effective_settings.llm_temperature,
            max_tokens=effective_settings.llm_max_tokens,
        )
        for case in case_records:
            query = str(case["query"]).strip()
            query_class = str(case.get("query_class") or classify_query(query))
            use_incident = bool(case.get("use_incident_report", True))
            context_items = retrieve_firecast_context(
                query,
                query_class=query_class,
                retrieval_k=retrieval_k,
                incident_profile=incident_bundle["incident_profile"] if use_incident else None,
                incident_chunks=incident_bundle["incident_chunks"] if use_incident else [],
                incident_embeddings=incident_bundle["incident_embeddings"] if use_incident else None,
                incident_keyword_index=incident_bundle["incident_keyword_index"] if use_incident else None,
                incident_embedding_provider=str(incident_bundle["embedding_provider"]) if use_incident else settings.embedding_provider,
                incident_embedding_model=str(incident_bundle["embedding_model"]) if use_incident else settings.embedding_model,
                doctrine_store=doctrine_store,
            )
            rag_messages = build_messages_upgraded(
                query=query,
                query_class=query_class,
                incident_profile=incident_bundle["incident_profile"] if use_incident else None,
                context_items=context_items,
            )
            retrieval_sources = [
                {
                    "source_type": item.get("source_type"),
                    "section": item.get("section") or item.get("section_path") or item.get("label"),
                    "chunk_id": item.get("chunk_id") or item.get("fact_key"),
                }
                for item in context_items
            ]
            started = time.perf_counter()
            completion: ChatCompletionResult | None = None
            reply = ""
            error = ""
            try:
                completion = llm.chat_completion_result(rag_messages)
                reply = completion.content
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                if fail_on_rate_limit and _is_rate_limit_error(error):
                    raise RuntimeError(f"Rate limit error for {effective_settings.llm_provider}:{effective_settings.llm_model}: {error}") from exc
            latency_ms = round((time.perf_counter() - started) * 1000, 2)

            judge_result = {
                "judge_system_prompt": "",
                "judge_user_prompt": "",
                "judge_reply": "",
                "judge_error": "",
                "judge_latency_ms": None,
                "correctness_score": None,
                "faithfulness_score": None,
                "refusal_compliance_score": None,
                "multi_step_reasoning_score": None,
                "judge_rationale": "",
                **_result_metrics("judge", None),
            }
            if not error and reply.strip():
                judge_result = evaluate_with_groq_quality_judge(
                    judge_llm=judge_llm,
                    case_record=case,
                    rag_messages=rag_messages,
                    rag_reply=reply,
                    retrieval_sources=retrieval_sources,
                )
            query_metrics = _text_metrics(query)
            reply_metrics = _text_metrics(reply)
            rows.append(
                {
                    "case_id": str(case["id"]),
                    "label": str(model_record.get("label") or f"{effective_settings.llm_provider}:{effective_settings.llm_model}"),
                    "provider": effective_settings.llm_provider,
                    "model": effective_settings.llm_model,
                    "query": query,
                    "query_class": query_class,
                    "use_incident_report": use_incident,
                    "latency_ms": latency_ms,
                    "correctness_score": judge_result["correctness_score"],
                    "faithfulness_score": judge_result["faithfulness_score"],
                    "refusal_compliance_score": judge_result["refusal_compliance_score"],
                    "multi_step_reasoning_score": judge_result["multi_step_reasoning_score"],
                    "reply": reply,
                    "error": error,
                    "retrieval_context_count": len(context_items),
                    "retrieval_sources": json.dumps(retrieval_sources, ensure_ascii=True),
                    "rag_system_prompt": rag_messages[0]["content"],
                    "rag_user_prompt": rag_messages[1]["content"],
                    "judge_provider": judge_provider,
                    "judge_model": judge_model,
                    "judge_latency_ms": judge_result["judge_latency_ms"],
                    "judge_error": judge_result["judge_error"],
                    "judge_rationale": judge_result["judge_rationale"],
                    "judge_system_prompt": judge_result["judge_system_prompt"],
                    "judge_user_prompt": judge_result["judge_user_prompt"],
                    "judge_reply": judge_result["judge_reply"],
                    "query_char_count": query_metrics["char_count"],
                    "query_word_count": query_metrics["word_count"],
                    "reply_char_count": reply_metrics["char_count"],
                    "reply_word_count": reply_metrics["word_count"],
                    **_result_metrics("model", completion),
                    "judge_response_id": judge_result["judge_response_id"],
                    "judge_finish_reason": judge_result["judge_finish_reason"],
                    "judge_prompt_tokens": judge_result["judge_prompt_tokens"],
                    "judge_completion_tokens": judge_result["judge_completion_tokens"],
                    "judge_total_tokens": judge_result["judge_total_tokens"],
                    "judge_reasoning_tokens": judge_result["judge_reasoning_tokens"],
                    "judge_cached_tokens": judge_result["judge_cached_tokens"],
                    "judge_response_metadata": judge_result["judge_response_metadata"],
                }
            )
    return {
        "started_at_epoch": started_at,
        "case_count": len(case_records),
        "result_count": len(rows),
        "judge_provider": judge_provider,
        "judge_model": judge_model,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strict faithfulness/refusal/multi-step quality evals for RAG answers.")
    parser.add_argument("--cases", default="apps/api/evals/quality_cases.strict.json")
    parser.add_argument("--models", default="apps/api/evals/models.top4.upgraded.json")
    parser.add_argument("--doctrine-manifest", default="apps/firecastbot/incident_response_docs/doctrine_retrieval_manifest.bge_base.json")
    parser.add_argument("--incident-report", default="apps/firecastbot/incident_reports/incident_report_boulder.pdf")
    parser.add_argument("--judge-provider", default="groq")
    parser.add_argument("--judge-model", default="llama-3.3-70b-versatile")
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-max-tokens", type=int, default=900)
    parser.add_argument("--retrieval-k", type=int, default=5)
    parser.add_argument("--output-json", default="apps/api/evals/rag_quality_eval_top4_groq_judge.json")
    parser.add_argument("--output-csv", default="apps/api/evals/rag_quality_eval_top4_groq_judge.csv")
    parser.add_argument("--fail-on-rate-limit", action="store_true", default=True)
    args = parser.parse_args()
    cases = load_json_records(Path(args.cases))
    models = load_json_records(Path(args.models))
    report = run_quality_eval(
        case_records=cases,
        model_records=models,
        doctrine_manifest=Path(args.doctrine_manifest),
        incident_report=Path(args.incident_report),
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
        judge_temperature=args.judge_temperature,
        judge_max_tokens=args.judge_max_tokens,
        retrieval_k=args.retrieval_k,
        fail_on_rate_limit=args.fail_on_rate_limit,
    )
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(report["rows"], Path(args.output_csv))
    print(f"Wrote {report['result_count']} quality eval rows to {output_json} and {args.output_csv}.")


if __name__ == "__main__":
    main()
