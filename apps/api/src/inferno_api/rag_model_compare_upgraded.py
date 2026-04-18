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

from firecastbot.config import Settings, get_settings
from firecastbot.services.llm_service import ChatCompletionResult, LLMService
from inferno_api.firecastbot_runtime import (
    FACT_OUTPUT_LABELS,
    build_runtime_incident_report,
    classify_query,
    load_doctrine_assets,
    normalize_whitespace,
    retrieve_firecast_context,
)
from inferno_api.rag_model_compare import (
    _is_rate_limit_error,
    _result_metrics,
    _text_metrics,
    evaluate_correctness,
    load_json_records,
    normalize_settings,
    write_csv,
)


def render_context_items_strict(items: list[dict[str, Any]], *, limit_chars: int = 4000) -> str:
    lines: list[str] = []
    for item in items:
        source_type = str(item.get("source_type", "unknown"))
        if source_type == "incident_fact":
            label = str(item.get("label") or "Fact")
            text = str(item.get("text") or "")
            lines.append(f"[Incident Report] {label}: {text.split(':', 1)[-1].strip()}")
            continue
        source_label = "[Doctrine]" if source_type == "doctrine" else "[Incident Report]"
        section = str(item.get("section") or item.get("section_path") or "")
        chunk_id = str(item.get("chunk_id") or item.get("fact_key") or "")
        text = normalize_whitespace(str(item.get("text") or ""))
        lines.append(f"{source_label} section={section} chunk_id={chunk_id} text={text}")
    payload = "\n".join(lines)
    if len(payload) <= limit_chars:
        return payload
    return payload[:limit_chars].rstrip() + "\n[... context truncated for evaluation ...]"


def build_messages_upgraded(
    *,
    query: str,
    query_class: str,
    incident_profile: dict[str, Any] | None,
    context_items: list[dict[str, Any]],
) -> list[dict[str, str]]:
    incident_facts = incident_profile.get("facts", {}) if incident_profile else {}
    facts_block = "\n".join(
        f"- {FACT_OUTPUT_LABELS[key]}: {value}"
        for key, value in incident_facts.items()
        if value
    ) or "- No incident facts are available."
    answer_contract = (
        "Output exactly three sections titled: "
        "1. Incident-grounded facts, 2. Doctrine-grounded guidance, 3. Suggested interpretation."
        if query_class in {"incident+doctrine synthesis", "safety-critical"}
        else "Write a concise answer grounded only in the provided evidence."
    )
    system_prompt = (
        "You are FireCastBot, a wildfire decision-support assistant. "
        "You must be evidence-grounded, conservative, and explicit about uncertainty. "
        "Never invent incident facts. Never attribute doctrine content to the incident report. "
        "If evidence is missing, say so plainly. Prefer direct quotations of facts only when needed, "
        "otherwise summarize concisely. Every substantive claim must be attributable to [Incident Report] or [Doctrine]."
    )
    user_prompt = (
        f"Task: answer the user query for query class `{query_class}`.\n\n"
        "Follow these rules strictly:\n"
        "- Use only the incident facts and retrieved context below.\n"
        "- Distinguish clearly between incident facts and doctrine.\n"
        "- Do not speculate beyond the provided evidence.\n"
        "- If the incident report lacks a needed detail, say that explicitly.\n"
        "- Mention that local SOPs, IC direction, or real-time conditions may override general doctrine.\n"
        f"- {answer_contract}\n\n"
        f"Incident profile:\n{facts_block}\n\n"
        f"Retrieved evidence:\n{render_context_items_strict(context_items)}\n\n"
        f"User question:\n{query}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


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


def build_strict_judge_messages(
    *,
    query: str,
    prompt_record: dict[str, Any],
    rag_messages: list[dict[str, str]],
    rag_reply: str,
    retrieval_sources: list[dict[str, Any]],
) -> list[dict[str, str]]:
    expected_contains = [str(item).strip() for item in prompt_record.get("expected_contains", []) if str(item).strip()]
    rubric = str(prompt_record.get("rubric") or "").strip() or "No rubric provided."
    reference_answer = str(prompt_record.get("reference_answer") or "").strip() or "No reference answer provided."
    schema = {
        "correctness_score": "number in [0,1]",
        "groundedness_score": "number in [0,1]",
        "instruction_following_score": "number in [0,1]",
        "matched_expected": ["subset of expected key phrases actually satisfied"],
        "error": "empty string unless evaluation cannot be completed",
        "rationale": "one short sentence",
    }
    system = (
        "You are a strict evaluation judge for a wildfire RAG benchmark. "
        "Be conservative. Penalize unsupported claims, hedging that avoids the question, and missing required structure. "
        "Return JSON only."
    )
    user = (
        f"Evaluate the assistant answer.\n\n"
        f"Query:\n{query}\n\n"
        f"Expected key phrases:\n{json.dumps(expected_contains, ensure_ascii=True)}\n\n"
        f"Rubric:\n{rubric}\n\n"
        f"Reference answer:\n{reference_answer}\n\n"
        f"Retrieved sources:\n{json.dumps(retrieval_sources, ensure_ascii=True)}\n\n"
        f"RAG system prompt:\n{rag_messages[0]['content'][:1800]}\n\n"
        f"RAG user prompt:\n{rag_messages[1]['content'][:3200]}\n\n"
        f"Assistant answer:\n{rag_reply[:3200]}\n\n"
        "Scoring guidance:\n"
        "- correctness_score: does the answer satisfy the rubric and answer the query well?\n"
        "- groundedness_score: are claims supported by the supplied evidence only?\n"
        "- instruction_following_score: did the answer follow the requested structure and constraints?\n"
        "- matched_expected: include only expected phrases that are genuinely satisfied, not approximate.\n"
        "- If evidence is missing and the model states that correctly, do not penalize correctness for refusing to invent facts.\n\n"
        f"Return JSON matching this schema:\n{json.dumps(schema, ensure_ascii=True)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def evaluate_with_strict_groq_judge(
    *,
    judge_llm: LLMService,
    query: str,
    prompt_record: dict[str, Any],
    rag_messages: list[dict[str, str]],
    rag_reply: str,
    retrieval_sources: list[dict[str, Any]],
) -> dict[str, Any]:
    judge_messages = build_strict_judge_messages(
        query=query,
        prompt_record=prompt_record,
        rag_messages=rag_messages,
        rag_reply=rag_reply,
        retrieval_sources=retrieval_sources,
    )
    judge_completion: ChatCompletionResult | None = None
    judge_reply = ""
    judge_error = ""
    parsed: dict[str, Any] = {}
    started = time.perf_counter()
    try:
        judge_completion = judge_llm.chat_completion_result(judge_messages)
        judge_reply = judge_completion.content
        parsed = _extract_json_object(judge_reply)
    except Exception as exc:
        judge_error = f"{type(exc).__name__}: {exc}"
    latency_ms = round((time.perf_counter() - started) * 1000, 2)

    def _score(key: str) -> float | None:
        value = parsed.get(key)
        if value in {None, ""}:
            return None
        try:
            return round(float(value), 4)
        except (TypeError, ValueError):
            return None

    matched_expected = parsed.get("matched_expected")
    if not isinstance(matched_expected, list):
        matched_expected = []
    matched_expected = [str(item).strip() for item in matched_expected if str(item).strip()]

    return {
        "judge_system_prompt": judge_messages[0]["content"],
        "judge_user_prompt": judge_messages[1]["content"],
        "judge_reply": judge_reply,
        "judge_error": judge_error or str(parsed.get("error") or ""),
        "judge_latency_ms": latency_ms,
        "correctness_score": _score("correctness_score"),
        "groundedness_score": _score("groundedness_score"),
        "instruction_following_score": _score("instruction_following_score"),
        "matched_expected": matched_expected,
        "judge_rationale": str(parsed.get("rationale") or ""),
        **_result_metrics("judge", judge_completion),
    }


def run_upgraded_evaluation(
    *,
    settings: Settings,
    doctrine_manifest: Path,
    prompt_records: list[dict[str, Any]],
    model_records: list[dict[str, Any]],
    incident_report_path: Path,
    judge_provider: str,
    judge_model: str,
    judge_temperature: float,
    judge_max_tokens: int,
    retrieval_k: int,
    fail_on_rate_limit: bool,
) -> dict[str, Any]:
    doctrine_store = load_doctrine_assets(doctrine_manifest)
    incident_bundle = build_runtime_incident_report(
        incident_report_path.read_bytes(),
        incident_report_path.name,
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
    results: list[dict[str, Any]] = []
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

        for prompt_record in prompt_records:
            query = str(prompt_record["query"]).strip()
            query_class = str(prompt_record.get("query_class") or classify_query(query))
            context_items = retrieve_firecast_context(
                query,
                query_class=query_class,
                retrieval_k=retrieval_k,
                incident_profile=incident_bundle["incident_profile"],
                incident_chunks=incident_bundle["incident_chunks"],
                incident_embeddings=incident_bundle["incident_embeddings"],
                incident_keyword_index=incident_bundle["incident_keyword_index"],
                incident_embedding_provider=str(incident_bundle["embedding_provider"]),
                incident_embedding_model=str(incident_bundle["embedding_model"]),
                doctrine_store=doctrine_store,
            )
            rag_messages = build_messages_upgraded(
                query=query,
                query_class=query_class,
                incident_profile=incident_bundle["incident_profile"],
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
                    raise RuntimeError(
                        f"Rate limit error for {effective_settings.llm_provider}:{effective_settings.llm_model} on prompt {prompt_record.get('id') or query[:40]}: {error}"
                    ) from exc
            latency_ms = round((time.perf_counter() - started) * 1000, 2)

            keyword_correctness = evaluate_correctness(reply, prompt_record)
            judge_result = {
                "judge_system_prompt": "",
                "judge_user_prompt": "",
                "judge_reply": "",
                "judge_error": "",
                "judge_latency_ms": None,
                "correctness_score": None,
                "groundedness_score": None,
                "instruction_following_score": None,
                "matched_expected": [],
                "judge_rationale": "",
                **_result_metrics("judge", None),
            }
            if not error and reply.strip():
                judge_result = evaluate_with_strict_groq_judge(
                    judge_llm=judge_llm,
                    query=query,
                    prompt_record=prompt_record,
                    rag_messages=rag_messages,
                    rag_reply=reply,
                    retrieval_sources=retrieval_sources,
                )
                if fail_on_rate_limit and judge_result["judge_error"] and _is_rate_limit_error(judge_result["judge_error"]):
                    raise RuntimeError(
                        f"Judge rate limit error for {judge_provider}:{judge_model} on prompt {prompt_record.get('id') or query[:40]}: {judge_result['judge_error']}"
                    )

            matched_expected = judge_result["matched_expected"] or keyword_correctness["matched_expected"]
            rag_prompt_text = "\n\n".join(message["content"] for message in rag_messages)
            query_metrics = _text_metrics(query)
            reply_metrics = _text_metrics(reply)
            rag_prompt_metrics = _text_metrics(rag_prompt_text)
            results.append(
                {
                    "provider": effective_settings.llm_provider,
                    "model": effective_settings.llm_model,
                    "label": str(model_record.get("label") or f"{effective_settings.llm_provider}:{effective_settings.llm_model}"),
                    "prompt_id": str(prompt_record.get("id") or query[:40]),
                    "query_class": query_class,
                    "latency_ms": latency_ms,
                    "correctness_score": judge_result["correctness_score"],
                    "groundedness_score": judge_result["groundedness_score"],
                    "instruction_following_score": judge_result["instruction_following_score"],
                    "keyword_correctness_score": keyword_correctness["correctness_score"],
                    "matched_expected_count": len(matched_expected),
                    "expected_count": len(keyword_correctness["expected_contains"]),
                    "reply": reply,
                    "error": error,
                    "query": query,
                    "query_char_count": query_metrics["char_count"],
                    "query_word_count": query_metrics["word_count"],
                    "retrieval_context_count": len(context_items),
                    "retrieval_sources": json.dumps(retrieval_sources, ensure_ascii=True),
                    "matched_expected": json.dumps(matched_expected, ensure_ascii=True),
                    "rag_system_prompt": rag_messages[0]["content"],
                    "rag_user_prompt": rag_messages[1]["content"],
                    "rag_prompt_char_count": rag_prompt_metrics["char_count"],
                    "rag_prompt_word_count": rag_prompt_metrics["word_count"],
                    "reply_char_count": reply_metrics["char_count"],
                    "reply_word_count": reply_metrics["word_count"],
                    "reply_line_count": reply_metrics["line_count"],
                    "model_provider": effective_settings.llm_provider,
                    "model_temperature": effective_settings.llm_temperature,
                    "model_max_tokens": effective_settings.llm_max_tokens,
                    "judge_provider": judge_provider,
                    "judge_model": judge_model,
                    "judge_temperature": judge_temperature,
                    "judge_max_tokens": judge_max_tokens,
                    "judge_latency_ms": judge_result["judge_latency_ms"],
                    "judge_error": judge_result["judge_error"],
                    "judge_rationale": judge_result["judge_rationale"],
                    "judge_system_prompt": judge_result["judge_system_prompt"],
                    "judge_user_prompt": judge_result["judge_user_prompt"],
                    "judge_reply": judge_result["judge_reply"],
                    "judge_reply_char_count": len(judge_result["judge_reply"]),
                    "judge_reply_word_count": len(judge_result["judge_reply"].split()),
                    "embedding_provider": settings.embedding_provider,
                    "embedding_model": settings.embedding_model,
                    "doctrine_manifest": str(doctrine_manifest),
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
        "doctrine_manifest": str(doctrine_manifest),
        "incident_report": str(incident_report_path),
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "judge_provider": judge_provider,
        "judge_model": judge_model,
        "retrieval_k": retrieval_k,
        "result_count": len(results),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an upgraded RAG comparison with stronger free embeddings, upgraded prompts, and a strict Groq judge."
    )
    parser.add_argument("--prompts", default="apps/api/evals/prompts.sample.json")
    parser.add_argument("--models", default="apps/api/evals/models.openrouter.all.json")
    parser.add_argument("--incident-report", default="apps/firecastbot/incident_reports/incident_report_boulder.pdf")
    parser.add_argument("--doctrine-manifest", default="apps/firecastbot/incident_response_docs/doctrine_retrieval_manifest.bge_base.json")
    parser.add_argument("--embedding-provider", default="sentence-transformers")
    parser.add_argument("--embedding-model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--judge-provider", default="groq")
    parser.add_argument("--judge-model", default="llama-3.3-70b-versatile")
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-max-tokens", type=int, default=900)
    parser.add_argument("--retrieval-k", type=int, default=5)
    parser.add_argument("--output-json", default="apps/api/evals/rag_model_compare_upgraded_bge_groq_judge.json")
    parser.add_argument("--output-csv", default="apps/api/evals/rag_model_compare_upgraded_bge_groq_judge.csv")
    parser.add_argument("--fail-on-rate-limit", action="store_true", default=True)
    args = parser.parse_args()

    settings = normalize_settings(
        get_settings(),
        {
            "embedding_provider": args.embedding_provider,
            "embedding_model": args.embedding_model,
        },
    )
    prompt_records = load_json_records(Path(args.prompts))
    model_records = load_json_records(Path(args.models))
    report = run_upgraded_evaluation(
        settings=settings,
        doctrine_manifest=Path(args.doctrine_manifest),
        prompt_records=prompt_records,
        model_records=model_records,
        incident_report_path=Path(args.incident_report),
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
    write_csv(report["results"], Path(args.output_csv))
    print(f"Wrote {report['result_count']} evaluation rows to {output_json} and {args.output_csv}.")


if __name__ == "__main__":
    main()
