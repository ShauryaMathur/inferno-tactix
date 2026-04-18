from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests

PACKAGE_ROOT = Path(__file__).resolve().parent
APP_ROOT = PACKAGE_ROOT.parent.parent
REPO_ROOT = APP_ROOT.parent.parent
APPS_ROOT = REPO_ROOT / "apps"

if str(APPS_ROOT) not in sys.path:
    sys.path.insert(0, str(APPS_ROOT))

from firecastbot.config import Settings, get_settings
from firecastbot.services.llm_service import ChatCompletionResult, LLMService
from inferno_api.firecastbot_runtime import (
    build_grounded_prompt,
    build_runtime_incident_report,
    classify_query,
    load_doctrine_assets,
    retrieve_firecast_context,
)


def load_json_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return data["items"]
    raise ValueError(f"Unsupported record format in {path}")


def normalize_settings(base: Settings, override: dict[str, Any]) -> Settings:
    payload = dict(base.__dict__)
    payload.update({key: value for key, value in override.items() if value is not None})
    return Settings(**payload)


def build_messages(
    *,
    query: str,
    query_class: str,
    incident_profile: dict[str, Any] | None,
    context_items: list[dict[str, Any]],
) -> list[dict[str, str]]:
    prompt = build_grounded_prompt(
        query=query,
        query_class=query_class,
        incident_profile=incident_profile,
        context_items=context_items,
        conversation_summary="",
        recent_conversation=[],
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a wildfire decision-support assistant grounded in incident facts and doctrine. "
                "Be spatial-context aware: tailor guidance to the incident's geography, terrain, and stated weather. "
                "Be risk-context aware: use the incident's stated Overall Risk Level to calibrate urgency, caution, and safety emphasis. "
                "You may mention likely regional conditions when location strongly implies them, but label those as inferred context, "
                "not confirmed incident facts."
            ),
        },
        {"role": "user", "content": prompt},
    ]


def evaluate_correctness(reply: str, prompt: dict[str, Any]) -> dict[str, Any]:
    expected_contains = [str(item).strip() for item in prompt.get("expected_contains", []) if str(item).strip()]
    lowered_reply = reply.casefold()
    matched = [item for item in expected_contains if item.casefold() in lowered_reply]
    score = None
    if expected_contains:
        score = round(len(matched) / len(expected_contains), 4)
    return {
        "expected_contains": expected_contains,
        "matched_expected": matched,
        "correctness_score": score,
    }


def _is_rate_limit_error(message: str) -> bool:
    lowered = message.casefold()
    return "ratelimit" in lowered or "rate limit" in lowered or "code': 429" in lowered or "error code: 429" in lowered


def _text_metrics(text: str) -> dict[str, int]:
    return {
        "char_count": len(text),
        "word_count": len(text.split()),
        "line_count": len(text.splitlines()),
    }


def _result_metrics(prefix: str, result: ChatCompletionResult | None) -> dict[str, Any]:
    if result is None:
        return {
            f"{prefix}_response_id": "",
            f"{prefix}_finish_reason": "",
            f"{prefix}_prompt_tokens": None,
            f"{prefix}_completion_tokens": None,
            f"{prefix}_total_tokens": None,
            f"{prefix}_reasoning_tokens": None,
            f"{prefix}_cached_tokens": None,
            f"{prefix}_response_metadata": json.dumps({}, ensure_ascii=True),
        }
    return {
        f"{prefix}_response_id": result.response_id,
        f"{prefix}_finish_reason": result.finish_reason,
        f"{prefix}_prompt_tokens": result.prompt_tokens,
        f"{prefix}_completion_tokens": result.completion_tokens,
        f"{prefix}_total_tokens": result.total_tokens,
        f"{prefix}_reasoning_tokens": result.reasoning_tokens,
        f"{prefix}_cached_tokens": result.cached_tokens,
        f"{prefix}_response_metadata": json.dumps(result.response_metadata or {}, ensure_ascii=True),
    }


def _extract_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if not candidate:
        raise ValueError("Judge returned an empty response.")
    try:
        payload = json.loads(candidate)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", candidate, re.DOTALL)
    if not match:
        raise ValueError("Judge response did not contain a JSON object.")
    payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("Judge JSON payload must be an object.")
    return payload


def build_judge_messages(
    *,
    query: str,
    prompt_record: dict[str, Any],
    rag_system_prompt: str,
    rag_user_prompt: str,
    rag_reply: str,
    retrieval_sources: list[dict[str, Any]],
) -> list[dict[str, str]]:
    expected_contains = [str(item).strip() for item in prompt_record.get("expected_contains", []) if str(item).strip()]
    rubric = str(prompt_record.get("rubric") or "").strip() or "No rubric provided."
    reference_answer = str(prompt_record.get("reference_answer") or "").strip() or "No reference answer provided."
    judge_schema = {
        "correctness_score": "number from 0.0 to 1.0",
        "matched_expected": ["subset of expected_contains strings actually satisfied by the answer"],
        "groundedness_score": "number from 0.0 to 1.0",
        "instruction_following_score": "number from 0.0 to 1.0",
        "error": "empty string if no problem, otherwise concise issue description",
        "rationale": "short explanation of the score",
    }
    return [
        {
            "role": "system",
            "content": (
                "You are an exacting evaluator for a wildfire RAG assistant. "
                "Score the assistant response only against the supplied evidence and rubric. "
                "Return compact JSON only with no markdown fences or extra commentary."
            ),
        },
        {
            "role": "user",
            "content": (
                "Evaluate the assistant response.\n\n"
                f"User query:\n{query}\n\n"
                f"Expected key phrases:\n{json.dumps(expected_contains, ensure_ascii=True)}\n\n"
                f"Rubric:\n{rubric}\n\n"
                f"Reference answer:\n{reference_answer}\n\n"
                f"RAG system prompt:\n{rag_system_prompt}\n\n"
                f"RAG user prompt:\n{rag_user_prompt}\n\n"
                f"Retrieved sources:\n{json.dumps(retrieval_sources, ensure_ascii=True)}\n\n"
                f"Assistant response:\n{rag_reply}\n\n"
                "Scoring rules:\n"
                "- correctness_score should reflect whether the answer satisfies the rubric and the expected key phrases.\n"
                "- groundedness_score should reflect whether the answer stays faithful to the provided sources and avoids unsupported claims.\n"
                "- instruction_following_score should reflect whether the answer follows the structure and constraints implied by the RAG prompt.\n"
                "- matched_expected must contain only strings from expected key phrases.\n"
                "- Use empty string for error when the answer is evaluable.\n\n"
                "- Keep rationale to one short sentence.\n\n"
                f"Return JSON with this shape:\n{json.dumps(judge_schema, ensure_ascii=True)}"
            ),
        },
    ]


def evaluate_with_gemini_judge_rest(
    *,
    api_key: str,
    judge_model: str,
    judge_temperature: float,
    judge_max_tokens: int,
    judge_messages: list[dict[str, str]],
) -> tuple[ChatCompletionResult, dict[str, Any]]:
    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{judge_model}:generateContent",
        headers={
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        },
        json={
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": (
                                f"SYSTEM:\n{judge_messages[0]['content']}\n\n"
                                f"USER:\n{judge_messages[1]['content']}"
                            )
                        }
                    ],
                }
            ],
            "generationConfig": {
                "temperature": judge_temperature,
                "maxOutputTokens": judge_max_tokens,
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "OBJECT",
                    "properties": {
                        "correctness_score": {"type": "NUMBER"},
                        "matched_expected": {"type": "ARRAY", "items": {"type": "STRING"}},
                        "groundedness_score": {"type": "NUMBER"},
                        "instruction_following_score": {"type": "NUMBER"},
                        "error": {"type": "STRING"},
                        "rationale": {"type": "STRING"},
                    },
                    "required": [
                        "correctness_score",
                        "matched_expected",
                        "groundedness_score",
                        "instruction_following_score",
                        "error",
                        "rationale",
                    ],
                },
            },
        },
        timeout=300,
    )
    response.raise_for_status()
    payload = response.json()
    candidates = payload.get("candidates") or []
    if not candidates:
        raise RuntimeError("Gemini judge returned no candidates.")
    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    text = "".join(str(part.get("text") or "") for part in parts)
    if not text:
        raise RuntimeError("Gemini judge returned no text content.")
    usage = payload.get("usageMetadata") or {}
    result = ChatCompletionResult(
        content=text,
        provider="gemini",
        model=judge_model,
        response_id=str(payload.get("responseId") or ""),
        finish_reason=str(candidates[0].get("finishReason") or ""),
        prompt_tokens=usage.get("promptTokenCount"),
        completion_tokens=usage.get("candidatesTokenCount"),
        total_tokens=usage.get("totalTokenCount"),
        cached_tokens=usage.get("cachedContentTokenCount"),
        response_metadata={"traffic_type": usage.get("trafficType")},
    )
    return result, _extract_json_object(text)


def evaluate_with_gemini_judge_text_fallback(
    *,
    api_key: str,
    judge_model: str,
    judge_temperature: float,
    judge_max_tokens: int,
    judge_messages: list[dict[str, str]],
) -> tuple[ChatCompletionResult, dict[str, Any]]:
    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{judge_model}:generateContent",
        headers={
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        },
        json={
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": (
                                f"SYSTEM:\n{judge_messages[0]['content']}\n\n"
                                f"USER:\n{judge_messages[1]['content']}\n\n"
                                "Return JSON only."
                            )
                        }
                    ],
                }
            ],
            "generationConfig": {
                "temperature": judge_temperature,
                "maxOutputTokens": judge_max_tokens,
            },
        },
        timeout=300,
    )
    response.raise_for_status()
    payload = response.json()
    candidates = payload.get("candidates") or []
    if not candidates:
        raise RuntimeError("Gemini judge fallback returned no candidates.")
    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    text = "".join(str(part.get("text") or "") for part in parts)
    if not text:
        raise RuntimeError("Gemini judge fallback returned no text content.")
    usage = payload.get("usageMetadata") or {}
    result = ChatCompletionResult(
        content=text,
        provider="gemini",
        model=judge_model,
        response_id=str(payload.get("responseId") or ""),
        finish_reason=str(candidates[0].get("finishReason") or ""),
        prompt_tokens=usage.get("promptTokenCount"),
        completion_tokens=usage.get("candidatesTokenCount"),
        total_tokens=usage.get("totalTokenCount"),
        cached_tokens=usage.get("cachedContentTokenCount"),
        response_metadata={"traffic_type": usage.get("trafficType"), "mode": "text_fallback"},
    )
    return result, _extract_json_object(text)


def evaluate_with_judge(
    *,
    settings: Settings,
    judge_llm: LLMService | None,
    judge_provider: str,
    judge_model: str,
    judge_temperature: float,
    judge_max_tokens: int,
    query: str,
    prompt_record: dict[str, Any],
    rag_messages: list[dict[str, str]],
    rag_reply: str,
    retrieval_sources: list[dict[str, Any]],
) -> dict[str, Any]:
    judge_messages = build_judge_messages(
        query=query,
        prompt_record=prompt_record,
        rag_system_prompt=rag_messages[0]["content"],
        rag_user_prompt=rag_messages[1]["content"],
        rag_reply=rag_reply,
        retrieval_sources=retrieval_sources,
    )
    judge_reply = ""
    judge_error = ""
    parsed: dict[str, Any] = {}
    judge_completion: ChatCompletionResult | None = None
    request_started = time.perf_counter()
    try:
        if judge_provider.strip().casefold() == "gemini":
            api_key = settings.require_api_key("gemini")
            try:
                judge_completion, parsed = evaluate_with_gemini_judge_rest(
                    api_key=api_key,
                    judge_model=judge_model,
                    judge_temperature=judge_temperature,
                    judge_max_tokens=judge_max_tokens,
                    judge_messages=judge_messages,
                )
            except Exception:
                judge_completion, parsed = evaluate_with_gemini_judge_text_fallback(
                    api_key=api_key,
                    judge_model=judge_model,
                    judge_temperature=judge_temperature,
                    judge_max_tokens=judge_max_tokens,
                    judge_messages=judge_messages,
                )
            judge_reply = judge_completion.content
        else:
            if judge_llm is None:
                raise RuntimeError("Judge LLM is not configured.")
            judge_completion = judge_llm.chat_completion_result(judge_messages)
            judge_reply = judge_completion.content
            parsed = _extract_json_object(judge_reply)
    except Exception as exc:
        judge_error = f"{type(exc).__name__}: {exc}"
    judge_latency_ms = round((time.perf_counter() - request_started) * 1000, 2)

    matched_expected = parsed.get("matched_expected")
    if not isinstance(matched_expected, list):
        matched_expected = []
    matched_expected = [str(item).strip() for item in matched_expected if str(item).strip()]

    def _score(key: str) -> float | None:
        value = parsed.get(key)
        if value is None or value == "":
            return None
        try:
            return round(float(value), 4)
        except (TypeError, ValueError):
            return None

    return {
        "judge_system_prompt": judge_messages[0]["content"],
        "judge_user_prompt": judge_messages[1]["content"],
        "judge_reply": judge_reply,
        "judge_error": judge_error or str(parsed.get("error") or ""),
        "judge_latency_ms": judge_latency_ms,
        "correctness_score": _score("correctness_score"),
        "matched_expected": matched_expected,
        "groundedness_score": _score("groundedness_score"),
        "instruction_following_score": _score("instruction_following_score"),
        "judge_rationale": str(parsed.get("rationale") or ""),
        **_result_metrics("judge", judge_completion),
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_evaluation(
    *,
    settings: Settings,
    doctrine_manifest: Path,
    prompt_records: list[dict[str, Any]],
    model_records: list[dict[str, Any]],
    incident_report_path: Path | None,
) -> dict[str, Any]:
    doctrine_store = load_doctrine_assets(doctrine_manifest)
    incident_bundle: dict[str, Any] | None = None
    if incident_report_path is not None:
        incident_bundle = build_runtime_incident_report(
            incident_report_path.read_bytes(),
            incident_report_path.name,
            settings.embedding_provider,
            settings.embedding_model,
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
            incident_profile = incident_bundle["incident_profile"] if incident_bundle else None
            incident_chunks = incident_bundle["incident_chunks"] if incident_bundle else []
            incident_embeddings = incident_bundle["incident_embeddings"] if incident_bundle else None
            incident_keyword_index = incident_bundle["incident_keyword_index"] if incident_bundle else None
            incident_embedding_provider = (
                str(incident_bundle["embedding_provider"]) if incident_bundle else settings.embedding_provider
            )
            incident_embedding_model = (
                str(incident_bundle["embedding_model"]) if incident_bundle else settings.embedding_model
            )
            context_items = retrieve_firecast_context(
                query,
                query_class=query_class,
                retrieval_k=settings.retrieval_k,
                incident_profile=incident_profile,
                incident_chunks=incident_chunks,
                incident_embeddings=incident_embeddings,
                incident_keyword_index=incident_keyword_index,
                incident_embedding_provider=incident_embedding_provider,
                incident_embedding_model=incident_embedding_model,
                doctrine_store=doctrine_store,
            )
            messages = build_messages(
                query=query,
                query_class=query_class,
                incident_profile=incident_profile,
                context_items=context_items,
            )
            request_started = time.perf_counter()
            reply = ""
            error = ""
            try:
                reply = llm.chat_completion(messages)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            latency_ms = round((time.perf_counter() - request_started) * 1000, 2)
            correctness = evaluate_correctness(reply, prompt_record)
            results.append(
                {
                    "provider": effective_settings.llm_provider,
                    "model": effective_settings.llm_model,
                    "label": str(model_record.get("label") or f"{effective_settings.llm_provider}:{effective_settings.llm_model}"),
                    "prompt_id": str(prompt_record.get("id") or query[:40]),
                    "query_class": query_class,
                    "latency_ms": latency_ms,
                    "correctness_score": correctness["correctness_score"],
                    "matched_expected_count": len(correctness["matched_expected"]),
                    "expected_count": len(correctness["expected_contains"]),
                    "reply": reply,
                    "error": error,
                    "query": query,
                    "retrieval_context_count": len(context_items),
                    "retrieval_sources": json.dumps(
                        [
                            {
                                "source_type": item.get("source_type"),
                                "section": item.get("section") or item.get("section_path") or item.get("label"),
                                "chunk_id": item.get("chunk_id") or item.get("fact_key"),
                            }
                            for item in context_items
                        ],
                        ensure_ascii=True,
                    ),
                    "matched_expected": json.dumps(correctness["matched_expected"], ensure_ascii=True),
                }
            )

    return {
        "started_at_epoch": started_at,
        "doctrine_manifest": str(doctrine_manifest),
        "incident_report": str(incident_report_path) if incident_report_path else None,
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "result_count": len(results),
        "results": results,
    }


def run_evaluation_with_judge(
    *,
    settings: Settings,
    doctrine_manifest: Path,
    prompt_records: list[dict[str, Any]],
    model_records: list[dict[str, Any]],
    incident_report_path: Path | None,
    judge_provider: str,
    judge_model: str,
    judge_temperature: float,
    judge_max_tokens: int,
    fail_on_rate_limit: bool = False,
) -> dict[str, Any]:
    doctrine_store = load_doctrine_assets(doctrine_manifest)
    incident_bundle: dict[str, Any] | None = None
    if incident_report_path is not None:
        incident_bundle = build_runtime_incident_report(
            incident_report_path.read_bytes(),
            incident_report_path.name,
            settings.embedding_provider,
            settings.embedding_model,
        )

    judge_settings = normalize_settings(
        settings,
        {
            "llm_provider": judge_provider,
            "llm_model": judge_model,
            "llm_temperature": judge_temperature,
            "llm_max_tokens": judge_max_tokens,
        },
    )
    judge_llm: LLMService | None = None
    if judge_provider.strip().casefold() != "gemini":
        judge_llm = LLMService(
            judge_settings,
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
            incident_profile = incident_bundle["incident_profile"] if incident_bundle else None
            incident_chunks = incident_bundle["incident_chunks"] if incident_bundle else []
            incident_embeddings = incident_bundle["incident_embeddings"] if incident_bundle else None
            incident_keyword_index = incident_bundle["incident_keyword_index"] if incident_bundle else None
            incident_embedding_provider = (
                str(incident_bundle["embedding_provider"]) if incident_bundle else settings.embedding_provider
            )
            incident_embedding_model = (
                str(incident_bundle["embedding_model"]) if incident_bundle else settings.embedding_model
            )
            context_items = retrieve_firecast_context(
                query,
                query_class=query_class,
                retrieval_k=settings.retrieval_k,
                incident_profile=incident_profile,
                incident_chunks=incident_chunks,
                incident_embeddings=incident_embeddings,
                incident_keyword_index=incident_keyword_index,
                incident_embedding_provider=incident_embedding_provider,
                incident_embedding_model=incident_embedding_model,
                doctrine_store=doctrine_store,
            )
            messages = build_messages(
                query=query,
                query_class=query_class,
                incident_profile=incident_profile,
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
            request_started = time.perf_counter()
            reply = ""
            error = ""
            completion: ChatCompletionResult | None = None
            try:
                completion = llm.chat_completion_result(messages)
                reply = completion.content
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                if fail_on_rate_limit and _is_rate_limit_error(error):
                    raise RuntimeError(
                        f"Rate limit error for {effective_settings.llm_provider}:{effective_settings.llm_model} on prompt {prompt_record.get('id') or query[:40]}: {error}"
                    ) from exc
            latency_ms = round((time.perf_counter() - request_started) * 1000, 2)

            keyword_correctness = evaluate_correctness(reply, prompt_record)
            judge_result = {
                "judge_system_prompt": "",
                "judge_user_prompt": "",
                "judge_reply": "",
                "judge_error": "",
                "judge_latency_ms": None,
                "correctness_score": None,
                "matched_expected": [],
                "groundedness_score": None,
                "instruction_following_score": None,
                "judge_rationale": "",
            }
            if not error and reply.strip():
                judge_result = evaluate_with_judge(
                    settings=settings,
                    judge_llm=judge_llm,
                    judge_provider=judge_provider,
                    judge_model=judge_model,
                    judge_temperature=judge_temperature,
                    judge_max_tokens=judge_max_tokens,
                    query=query,
                    prompt_record=prompt_record,
                    rag_messages=messages,
                    rag_reply=reply,
                    retrieval_sources=retrieval_sources,
                )
                if fail_on_rate_limit and judge_result["judge_error"] and _is_rate_limit_error(judge_result["judge_error"]):
                    raise RuntimeError(
                        f"Judge rate limit error for {judge_provider}:{judge_model} on prompt {prompt_record.get('id') or query[:40]}: {judge_result['judge_error']}"
                    )

            matched_expected = judge_result["matched_expected"] or keyword_correctness["matched_expected"]
            rag_prompt_text = "\n\n".join(message["content"] for message in messages)
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
                    "rag_system_prompt": messages[0]["content"],
                    "rag_user_prompt": messages[1]["content"],
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
        "incident_report": str(incident_report_path) if incident_report_path else None,
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "judge_provider": judge_provider,
        "judge_model": judge_model,
        "fail_on_rate_limit": fail_on_rate_limit,
        "result_count": len(results),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare multiple LLMs on the same FireCastBot RAG prompts."
    )
    parser.add_argument(
        "--prompts",
        required=True,
        help="Path to prompt records JSON/JSONL. Each item needs at least a query field.",
    )
    parser.add_argument(
        "--models",
        required=True,
        help="Path to model records JSON/JSONL. Each item needs provider and model.",
    )
    parser.add_argument(
        "--incident-report",
        help="Optional path to an incident report PDF to include runtime incident retrieval.",
    )
    parser.add_argument(
        "--doctrine-manifest",
        default="apps/chatwithme/incident_response_docs/doctrine_retrieval_manifest.json",
        help="Path to doctrine retrieval manifest.",
    )
    parser.add_argument(
        "--output-json",
        default="apps/api/evals/rag_model_compare.json",
        help="Path to write the evaluation JSON report.",
    )
    parser.add_argument(
        "--output-csv",
        default="apps/api/evals/rag_model_compare.csv",
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
