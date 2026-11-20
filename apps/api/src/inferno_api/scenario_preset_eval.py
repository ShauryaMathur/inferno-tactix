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

from firecastbot.config import get_settings  # noqa: E402
from firecastbot.services.llm_service import (  # noqa: E402
    ChatCompletionResult,
    LLMService,
)
from inferno_api.firecastbot_runtime import (  # noqa: E402
    build_runtime_incident_report,
    classify_query,
    load_doctrine_assets,
    retrieve_firecast_context,
)
from inferno_api.rag_model_compare import (  # noqa: E402
    _is_rate_limit_error,
    _result_metrics,
    _text_metrics,
    normalize_settings,
    write_csv,
)
from inferno_api.rag_model_compare import build_messages as build_messages_baseline  # noqa: E402
from inferno_api.rag_model_compare_upgraded import (  # noqa: E402
    build_messages_upgraded,
)


SCENARIO_SPECS = [
    {
        "scenario_id": "A",
        "preset_id": "high",
        "label": "Scenario A: Critical Threat",
        "filename": "incident_report_high.pdf",
        "profile_summary": (
            "High-risk WUI incident with strong Santa Ana winds, severe spotting, "
            "low initial-attack containment probability, and high evacuation urgency."
        ),
        "expected_posture": [
            "life safety",
            "evacuation support",
            "structure triage",
            "point protection",
            "indirect control",
            "avoid exposed head-fire engagement",
        ],
        "rubric": (
            "The answer should recommend a conservative tactical posture. It should prioritize "
            "life safety, evacuation support, indirect control, and selective structure defense "
            "only where defensible. It should explicitly avoid aggressive exposure of crews in "
            "head-fire or wind-aligned positions."
        ),
    },
    {
        "scenario_id": "B",
        "preset_id": "medium",
        "label": "Scenario B: Medium Risk",
        "filename": "incident_report_boulder.pdf",
        "profile_summary": (
            "Moderate-to-high risk foothills incident with mixed fuels, moderate spotting, "
            "intermittent torching, and moderate containment potential."
        ),
        "expected_posture": [
            "mixed direct and indirect tactics",
            "anchor points",
            "flank action",
            "contingencies for wind shifts",
            "balanced risk management",
        ],
        "rubric": (
            "The answer should recommend a balanced tactical posture. It should justify direct "
            "attack where anchors, flanks, or favorable terrain exist, while also endorsing "
            "indirect line and contingency planning for wind-aligned runs and spotting."
        ),
    },
    {
        "scenario_id": "C",
        "preset_id": "low",
        "label": "Scenario C: Low Risk",
        "filename": "incident_report_low.pdf",
        "profile_summary": (
            "Low-risk initial-attack incident with low spotting, low spread rates, high "
            "containment probability, and strong suppression feasibility."
        ),
        "expected_posture": [
            "direct attack",
            "rapid containment",
            "perimeter hold",
            "mop-up",
            "holding heel and flanks",
        ],
        "rubric": (
            "The answer should recommend an assertive but still doctrine-consistent containment "
            "posture. It should prioritize direct attack, quick edge securement, perimeter hold, "
            "and mop-up while still acknowledging routine safety controls."
        ),
    },
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


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True)


def _score_from_parsed(parsed: dict[str, Any], key: str) -> float | None:
    value = parsed.get(key)
    if value in {None, ""}:
        return None
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def _build_incident_summary(bundle: dict[str, Any], spec: dict[str, Any]) -> str:
    facts = bundle.get("incident_profile", {}).get("facts", {})
    lines = [
        f"Scenario label: {spec['label']}",
        f"Scenario intent: {spec['profile_summary']}",
    ]
    fact_map = {
        "location": "Location",
        "overall_risk_level": "Overall Risk Level",
        "current_fire_behavior": "Current Fire Behavior",
        "weather": "Weather",
        "values_at_risk": "Values at Risk",
        "strategy_tactics": "Strategy / Tactics",
        "safety_concerns": "Safety Concerns",
    }
    for key, label in fact_map.items():
        value = str(facts.get(key) or "").strip()
        if value:
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


def _render_evidence_blocks(context_items: list[dict[str, Any]], *, limit_chars: int = 5000) -> tuple[str, str]:
    incident_lines: list[str] = []
    doctrine_lines: list[str] = []
    for item in context_items:
        source_type = str(item.get("source_type", ""))
        section = str(item.get("section") or item.get("section_path") or item.get("label") or "")
        chunk_id = str(item.get("chunk_id") or item.get("fact_key") or "")
        text = str(item.get("text") or "")
        line = f"section={section} chunk_id={chunk_id} text={text}"
        if source_type == "doctrine":
            doctrine_lines.append(line)
        else:
            incident_lines.append(line)

    def _trim(lines: list[str]) -> str:
        payload = "\n".join(lines)
        if len(payload) <= limit_chars:
            return payload
        return payload[:limit_chars].rstrip() + "\n[... evidence truncated ...]"

    return _trim(incident_lines), _trim(doctrine_lines)


def build_scenario_judge_messages(
    *,
    query: str,
    scenario_spec: dict[str, Any],
    incident_summary: str,
    incident_evidence: str,
    doctrine_evidence: str,
    rag_messages: list[dict[str, str]],
    rag_reply: str,
    retrieval_sources: list[dict[str, Any]],
) -> list[dict[str, str]]:
    schema = {
        "groundedness_score": "number in [0,1]",
        "scenario_appropriateness_score": "number in [0,1]",
        "safety_alignment_score": "number in [0,1]",
        "doctrine_consistency_score": "number in [0,1]",
        "response_summary": "1-2 sentences summarizing the substantive recommendation for this scenario, suitable for a compact results table",
        "paper_summary": "2-3 sentences suitable for a results section, explaining how well the answer fits this scenario",
        "error": "empty string unless evaluation cannot be completed",
        "rationale": "one short sentence",
    }
    system = (
        "You are a strict judge for wildfire incident-plus-doctrine answers. "
        "Evaluate whether the answer is grounded in the supplied scenario and doctrine, "
        "appropriately calibrated to risk, and conservative about safety. Return JSON only."
    )
    user = (
        f"Scenario:\n{incident_summary}\n\n"
        f"User query:\n{query}\n\n"
        f"Scenario rubric:\n{scenario_spec['rubric']}\n\n"
        f"Expected posture cues:\n{_safe_json(scenario_spec['expected_posture'])}\n\n"
        f"Retrieved incident evidence:\n{incident_evidence or '[none]'}\n\n"
        f"Retrieved doctrine evidence:\n{doctrine_evidence or '[none]'}\n\n"
        f"Retrieved sources:\n{_safe_json(retrieval_sources)}\n\n"
        f"RAG system prompt:\n{rag_messages[0]['content'][:1800]}\n\n"
        f"RAG user prompt:\n{rag_messages[1]['content'][:3200]}\n\n"
        f"Assistant answer:\n{rag_reply[:3200]}\n\n"
        "Scoring guidance:\n"
        "- groundedness_score: are claims supported by the supplied incident/doctrine evidence only?\n"
        "- scenario_appropriateness_score: is the tactical posture appropriate for this scenario's severity and conditions?\n"
        "- safety_alignment_score: does the answer prioritize safe engagement and avoid reckless tactics?\n"
        "- doctrine_consistency_score: is the advice consistent with wildfire doctrine and risk-management principles?\n"
        "- If the answer is generic and could apply equally to all scenarios, reduce scenario_appropriateness_score.\n"
        "- Be strict: a score of 1.0 should be rare and reserved for answers that are specific, fully evidence-grounded, tactically appropriate, and clearly caveated.\n"
        "- Penalize answers that overstate confidence, rely on thin doctrine support, paraphrase unsupported incident details, or merely restate the incident report without real doctrine integration.\n"
        "- A competent but imperfect answer should usually land in the 0.6-0.9 range, not 1.0.\n"
        f"Return JSON matching this schema:\n{_safe_json(schema)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def evaluate_scenario_with_judge(
    *,
    judge_llm: LLMService,
    query: str,
    scenario_spec: dict[str, Any],
    incident_summary: str,
    incident_evidence: str,
    doctrine_evidence: str,
    rag_messages: list[dict[str, str]],
    rag_reply: str,
    retrieval_sources: list[dict[str, Any]],
) -> dict[str, Any]:
    judge_messages = build_scenario_judge_messages(
        query=query,
        scenario_spec=scenario_spec,
        incident_summary=incident_summary,
        incident_evidence=incident_evidence,
        doctrine_evidence=doctrine_evidence,
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
    return {
        "judge_system_prompt": judge_messages[0]["content"],
        "judge_user_prompt": judge_messages[1]["content"],
        "judge_reply": reply,
        "judge_error": error or str(parsed.get("error") or ""),
        "judge_latency_ms": latency_ms,
        "groundedness_score": _score_from_parsed(parsed, "groundedness_score"),
        "scenario_appropriateness_score": _score_from_parsed(
            parsed, "scenario_appropriateness_score"
        ),
        "safety_alignment_score": _score_from_parsed(parsed, "safety_alignment_score"),
        "doctrine_consistency_score": _score_from_parsed(parsed, "doctrine_consistency_score"),
        "response_summary": str(parsed.get("response_summary") or "").strip(),
        "paper_summary": str(parsed.get("paper_summary") or "").strip(),
        "judge_rationale": str(parsed.get("rationale") or ""),
        **_result_metrics("judge", completion),
    }


def build_cross_scenario_judge_messages(
    *,
    query: str,
    scenario_payloads: list[dict[str, Any]],
) -> list[dict[str, str]]:
    schema = {
        "scenario_differentiation_score": "number in [0,1]",
        "risk_proportionality_score": "number in [0,1]",
        "cross_scenario_doctrine_consistency_score": "number in [0,1]",
        "response_summary": "1-2 sentences summarizing how the three responses differ across scenarios, suitable for a compact results table",
        "paper_summary": "2-4 sentences suitable for a results section, summarizing how the three responses differ across scenarios",
        "error": "empty string unless evaluation cannot be completed",
        "rationale": "one short sentence",
    }
    system = (
        "You are a strict judge comparing wildfire tactical answers across three scenarios. "
        "Score whether the answers differ appropriately with risk while remaining doctrine-consistent. "
        "Return JSON only."
    )
    scenario_blocks = []
    for payload in scenario_payloads:
        scenario_blocks.append(
            "\n".join(
                [
                    f"{payload['scenario_id']} {payload['scenario_label']}",
                    f"Scenario summary: {payload['scenario_summary']}",
                    f"Expected posture cues: {_safe_json(payload['expected_posture'])}",
                    f"Answer: {payload['reply']}",
                ]
            )
        )
    scenario_blocks_text = "\n\n".join(scenario_blocks)
    user = (
        f"Shared user query:\n{query}\n\n"
        "Expected directional behavior:\n"
        "- Scenario A should be the most conservative and life-safety focused.\n"
        "- Scenario B should recommend a mixed direct/indirect posture.\n"
        "- Scenario C should be the most containment-oriented and permissive of direct attack.\n\n"
        f"Scenario answers:\n\n{scenario_blocks_text}\n\n"
        "Scoring guidance:\n"
        "- scenario_differentiation_score: do the three answers materially differ rather than repeating the same advice?\n"
        "- risk_proportionality_score: do the answers become progressively less conservative from A -> B -> C?\n"
        "- cross_scenario_doctrine_consistency_score: despite differences, do all three remain doctrine-consistent?\n"
        "- Be strict: do not award 1.0 unless the progression from A to B to C is clear, proportional, and well distinguished.\n"
        "- Penalize shallow paraphrase, repeated boilerplate, or only weak separation between scenarios.\n"
        f"Return JSON matching this schema:\n{_safe_json(schema)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def evaluate_cross_scenario_with_judge(
    *,
    judge_llm: LLMService,
    query: str,
    scenario_payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    judge_messages = build_cross_scenario_judge_messages(
        query=query,
        scenario_payloads=scenario_payloads,
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
    return {
        "judge_system_prompt": judge_messages[0]["content"],
        "judge_user_prompt": judge_messages[1]["content"],
        "judge_reply": reply,
        "judge_error": error or str(parsed.get("error") or ""),
        "judge_latency_ms": latency_ms,
        "scenario_differentiation_score": _score_from_parsed(
            parsed, "scenario_differentiation_score"
        ),
        "risk_proportionality_score": _score_from_parsed(parsed, "risk_proportionality_score"),
        "cross_scenario_doctrine_consistency_score": _score_from_parsed(
            parsed, "cross_scenario_doctrine_consistency_score"
        ),
        "response_summary": str(parsed.get("response_summary") or "").strip(),
        "paper_summary": str(parsed.get("paper_summary") or "").strip(),
        "judge_rationale": str(parsed.get("rationale") or ""),
        **_result_metrics("judge", completion),
    }


def _build_prompt_messages(
    *,
    prompt_style: str,
    query: str,
    query_class: str,
    incident_profile: dict[str, Any],
    context_items: list[dict[str, Any]],
) -> list[dict[str, str]]:
    if prompt_style == "baseline":
        return build_messages_baseline(
            query=query,
            query_class=query_class,
            incident_profile=incident_profile,
            context_items=context_items,
        )
    return build_messages_upgraded(
        query=query,
        query_class=query_class,
        incident_profile=incident_profile,
        context_items=context_items,
    )


def run_preset_scenario_eval(
    *,
    settings,
    doctrine_manifest: Path,
    query: str,
    query_class: str,
    prompt_style: str,
    llm_provider: str,
    llm_model: str,
    llm_temperature: float,
    llm_max_tokens: int,
    judge_provider: str,
    judge_model: str,
    judge_temperature: float,
    judge_max_tokens: int,
    retrieval_k: int,
    fail_on_rate_limit: bool,
) -> dict[str, Any]:
    doctrine_store = load_doctrine_assets(doctrine_manifest)
    llm = LLMService(
        settings,
        provider=llm_provider,
        model=llm_model,
        temperature=llm_temperature,
        max_tokens=llm_max_tokens,
    )
    judge_llm = LLMService(
        settings,
        provider=judge_provider,
        model=judge_model,
        temperature=judge_temperature,
        max_tokens=judge_max_tokens,
    )

    rows: list[dict[str, Any]] = []
    scenario_payloads: list[dict[str, Any]] = []
    reports_root = REPO_ROOT / "apps" / "firecastbot" / "incident_reports"

    for scenario_spec in SCENARIO_SPECS:
        incident_report_path = reports_root / scenario_spec["filename"]
        incident_bundle = build_runtime_incident_report(
            incident_report_path.read_bytes(),
            incident_report_path.name,
            settings.embedding_provider,
            settings.embedding_model,
        )
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
        rag_messages = _build_prompt_messages(
            prompt_style=prompt_style,
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
                    f"Rate limit error for {llm_provider}:{llm_model} on {scenario_spec['scenario_id']}: {error}"
                ) from exc
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        incident_summary = _build_incident_summary(incident_bundle, scenario_spec)
        incident_evidence, doctrine_evidence = _render_evidence_blocks(context_items)
        judge_result = {
            "judge_system_prompt": "",
            "judge_user_prompt": "",
            "judge_reply": "",
            "judge_error": "",
            "judge_latency_ms": None,
            "groundedness_score": None,
            "scenario_appropriateness_score": None,
            "safety_alignment_score": None,
            "doctrine_consistency_score": None,
            "response_summary": "",
            "paper_summary": "",
            "judge_rationale": "",
            **_result_metrics("judge", None),
        }
        if not error and reply.strip():
            judge_result = evaluate_scenario_with_judge(
                judge_llm=judge_llm,
                query=query,
                scenario_spec=scenario_spec,
                incident_summary=incident_summary,
                incident_evidence=incident_evidence,
                doctrine_evidence=doctrine_evidence,
                rag_messages=rag_messages,
                rag_reply=reply,
                retrieval_sources=retrieval_sources,
            )
            if fail_on_rate_limit and judge_result["judge_error"] and _is_rate_limit_error(
                judge_result["judge_error"]
            ):
                raise RuntimeError(
                    f"Judge rate limit error for {judge_provider}:{judge_model} on {scenario_spec['scenario_id']}: {judge_result['judge_error']}"
                )

        query_metrics = _text_metrics(query)
        reply_metrics = _text_metrics(reply)
        row = {
            "row_type": "scenario",
            "prompt_style": prompt_style,
            "scenario_id": scenario_spec["scenario_id"],
            "preset_id": scenario_spec["preset_id"],
            "scenario_label": scenario_spec["label"],
            "incident_report": str(incident_report_path),
            "scenario_summary": incident_summary,
            "scenario_profile_summary": scenario_spec["profile_summary"],
            "expected_posture": _safe_json(scenario_spec["expected_posture"]),
            "scenario_rubric": scenario_spec["rubric"],
            "query": query,
            "query_class": query_class,
            "query_char_count": query_metrics["char_count"],
            "query_word_count": query_metrics["word_count"],
            "query_line_count": query_metrics["line_count"],
            "reply": reply,
            "error": error,
            "latency_ms": latency_ms,
            "response_char_count": reply_metrics["char_count"],
            "response_word_count": reply_metrics["word_count"],
            "response_line_count": reply_metrics["line_count"],
            "retrieval_k": retrieval_k,
            "retrieval_context_count": len(context_items),
            "retrieved_incident_evidence": incident_evidence,
            "retrieved_doctrine_evidence": doctrine_evidence,
            "retrieval_sources": _safe_json(retrieval_sources),
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "llm_temperature": llm_temperature,
            "llm_max_tokens": llm_max_tokens,
            "judge_provider": judge_provider,
            "judge_model": judge_model,
            "judge_temperature": judge_temperature,
            "judge_max_tokens": judge_max_tokens,
            "embedding_provider": settings.embedding_provider,
            "embedding_model": settings.embedding_model,
            "doctrine_manifest": str(doctrine_manifest),
            "groundedness_score": judge_result["groundedness_score"],
            "scenario_appropriateness_score": judge_result[
                "scenario_appropriateness_score"
            ],
            "safety_alignment_score": judge_result["safety_alignment_score"],
            "doctrine_consistency_score": judge_result["doctrine_consistency_score"],
            "response_summary": judge_result["response_summary"],
            "paper_summary": judge_result["paper_summary"],
            "judge_latency_ms": judge_result["judge_latency_ms"],
            "judge_error": judge_result["judge_error"],
            "judge_rationale": judge_result["judge_rationale"],
            "judge_system_prompt": judge_result["judge_system_prompt"],
            "judge_user_prompt": judge_result["judge_user_prompt"],
            "judge_reply": judge_result["judge_reply"],
            **_result_metrics("model", completion),
            "judge_response_id": judge_result["judge_response_id"],
            "judge_finish_reason": judge_result["judge_finish_reason"],
            "judge_prompt_tokens": judge_result["judge_prompt_tokens"],
            "judge_completion_tokens": judge_result["judge_completion_tokens"],
            "judge_total_tokens": judge_result["judge_total_tokens"],
            "judge_reasoning_tokens": judge_result["judge_reasoning_tokens"],
            "judge_cached_tokens": judge_result["judge_cached_tokens"],
            "judge_response_metadata": judge_result["judge_response_metadata"],
            "scenario_differentiation_score": None,
            "risk_proportionality_score": None,
            "cross_scenario_doctrine_consistency_score": None,
            "aggregated_model_latency_ms": None,
            "aggregated_model_total_tokens": None,
            "aggregated_response_word_count": None,
        }
        rows.append(row)
        scenario_payloads.append(
            {
                "scenario_id": scenario_spec["scenario_id"],
                "scenario_label": scenario_spec["label"],
                "scenario_summary": incident_summary,
                "expected_posture": scenario_spec["expected_posture"],
                "reply": reply,
            }
        )

    cross_result = evaluate_cross_scenario_with_judge(
        judge_llm=judge_llm,
        query=query,
        scenario_payloads=scenario_payloads,
    )
    if fail_on_rate_limit and cross_result["judge_error"] and _is_rate_limit_error(
        cross_result["judge_error"]
    ):
        raise RuntimeError(
            f"Cross-scenario judge rate limit error for {judge_provider}:{judge_model}: {cross_result['judge_error']}"
        )

    scenario_rows = [row for row in rows if row["row_type"] == "scenario"]
    aggregated_model_latency_ms = round(
        sum(float(row["latency_ms"] or 0.0) for row in scenario_rows), 2
    )
    aggregated_model_total_tokens = sum(int(row["model_total_tokens"] or 0) for row in scenario_rows)
    aggregated_response_word_count = sum(
        int(row["response_word_count"] or 0) for row in scenario_rows
    )
    cross_row = {
        "row_type": "cross_scenario",
        "prompt_style": prompt_style,
        "scenario_id": "ALL",
        "preset_id": "ALL",
        "scenario_label": "A/B/C comparison",
        "incident_report": "",
        "scenario_summary": _safe_json(
            [{item["scenario_id"]: item["scenario_summary"]} for item in scenario_payloads]
        ),
        "scenario_profile_summary": "",
        "expected_posture": _safe_json(
            [{item["scenario_id"]: item["expected_posture"]} for item in scenario_payloads]
        ),
        "scenario_rubric": "",
        "query": query,
        "query_class": query_class,
        "query_char_count": _text_metrics(query)["char_count"],
        "query_word_count": _text_metrics(query)["word_count"],
        "query_line_count": _text_metrics(query)["line_count"],
        "reply": _safe_json({item["scenario_id"]: item["reply"] for item in scenario_payloads}),
        "error": "",
        "latency_ms": None,
        "response_char_count": None,
        "response_word_count": None,
        "response_line_count": None,
        "retrieval_k": retrieval_k,
        "retrieval_context_count": None,
        "retrieved_incident_evidence": "",
        "retrieved_doctrine_evidence": "",
        "retrieval_sources": "",
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "llm_temperature": llm_temperature,
        "llm_max_tokens": llm_max_tokens,
        "judge_provider": judge_provider,
        "judge_model": judge_model,
        "judge_temperature": judge_temperature,
        "judge_max_tokens": judge_max_tokens,
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "doctrine_manifest": str(doctrine_manifest),
        "groundedness_score": None,
        "scenario_appropriateness_score": None,
        "safety_alignment_score": None,
        "doctrine_consistency_score": None,
        "response_summary": cross_result["response_summary"],
        "paper_summary": cross_result["paper_summary"],
        "judge_latency_ms": cross_result["judge_latency_ms"],
        "judge_error": cross_result["judge_error"],
        "judge_rationale": cross_result["judge_rationale"],
        "judge_system_prompt": cross_result["judge_system_prompt"],
        "judge_user_prompt": cross_result["judge_user_prompt"],
        "judge_reply": cross_result["judge_reply"],
        "model_response_id": "",
        "model_finish_reason": "",
        "model_prompt_tokens": None,
        "model_completion_tokens": None,
        "model_total_tokens": None,
        "model_reasoning_tokens": None,
        "model_cached_tokens": None,
        "model_response_metadata": _safe_json({}),
        "judge_response_id": cross_result["judge_response_id"],
        "judge_finish_reason": cross_result["judge_finish_reason"],
        "judge_prompt_tokens": cross_result["judge_prompt_tokens"],
        "judge_completion_tokens": cross_result["judge_completion_tokens"],
        "judge_total_tokens": cross_result["judge_total_tokens"],
        "judge_reasoning_tokens": cross_result["judge_reasoning_tokens"],
        "judge_cached_tokens": cross_result["judge_cached_tokens"],
        "judge_response_metadata": cross_result["judge_response_metadata"],
        "scenario_differentiation_score": cross_result["scenario_differentiation_score"],
        "risk_proportionality_score": cross_result["risk_proportionality_score"],
        "cross_scenario_doctrine_consistency_score": cross_result[
            "cross_scenario_doctrine_consistency_score"
        ],
        "aggregated_model_latency_ms": aggregated_model_latency_ms,
        "aggregated_model_total_tokens": aggregated_model_total_tokens,
        "aggregated_response_word_count": aggregated_response_word_count,
    }
    rows.append(cross_row)

    return {
        "query": query,
        "query_class": query_class,
        "prompt_style": prompt_style,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "judge_provider": judge_provider,
        "judge_model": judge_model,
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "doctrine_manifest": str(doctrine_manifest),
        "retrieval_k": retrieval_k,
        "result_count": len(rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a single incident+doctrine synthesis question across the three FireCastBot "
            "preset incident reports, with Gemini judging both per-scenario quality and "
            "cross-scenario differentiation."
        )
    )
    parser.add_argument(
        "--query",
        default=(
            "Given the current incident conditions, what tactical posture should firefighters "
            "prioritize over the next operational period, and why?"
        ),
    )
    parser.add_argument("--query-class", default="incident+doctrine synthesis")
    parser.add_argument("--prompt-style", choices=["baseline", "upgraded"], default="upgraded")
    parser.add_argument(
        "--doctrine-manifest",
        default="apps/firecastbot/incident_response_docs/doctrine_retrieval_manifest.json",
    )
    parser.add_argument("--embedding-provider", default="openrouter")
    parser.add_argument("--embedding-model", default="openai/text-embedding-3-small")
    parser.add_argument("--llm-provider", default="openrouter")
    parser.add_argument("--llm-model", default="x-ai/grok-4.1-fast")
    parser.add_argument("--llm-temperature", type=float, default=0.2)
    parser.add_argument("--llm-max-tokens", type=int, default=1200)
    parser.add_argument("--judge-provider", default="gemini")
    parser.add_argument("--judge-model", default="gemini-2.5-pro")
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-max-tokens", type=int, default=1200)
    parser.add_argument("--retrieval-k", type=int, default=5)
    parser.add_argument(
        "--output-json",
        default="apps/api/evals/preset_scenario_eval_openrouter_grok_gemini.json",
    )
    parser.add_argument(
        "--output-csv",
        default="apps/api/evals/preset_scenario_eval_openrouter_grok_gemini.csv",
    )
    parser.add_argument("--fail-on-rate-limit", action="store_true", default=True)
    args = parser.parse_args()

    settings = normalize_settings(
        get_settings(),
        {
            "embedding_provider": args.embedding_provider,
            "embedding_model": args.embedding_model,
            "llm_provider": args.llm_provider,
            "llm_model": args.llm_model,
            "llm_temperature": args.llm_temperature,
            "llm_max_tokens": args.llm_max_tokens,
        },
    )
    query_class = str(args.query_class or "").strip() or classify_query(args.query)
    report = run_preset_scenario_eval(
        settings=settings,
        doctrine_manifest=Path(args.doctrine_manifest),
        query=args.query,
        query_class=query_class,
        prompt_style=args.prompt_style,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_temperature=args.llm_temperature,
        llm_max_tokens=args.llm_max_tokens,
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
    print(f"Wrote {report['result_count']} evaluation rows to {output_json} and {args.output_csv}.")


if __name__ == "__main__":
    main()
