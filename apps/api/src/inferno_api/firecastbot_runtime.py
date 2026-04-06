from __future__ import annotations

import io
import json
import math
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from pypdf import PdfReader

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9+/#.&'-]*")

SECTION_ALIASES = {
    "Incident Overview": (
        "incident overview",
        "overview",
        "incident summary",
        "incident information",
        "situation summary",
    ),
    "Current Situation": (
        "current situation",
        "current fire behavior",
        "situation",
        "fire behavior",
        "current conditions",
        "situation update",
    ),
    "Weather": ("weather", "forecast", "fire weather", "current weather"),
    "Resources": ("resources", "assigned resources", "organization", "resource summary"),
    "Objectives": ("objectives", "control objectives", "incident objectives"),
    "Safety Concerns": (
        "safety concerns",
        "hazards",
        "watch out",
        "risk",
        "hazard mitigation",
        "critical safety message",
    ),
    "Operations Plan": (
        "operations plan",
        "strategy",
        "tactics",
        "operations",
        "planned actions",
    ),
    "Command Structure": ("command structure", "command", "organization assignment", "ics organization"),
    "Values at Risk": ("values at risk", "values threatened", "exposures", "assets at risk"),
    "Terrain": ("terrain",),
    "Fuels": ("fuels", "fuel conditions"),
}

FACT_FIELD_ALIASES = {
    "incident_name": ("incident name", "fire name", "incident"),
    "operational_period": ("operational period", "op period", "shift", "period"),
    "location": ("location", "jurisdiction", "unit", "division", "branch"),
    "current_fire_behavior": ("current fire behavior", "fire behavior", "situation", "current situation"),
    "containment": ("containment", "percent contained", "% contained"),
    "weather": ("weather", "forecast", "wind", "relative humidity", "humidity"),
    "fuels": ("fuels", "fuel conditions"),
    "terrain": ("terrain", "slope", "aspect", "drainage"),
    "values_at_risk": ("values at risk", "values threatened", "structures threatened", "critical infrastructure"),
    "strategy_tactics": ("strategy", "tactics", "planned actions", "operations plan"),
    "assigned_resources": ("resources", "assigned resources", "personnel", "engines", "crews", "aircraft"),
    "safety_concerns": ("safety concerns", "hazards", "critical safety", "watch out"),
    "command_structure": ("command structure", "incident commander", "operations section chief", "command"),
    "unresolved_risks": ("unresolved risks", "outstanding concerns", "issues", "constraints"),
}

FACT_OUTPUT_LABELS = {
    "incident_name": "Incident Name",
    "operational_period": "Operational Period",
    "location": "Location",
    "current_fire_behavior": "Current Fire Behavior",
    "containment": "Containment",
    "weather": "Weather",
    "fuels": "Fuels",
    "terrain": "Terrain",
    "values_at_risk": "Values at Risk",
    "strategy_tactics": "Strategy / Tactics",
    "assigned_resources": "Assigned Resources",
    "safety_concerns": "Safety Concerns",
    "command_structure": "Command Structure",
    "unresolved_risks": "Unresolved Risks",
}

QUERY_CLASS_HINTS = {
    "incident-fact": (
        "what resources",
        "assigned resources",
        "who is the ic",
        "containment",
        "weather",
        "location",
        "operational period",
        "what is the fire behavior",
    ),
    "doctrine": (
        "watch out situations",
        "what does doctrine",
        "what does irpg",
        "what is lces",
        "what are the standard fire orders",
        "according to nwcg",
    ),
    "incident+doctrine synthesis": (
        "what strategy",
        "what tactics",
        "containment strategy",
        "next steps",
        "what makes sense here",
        "recommend",
        "how should",
        "what approach",
    ),
    "safety-critical": (
        "risk",
        "safety",
        "hazard",
        "east flank",
        "escape route",
        "safety zone",
        "watch out",
        "urgent",
    ),
}

SECTION_BONUS_HINTS = {
    "Weather": ("weather", "wind", "humidity", "forecast"),
    "Resources": ("resources", "assigned", "engines", "crews", "aircraft"),
    "Safety Concerns": ("safety", "risk", "hazard", "watch out", "lces"),
    "Operations Plan": ("strategy", "tactics", "containment", "operations", "plan"),
    "Command Structure": ("command", "ic", "operations section chief", "divs"),
    "Values at Risk": ("values at risk", "structures", "wui", "exposure"),
    "Current Situation": ("fire behavior", "situation", "spread", "current"),
}

TOPIC_MAP = {
    "lces": ["escape routes", "safety zones", "lookouts", "communications"],
    "containment": ["anchor point", "direct attack", "indirect attack", "line construction"],
    "wui": ["structure triage", "defensible space", "asset protection"],
}


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def slugify(value: str) -> str:
    value = value.casefold()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def tokenize(text: str) -> list[str]:
    return [token.casefold().strip("._-") for token in TOKEN_PATTERN.findall(text)]


def canonicalize_heading(line: str) -> str:
    value = normalize_whitespace(line)
    value = re.sub(r"^[0-9IVXivx.\-() ]+", "", value)
    value = value.rstrip(":")
    return value.casefold()


def detect_section_heading(line: str) -> str | None:
    candidate = canonicalize_heading(line)
    if not candidate:
        return None
    for section, aliases in SECTION_ALIASES.items():
        if candidate == section.casefold():
            return section
        if any(candidate == alias or candidate.startswith(f"{alias} ") for alias in aliases):
            return section
    return None


def split_into_semantic_blocks(text: str, max_chars: int = 1800) -> list[str]:
    paragraphs = [normalize_whitespace(part) for part in re.split(r"\n\s*\n", text) if normalize_whitespace(part)]
    if not paragraphs:
        paragraphs = [normalize_whitespace(text)] if normalize_whitespace(text) else []
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if not current:
            current = paragraph
            continue
        if len(current) + 2 + len(paragraph) <= max_chars:
            current = f"{current}\n\n{paragraph}"
            continue
        chunks.append(current)
        current = paragraph
    if current:
        chunks.append(current)
    return chunks


def extract_pdf_pages(pdf_bytes: bytes) -> list[str]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages: list[str] = []
    for page in reader.pages:
        extracted = page.extract_text() or ""
        extracted = extracted.replace("-\n", "").replace("\u0000", "")
        pages.append(extracted)
    return pages


def sectionize_incident_report(pages: list[str]) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    current = {
        "section": "Incident Overview",
        "page_start": 1,
        "page_end": 1,
        "lines": [],
    }
    for page_number, page_text in enumerate(pages, start=1):
        lines = [line.strip() for line in page_text.splitlines()]
        for line in lines:
            if not line:
                continue
            heading = detect_section_heading(line)
            if heading:
                if current["lines"]:
                    sections.append(
                        {
                            "section": current["section"],
                            "text": "\n".join(current["lines"]),
                            "page_start": current["page_start"],
                            "page_end": current["page_end"],
                        }
                    )
                current = {
                    "section": heading,
                    "page_start": page_number,
                    "page_end": page_number,
                    "lines": [],
                }
                continue
            current["page_end"] = page_number
            current["lines"].append(line)
    if current["lines"]:
        sections.append(
            {
                "section": current["section"],
                "text": "\n".join(current["lines"]),
                "page_start": current["page_start"],
                "page_end": current["page_end"],
            }
        )
    return sections


def _search_label_value(text: str, labels: tuple[str, ...]) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines:
        lowered = line.casefold()
        for label in labels:
            if lowered.startswith(f"{label}:"):
                return normalize_whitespace(line.split(":", 1)[1])
            if lowered.startswith(f"{label} -"):
                return normalize_whitespace(line.split("-", 1)[1])
    return ""


def _section_text(section_map: dict[str, str], *section_names: str) -> str:
    for name in section_names:
        text = section_map.get(name, "")
        if text:
            return normalize_whitespace(text)
    return ""


def build_incident_profile(
    incident_id: str,
    filename: str,
    full_text: str,
    section_map: dict[str, str],
    report_timestamp: str,
) -> dict[str, Any]:
    incident_name = _search_label_value(full_text, FACT_FIELD_ALIASES["incident_name"]) or Path(filename).stem.replace("_", " ")
    operational_period = _search_label_value(full_text, FACT_FIELD_ALIASES["operational_period"])
    location = _search_label_value(full_text, FACT_FIELD_ALIASES["location"]) or _section_text(section_map, "Incident Overview")
    containment = _search_label_value(full_text, FACT_FIELD_ALIASES["containment"])
    weather = _section_text(section_map, "Weather") or _search_label_value(full_text, FACT_FIELD_ALIASES["weather"])
    current_fire_behavior = _section_text(section_map, "Current Situation") or _search_label_value(full_text, FACT_FIELD_ALIASES["current_fire_behavior"])
    values_at_risk = _section_text(section_map, "Values at Risk") or _search_label_value(full_text, FACT_FIELD_ALIASES["values_at_risk"])
    resources = _section_text(section_map, "Resources") or _search_label_value(full_text, FACT_FIELD_ALIASES["assigned_resources"])
    safety = _section_text(section_map, "Safety Concerns") or _search_label_value(full_text, FACT_FIELD_ALIASES["safety_concerns"])
    operations = _section_text(section_map, "Operations Plan", "Objectives") or _search_label_value(full_text, FACT_FIELD_ALIASES["strategy_tactics"])
    command = _section_text(section_map, "Command Structure") or _search_label_value(full_text, FACT_FIELD_ALIASES["command_structure"])
    fuels = _section_text(section_map, "Fuels") or _search_label_value(full_text, FACT_FIELD_ALIASES["fuels"])
    terrain = _section_text(section_map, "Terrain") or _search_label_value(full_text, FACT_FIELD_ALIASES["terrain"])

    unresolved_risks = ""
    if safety:
        unresolved_risks = safety
    elif operations:
        unresolved_risks = operations

    return {
        "incident_id": incident_id,
        "report_timestamp": report_timestamp,
        "facts": {
            "incident_name": incident_name,
            "operational_period": operational_period,
            "location": location,
            "current_fire_behavior": current_fire_behavior,
            "containment": containment,
            "weather": weather,
            "fuels": fuels,
            "terrain": terrain,
            "values_at_risk": values_at_risk,
            "strategy_tactics": operations,
            "assigned_resources": resources,
            "safety_concerns": safety,
            "command_structure": command,
            "unresolved_risks": unresolved_risks,
        },
    }


@lru_cache(maxsize=2)
def get_embedding_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_texts(texts: list[str], model_name: str) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    model = get_embedding_model(model_name)
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.astype(np.float32)


def build_keyword_index(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    postings: dict[str, list[dict[str, Any]]] = defaultdict(list)
    documents: dict[str, dict[str, Any]] = {}
    lengths: list[int] = []
    for chunk in chunks:
        text = str(chunk["text"])
        tokens = tokenize(text)
        tf = Counter(tokens)
        doc_length = len(tokens)
        lengths.append(doc_length)
        documents[str(chunk["chunk_id"])] = {
            "length": doc_length,
            "metadata": chunk,
        }
        for term, count in tf.items():
            postings[term].append({"chunk_id": chunk["chunk_id"], "tf": count})
    doc_count = len(chunks)
    avgdl = sum(lengths) / doc_count if doc_count else 0.0
    vocabulary = {
        term: {
            "df": len(entries),
            "idf": math.log(1 + ((doc_count - len(entries) + 0.5) / (len(entries) + 0.5))) if doc_count else 0.0,
            "postings": entries,
        }
        for term, entries in postings.items()
    }
    return {
        "bm25": {"k1": 1.5, "b": 0.75, "avgdl": avgdl},
        "documents": documents,
        "vocabulary": vocabulary,
        "document_count": doc_count,
    }


def keyword_scores(query: str, keyword_index: dict[str, Any]) -> dict[str, float]:
    query_terms = tokenize(query)
    scores: dict[str, float] = defaultdict(float)
    bm25 = keyword_index["bm25"]
    for term in query_terms:
        term_info = keyword_index["vocabulary"].get(term)
        if not term_info:
            continue
        idf = float(term_info["idf"])
        for posting in term_info["postings"]:
            chunk_id = str(posting["chunk_id"])
            doc_length = float(keyword_index["documents"][chunk_id]["length"] or 1)
            tf = float(posting["tf"])
            denom = tf + bm25["k1"] * (1 - bm25["b"] + bm25["b"] * (doc_length / max(bm25["avgdl"], 1)))
            scores[chunk_id] += idf * ((tf * (bm25["k1"] + 1)) / denom)
    return dict(scores)


def semantic_scores(query: str, embeddings: np.ndarray, chunks: list[dict[str, Any]], model_name: str) -> dict[str, float]:
    if embeddings.size == 0 or not chunks:
        return {}
    query_embedding = embed_texts([query], model_name)
    similarities = embeddings @ query_embedding[0]
    return {str(chunk["chunk_id"]): float(score) for chunk, score in zip(chunks, similarities, strict=False)}


def classify_query(query: str) -> str:
    haystack = query.casefold()
    if any(hint in haystack for hint in QUERY_CLASS_HINTS["doctrine"]):
        return "doctrine"
    if any(hint in haystack for hint in QUERY_CLASS_HINTS["safety-critical"]):
        return "safety-critical"
    if any(hint in haystack for hint in QUERY_CLASS_HINTS["incident+doctrine synthesis"]):
        return "incident+doctrine synthesis"
    if any(hint in haystack for hint in QUERY_CLASS_HINTS["incident-fact"]):
        return "incident-fact"
    return "unclear / missing data"


def section_bonus(query: str, chunk: dict[str, Any]) -> float:
    section = str(chunk.get("section") or chunk.get("section_path") or "")
    for target_section, hints in SECTION_BONUS_HINTS.items():
        if section == target_section and any(hint in query.casefold() for hint in hints):
            return 0.35
    return 0.0


def doctrine_bonus(chunk: dict[str, Any]) -> float:
    authority_rank = float(chunk.get("authority_rank", 0))
    return authority_rank * 0.03


def recency_bonus(chunk: dict[str, Any]) -> float:
    if chunk.get("source_type") != "incident_report":
        return 0.0
    return 0.2


def retrieve_chunks(
    query: str,
    *,
    chunks: list[dict[str, Any]],
    embeddings: np.ndarray,
    keyword_index: dict[str, Any],
    model_name: str,
    limit: int,
) -> list[dict[str, Any]]:
    keyword = keyword_scores(query, keyword_index)
    semantic = semantic_scores(query, embeddings, chunks, model_name)
    ranked: list[tuple[float, dict[str, Any]]] = []
    for chunk in chunks:
        chunk_id = str(chunk["chunk_id"])
        score = semantic.get(chunk_id, 0.0)
        score += 0.25 * keyword.get(chunk_id, 0.0)
        score += section_bonus(query, chunk)
        score += doctrine_bonus(chunk)
        score += recency_bonus(chunk)
        if score > 0:
            ranked.append((score, chunk))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [chunk | {"retrieval_score": round(score, 4)} for score, chunk in ranked[:limit]]


def retrieve_fact_records(query: str, incident_profile: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    facts = incident_profile.get("facts", {})
    query_terms = set(tokenize(query))
    ranked: list[tuple[int, dict[str, Any]]] = []
    for key, value in facts.items():
        if not value:
            continue
        label = FACT_OUTPUT_LABELS[key]
        text = f"{label}: {value}"
        terms = set(tokenize(label + " " + str(value)))
        overlap = len(query_terms & terms)
        if overlap:
            ranked.append(
                (
                    overlap,
                    {
                        "source_type": "incident_fact",
                        "fact_key": key,
                        "label": label,
                        "text": text,
                    },
                )
            )
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [record for _, record in ranked[:limit]]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_doctrine_assets(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = manifest_path.parent
    kb_path = Path(manifest["source_kb"])
    if not kb_path.is_absolute():
        kb_path = base.parent.parent / "chatwithme" / "incident_response_docs" / kb_path.name
    chunks = load_jsonl(kb_path)
    keyword_path = Path(manifest["keyword_index"])
    if not keyword_path.is_absolute():
        keyword_path = base / keyword_path.name
    keyword_index = json.loads(keyword_path.read_text(encoding="utf-8"))
    documents = keyword_index.get("documents", {})
    if isinstance(documents, list):
        keyword_index["documents"] = {
            str(document["chunk_id"]): {
                "length": int(document.get("length", 0)),
                "metadata": document.get("metadata", {}),
            }
            for document in documents
        }
    dense = manifest.get("dense_index") or {}
    vectors_path = base / Path(dense.get("vectors_path", "")).name
    embeddings = np.load(vectors_path) if vectors_path.exists() else np.zeros((0, 0), dtype=np.float32)
    return {
        "chunks": chunks,
        "keyword_index": keyword_index,
        "embeddings": embeddings,
        "topic_map": TOPIC_MAP,
    }


def build_runtime_incident_report(pdf_bytes: bytes, filename: str, model_name: str) -> dict[str, Any]:
    report_timestamp = datetime.now(timezone.utc).isoformat()
    pages = extract_pdf_pages(pdf_bytes)
    sections = sectionize_incident_report(pages)
    full_text = "\n".join(page for page in pages if page.strip())
    fallback_incident_name = Path(filename).stem.replace("_", " ")
    inferred_name = _search_label_value(full_text, FACT_FIELD_ALIASES["incident_name"]) or fallback_incident_name
    incident_id = f"{slugify(inferred_name)}_{datetime.now(timezone.utc).strftime('%Y_%m_%d')}"
    section_map = {section["section"]: section["text"] for section in sections}
    profile = build_incident_profile(incident_id, filename, full_text, section_map, report_timestamp)

    chunks: list[dict[str, Any]] = []
    chunk_counter = 1
    for section in sections:
        for block in split_into_semantic_blocks(section["text"]):
            chunks.append(
                {
                    "chunk_id": f"{incident_id}_{slugify(section['section'])}_{chunk_counter:02d}",
                    "source_type": "incident_report",
                    "incident_id": incident_id,
                    "incident_name": profile["facts"]["incident_name"],
                    "section": section["section"],
                    "text": block,
                    "page_start": section["page_start"],
                    "page_end": section["page_end"],
                    "report_timestamp": report_timestamp,
                    "operational_period": profile["facts"]["operational_period"],
                }
            )
            chunk_counter += 1

    embeddings = embed_texts([str(chunk["text"]) for chunk in chunks], model_name)
    keyword_index = build_keyword_index(chunks)
    return {
        "incident_profile": profile,
        "incident_chunks": chunks,
        "incident_embeddings": embeddings,
        "incident_keyword_index": keyword_index,
    }


def merge_context(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    for group in groups:
        for item in group:
            key = str(item.get("chunk_id") or item.get("fact_key") or item.get("text"))
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return merged


def render_context_items(items: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for item in items:
        source_type = str(item.get("source_type", "unknown"))
        if source_type == "incident_fact":
            lines.append(f"[Incident Report] {item['label']}: {item['text'].split(':', 1)[-1].strip()}")
            continue
        section = str(item.get("section") or item.get("section_path") or "")
        text = normalize_whitespace(str(item.get("text", "")))
        if source_type == "doctrine":
            lines.append(f"[Doctrine] {item.get('doc_title', '')} | {section} | {text}")
        else:
            lines.append(f"[Incident Report] {section} | {text}")
    return "\n".join(lines)


def build_grounded_prompt(
    *,
    query: str,
    query_class: str,
    incident_profile: dict[str, Any] | None,
    context_items: list[dict[str, Any]],
    conversation: list[dict[str, str]],
) -> str:
    incident_facts = incident_profile.get("facts", {}) if incident_profile else {}
    facts_block = "\n".join(
        f"- {FACT_OUTPUT_LABELS[key]}: {value}"
        for key, value in incident_facts.items()
        if value
    ) or "No incident facts are loaded."
    structure = ""
    if query_class in {"incident+doctrine synthesis", "safety-critical"}:
        structure = (
            "Format the answer in 3 labeled parts: "
            "1. Incident-grounded facts, 2. Doctrine-grounded guidance, 3. Suggested interpretation."
        )

    return f"""
Answer the user's question using:
1. Incident-specific facts from the uploaded report
2. Doctrine and safety guidance from the trusted fire-reference knowledge base

Rules:
- Clearly distinguish incident facts from doctrine.
- Do not invent incident details that are not present.
- If incident data is insufficient, say so.
- Prefer safety-grounded, doctrine-backed reasoning.
- Do not present speculative tactical recommendations as certainties.
- Mention when local SOPs, IC direction, or real-time field conditions may override general guidance.
- Cite which source type supports each point: [Incident Report] or [Doctrine].
- Guide decision quality; do not hallucinate command authority.
- Query class: {query_class}
- {structure}

Incident profile:
{facts_block}

Retrieved context:
{render_context_items(context_items)}

Chat history:
{conversation}

User question:
{query}
"""
