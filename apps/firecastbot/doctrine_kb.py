from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DOC_SPECS = {
    "RedBook_Final.pdf": {
        "doc_title": "Red Book",
        "doc_short": "redbook",
        "default_domain": "doctrine",
        "start_page": 6,
        "agency": "NIFC",
        "doc_year": 2025,
        "authority_rank": 10,
    },
    "field_operations_guide.pdf": {
        "doc_title": "Field Operations Guide",
        "doc_short": "fog",
        "default_domain": "operations",
        "start_page": 7,
        "agency": "USFA/NFA",
        "doc_year": 2016,
        "authority_rank": 9,
    },
    "pms461.pdf": {
        "doc_title": "IRPG",
        "doc_short": "irpg",
        "default_domain": "safety",
        "start_page": 1,
        "agency": "NWCG",
        "doc_year": 2025,
        "authority_rank": 10,
    },
    "sm_0610.pdf": {
        "doc_title": "WUI Firefighting for the Structural Company Officer",
        "doc_short": "wuifsco",
        "default_domain": "operations",
        "start_page": 15,
        "agency": "USFA/NFA",
        "doc_year": 2014,
        "authority_rank": 8,
    },
    "cal_fire_prescribed_fire_guidebook.pdf": {
        "doc_title": "CAL FIRE Prescribed Fire Guidebook",
        "doc_short": "calfire_rx_guidebook",
        "default_domain": "doctrine",
        "start_page": 1,
        "agency": "CAL FIRE",
        "doc_year": 2019,
        "authority_rank": 7,
    },
    "how_to_prepare_your_home_for_wildfires.pdf": {
        "doc_title": "How to Prepare Your Home for Wildfires",
        "doc_short": "firewise_home_wildfires",
        "default_domain": "wui",
        "start_page": 1,
        "agency": "NFPA Firewise",
        "doc_year": 2024,
        "authority_rank": 6,
    },
}

SKIP_EXACT = {
    "table of contents",
    "this page intentionally left blank.",
    "this page left intentionally blank.",
    "acknowledgments",
    "appendix",
    "glossary/acronyms",
}

DOMAIN_KEYWORDS = [
    ("aviation", ("aviation", "helicopter", "airtanker", "retardant", "helispot", "aircraft", "uas")),
    ("medical", ("medical", "triage", "burn injuries", "cpr", "fatality", "casualty")),
    ("safety", ("safety", "watch out", "lces", "hazard", "lookout", "survival", "stress", "risk")),
    ("command", ("command", "planning", "logistics", "finance", "coordination", "communications", "leadership", "unified command")),
    ("fire_behavior", ("fire behavior", "fire environment", "weather", "fuel", "smoke", "windspeed")),
    ("wui", ("wildland urban interface", "wui", "structure protection")),
    ("doctrine", ("policy", "doctrine", "standards", "organization", "responsibilities", "program")),
]

TOPIC_TAG_RULES = [
    ("lces", ("lces",)),
    ("watch out situations", ("watch out", "18 watch out", "watchout")),
    ("risk management", ("risk management", "risk", "hazard mitigation")),
    ("aviation", ("aviation", "airtanker", "helicopter", "aircraft", "uas", "helispot")),
    ("medical", ("medical", "triage", "medevac", "ems", "injury", "burn")),
    ("incident command system", ("incident command system", "ics", "incident commander")),
    ("communications", ("communications", "radio", "frequency", "dispatch")),
    ("fire behavior", ("fire behavior", "smoke column", "windspeed", "weather", "fuel moisture")),
    ("wui", ("wildland urban interface", "wui", "structure protection")),
    ("leadership", ("leader", "leadership", "commander's intent", "leader’s intent")),
    ("preparedness", ("preparedness", "pre-position", "mobilization", "readiness")),
    ("suppression", ("suppression", "direct attack", "indirect attack", "fireline")),
]


@dataclass
class Chunk:
    doc_type: str
    doc_title: str
    section_path: str
    chunk_id: str
    text: str
    page_start: int
    page_end: int
    domain: str
    priority: str
    source_type: str
    agency: str
    topic_tags: list[str]
    doc_year: int
    authority_rank: int

    def to_dict(self) -> dict[str, object]:
        return {
            "doc_type": self.doc_type,
            "doc_title": self.doc_title,
            "section_path": self.section_path,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "domain": self.domain,
            "priority": self.priority,
            "source_type": self.source_type,
            "agency": self.agency,
            "topic_tags": self.topic_tags,
            "doc_year": self.doc_year,
            "authority_rank": self.authority_rank,
        }


def run_pdftotext(pdf_path: Path) -> list[str]:
    result = subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), "-"],
        capture_output=True,
        text=True,
        check=True,
    )
    pages = result.stdout.split("\f")
    return [page for page in pages if page.strip()]


def split_blocks(page_text: str) -> list[list[str]]:
    lines = [line.rstrip() for line in page_text.splitlines()]
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if line.strip():
            current.append(line)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_heading_text(text: str) -> str:
    cleaned = normalize_whitespace(text)
    cleaned = re.sub(r"\.{2,}\s*[A-Za-z0-9\-]+$", "", cleaned)
    cleaned = re.sub(r"\s+\b[ivxlcdmIVXLCDM]+$", "", cleaned)
    cleaned = re.sub(r"\s+SM\s+\d+-\d+$", "", cleaned)
    cleaned = re.sub(r"\s+\d+-\d+$", "", cleaned)
    return cleaned.strip(" :-")


def is_probably_heading(block_text: str) -> bool:
    text = clean_heading_text(block_text)
    if not text:
        return False
    if len(text) > 130:
        return False
    if text.lower() in SKIP_EXACT:
        return False
    if text.startswith(("•", "-", "*")):
        return False
    if re.search(r"[.!?;]$", text):
        return False
    words = text.split()
    if len(words) > 18:
        return False

    if re.match(r"^(chapter|unit)\s+\d+", text, flags=re.IGNORECASE):
        return True
    if re.match(r"^[A-Z][A-Z0-9/&(),'\- ]+$", text) and len(words) <= 14:
        return True

    title_case_words = [
        word for word in re.split(r"[\s/]+", text)
        if word and any(ch.isalpha() for ch in word)
    ]
    if title_case_words and all(word[:1].isupper() for word in title_case_words[: min(10, len(title_case_words))]):
        return True

    return False


def heading_level(text: str) -> int:
    cleaned = clean_heading_text(text)
    if re.match(r"^(chapter|unit)\s+\d+", cleaned, flags=re.IGNORECASE):
        return 1
    if re.match(r"^[A-Z][A-Z0-9/&(),'\- ]+$", cleaned):
        return 1
    if len(cleaned.split()) <= 6:
        return 2
    return 3


def semantic_heading(text: str) -> str:
    cleaned = clean_heading_text(text)
    chapter_only = re.match(r"^(chapter|unit)\s+\d+\s*$", cleaned, flags=re.IGNORECASE)
    if chapter_only:
        return cleaned.title()

    chapter_with_title = re.match(r"^(chapter|unit)\s+\d+\s+(.*)$", cleaned, flags=re.IGNORECASE)
    if chapter_with_title:
        return chapter_with_title.group(2).strip()

    return cleaned


def domain_for(section_path: str, default_domain: str) -> str:
    haystack = section_path.lower()
    for domain, keywords in DOMAIN_KEYWORDS:
        if any(keyword in haystack for keyword in keywords):
            return domain
    return default_domain


def topic_tags_for(section_path: str, text: str, domain: str) -> list[str]:
    haystack = f"{section_path} {text}".lower()
    tags: list[str] = [domain]
    for tag, keywords in TOPIC_TAG_RULES:
        if any(keyword in haystack for keyword in keywords) and tag not in tags:
            tags.append(tag)
    return tags


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def page_is_toc(page_text: str) -> bool:
    normalized = page_text.lower()
    return "table of contents" in normalized and normalized.count("chapter") + normalized.count("unit") >= 2


def build_chunks_for_doc(pdf_path: Path) -> list[Chunk]:
    spec = DOC_SPECS.get(
        pdf_path.name,
        {
            "doc_title": pdf_path.stem,
            "doc_short": slugify(pdf_path.stem),
            "default_domain": "doctrine",
            "start_page": 1,
            "agency": "Unknown",
            "doc_year": 0,
            "authority_rank": 5,
        },
    )
    pages = run_pdftotext(pdf_path)
    chunks: list[Chunk] = []
    hierarchy: list[str] = []
    chunk_lines: list[str] = []
    chunk_path = ""
    chunk_page_start: int | None = None
    chunk_counter = 1
    start_page = int(spec.get("start_page", 1))

    def flush(page_end: int) -> None:
        nonlocal chunk_lines, chunk_path, chunk_page_start, chunk_counter
        text = normalize_whitespace("\n".join(chunk_lines))
        if not text or not chunk_path or chunk_page_start is None:
            chunk_lines = []
            chunk_path = ""
            chunk_page_start = None
            return

        section_slug = slugify(chunk_path.split(" > ")[-1]) or f"section_{chunk_counter:02d}"
        chunk_id = f"{spec['doc_short']}_{section_slug}_{chunk_counter:02d}"
        domain = domain_for(chunk_path, str(spec["default_domain"]))
        chunks.append(
            Chunk(
                doc_type="doctrine",
                doc_title=str(spec["doc_title"]),
                section_path=chunk_path,
                chunk_id=chunk_id,
                text=text,
                page_start=chunk_page_start,
                page_end=page_end,
                domain=domain,
                priority="high_authority",
                source_type="doctrine",
                agency=str(spec["agency"]),
                topic_tags=topic_tags_for(chunk_path, text, domain),
                doc_year=int(spec["doc_year"]),
                authority_rank=int(spec["authority_rank"]),
            )
        )
        chunk_counter += 1
        chunk_lines = []
        chunk_path = ""
        chunk_page_start = None

    for page_number, page_text in enumerate(pages, start=1):
        if page_number < start_page or page_is_toc(page_text):
            continue

        blocks = split_blocks(page_text)
        pending_chapter_label: str | None = None

        for block in blocks:
            block_text = normalize_whitespace(" ".join(line.strip() for line in block))
            if not block_text:
                continue

            if is_probably_heading(block_text):
                current_heading = semantic_heading(block_text)

                if re.match(r"^(chapter|unit)\s+\d+\s*$", clean_heading_text(block_text), flags=re.IGNORECASE):
                    pending_chapter_label = current_heading
                    continue

                if pending_chapter_label:
                    current_heading = semantic_heading(f"{pending_chapter_label} {current_heading}")
                    pending_chapter_label = None

                level = heading_level(block_text)
                flush(page_number)
                while len(hierarchy) >= level:
                    hierarchy.pop()
                hierarchy.append(current_heading)
                continue

            if pending_chapter_label:
                hierarchy = [pending_chapter_label]
                pending_chapter_label = None

            if not hierarchy:
                hierarchy = ["Front Matter"]

            if chunk_page_start is None:
                chunk_page_start = page_number
                chunk_path = " > ".join(hierarchy)

            chunk_lines.append(block_text)

        if pending_chapter_label and not hierarchy:
            hierarchy = [pending_chapter_label]

    flush(len(pages))
    return chunks


def write_jsonl(chunks: Iterable[Chunk], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=True) + "\n")


def write_summary(chunks: list[Chunk], output_path: Path, docs_found: list[str]) -> None:
    summary = {
        "docs_found": docs_found,
        "docs_expected": sorted(DOC_SPECS.keys()),
        "doc_count": len(docs_found),
        "chunk_count": len(chunks),
        "by_doc": {},
    }
    by_doc: dict[str, int] = {}
    for chunk in chunks:
        by_doc[chunk.doc_title] = by_doc.get(chunk.doc_title, 0) + 1
    summary["by_doc"] = by_doc
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an offline doctrine KB from incident response PDFs.")
    parser.add_argument(
        "--docs-dir",
        default=str(Path("apps/firecastbot/incident_response_docs")),
        help="Directory containing doctrine PDFs.",
    )
    parser.add_argument(
        "--output",
        default=str(Path("apps/firecastbot/incident_response_docs/doctrine_kb.jsonl")),
        help="JSONL output path.",
    )
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir)
    pdf_paths = sorted(docs_dir.glob("*.pdf"))
    all_chunks: list[Chunk] = []
    for pdf_path in pdf_paths:
        all_chunks.extend(build_chunks_for_doc(pdf_path))

    output_path = Path(args.output)
    write_jsonl(all_chunks, output_path)
    write_summary(
        all_chunks,
        output_path.with_suffix(".summary.json"),
        [path.name for path in pdf_paths],
    )

    print(f"Built doctrine KB with {len(all_chunks)} chunks from {len(pdf_paths)} docs.")
    if len(pdf_paths) != len(DOC_SPECS):
        print(
            f"Warning: expected {len(DOC_SPECS)} known doctrine PDFs, found {len(pdf_paths)} in {docs_dir}."
        )


if __name__ == "__main__":
    main()
