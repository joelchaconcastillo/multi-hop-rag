"""Generate applied, user-friendly hard questions from policy PDFs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from multi_hop_rag.chunking import HierarchicalChunker


@dataclass
class SectionRecord:
    source: str
    section_id: str
    heading: str
    body: str


def extract_pdf_text(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(path)
    text_chunks: List[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text_chunks.append(page_text)
    return "\n".join(text_chunks)


def collect_sections(source: str, pdf_path: Path) -> List[SectionRecord]:
    text = extract_pdf_text(pdf_path)
    chunker = HierarchicalChunker(chunk_size=2000, chunk_overlap=200)
    extracted = chunker._extract_sections(text)
    records: List[SectionRecord] = []
    for idx, section in enumerate(extracted, start=1):
        metadata = section.get("metadata", {})
        section_id = metadata.get("section") or f"section_{idx}"
        heading = metadata.get("heading") or metadata.get("title") or ""
        body = section.get("content", "").strip()
        body = re.sub(r"\s+", " ", body)
        if not body:
            continue
        records.append(SectionRecord(source, str(section_id), str(heading), body))
    return records


def first_sentences(text: str, count: int = 2) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    cleaned = [s.strip() for s in sentences if s.strip()]
    return " ".join(cleaned[:count])


def iter_keyword_questions(section: SectionRecord) -> Iterable[Tuple[str, str]]:
    body = section.body.lower()
    heading = section.heading

    templates = [
        ("alcohol", "Can I pay for alcohol with grant funds?"),
        ("wine", "Can I pay for wine with grant funds?"),
        ("entertainment", "Can I charge entertainment costs to the grant?"),
        ("travel", "Can we use grant funds for travel expenses?"),
        ("meal", "Can we pay for meals with grant funds?"),
        ("equipment", "Can I buy equipment with grant funds?"),
        ("salary", "Can we use grant funds to pay administrative salaries?"),
        ("indirect", "Can indirect costs be charged to this grant?"),
        ("overhead", "Can overhead be charged to this grant?"),
        ("procurement", "What procurement rules must we follow for this grant?"),
        ("subaward", "Do we have to monitor subrecipients for this grant?"),
        ("subrecipient", "What are our responsibilities for subrecipients?"),
        ("matching", "Do we need cost sharing or matching for this grant?"),
        ("program income", "How should we handle program income?"),
        ("audit", "When is an audit required for this grant?"),
        ("record retention", "How long do we need to keep grant records?"),
        ("prior approval", "Do we need prior approval for certain costs?"),
    ]

    for keyword, question in templates:
        if keyword in body:
            yield question, heading


def build_applied_dataset(sections: List[SectionRecord], limit: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    seen_questions = set()

    for section in sections:
        for question, heading in iter_keyword_questions(section):
            if question in seen_questions:
                continue
            answer = first_sentences(section.body, 2)
            if not answer:
                continue
            rows.append({
                "id": f"applied_{len(rows) + 1}",
                "difficulty": "applied_hard",
                "source": section.source,
                "section": section.section_id,
                "question": question,
                "answer": answer,
            })
            seen_questions.add(question)
            if len(rows) >= limit:
                return rows

    for section in sections:
        if len(rows) >= limit:
            break
        if not section.heading:
            continue
        question = f"Can we use grant funds for {section.heading.lower()}?"
        if question in seen_questions:
            continue
        answer = first_sentences(section.body, 2)
        if not answer:
            continue
        rows.append({
            "id": f"applied_{len(rows) + 1}",
            "difficulty": "applied_hard",
            "source": section.source,
            "section": section.section_id,
            "question": question,
            "answer": answer,
        })
        seen_questions.add(question)

    return rows


def write_jsonl(path: Path, rows: List[Dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def parse_sources(values: List[str]) -> List[Tuple[str, Path]]:
    sources: List[Tuple[str, Path]] = []
    for value in values:
        if "=" not in value:
            raise ValueError("--source must be in name=path format")
        name, path = value.split("=", 1)
        sources.append((name.strip(), Path(path.strip())))
    return sources


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate applied hard questions.")
    parser.add_argument("--source", action="append", required=True, help="name=path to PDF")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()

    sources = parse_sources(args.source)
    all_sections: List[SectionRecord] = []
    for name, path in sources:
        all_sections.extend(collect_sections(name, path))

    rows = build_applied_dataset(all_sections, args.count)
    write_jsonl(Path(args.output), rows)
    print(f"Applied dataset written to {args.output} ({len(rows)} questions)")


if __name__ == "__main__":
    main()
