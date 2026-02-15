"""Generate applied easy/medium/hard QA datasets from policy PDFs using an LLM."""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from multi_hop_rag.chunking import HierarchicalChunker
from multi_hop_rag.config import Settings


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


def build_llm() -> ChatOpenAI:
    settings = Settings()
    return ChatOpenAI(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        temperature=0.4,
    )


def generate_single_qa(
    llm: ChatOpenAI,
    difficulty: str,
    section: SectionRecord,
) -> Dict[str, str] | None:
    system_prompt = (
        "You write user-friendly grant compliance Q&A based strictly on the given section. "
        "Return JSON with keys question and answer. Keep answers concise and grounded."
    )
    focus = {
        "easy": "basic, single-section question",
        "medium": "applied, practical question that uses the section",
    }[difficulty]
    user_prompt = (
        f"Difficulty: {difficulty} ({focus})\n"
        f"Source: {section.source}\n"
        f"Section: {section.section_id}\n"
        f"Heading: {section.heading}\n"
        f"Body: {section.body}\n\n"
        "Return ONLY JSON: {\"question\": \"...\", \"answer\": \"...\"}"
    )
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    try:
        payload = json.loads(response.content)
    except json.JSONDecodeError:
        return None
    question = payload.get("question")
    answer = payload.get("answer")
    if not question or not answer:
        return None
    return {
        "question": question,
        "answer": answer,
    }


def generate_hard_qa(
    llm: ChatOpenAI,
    sections: List[SectionRecord],
) -> Dict[str, str] | None:
    system_prompt = (
        "You write hard, multi-hop, user-friendly grant compliance Q&A. "
        "The question must require combining ALL provided sections. "
        "Return JSON with keys question and answer."
    )
    user_prompt = (
        "Combine these sections into one hard user question that requires multi-hop reasoning.\n"
        f"Section A ({sections[0].source} {sections[0].section_id}): {sections[0].body}\n\n"
        f"Section B ({sections[1].source} {sections[1].section_id}): {sections[1].body}\n\n"
        f"Section C ({sections[2].source} {sections[2].section_id}): {sections[2].body}\n\n"
        "Return ONLY JSON: {\"question\": \"...\", \"answer\": \"...\"}"
    )
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    try:
        payload = json.loads(response.content)
    except json.JSONDecodeError:
        return None
    question = payload.get("question")
    answer = payload.get("answer")
    if not question or not answer:
        return None
    return {
        "question": question,
        "answer": answer,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate applied easy/medium/hard QA datasets.")
    parser.add_argument("--source", action="append", required=True, help="name=path to PDF")
    parser.add_argument("--output-dir", required=True, help="Output directory for datasets")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    sources = parse_sources(args.source)
    sections: List[SectionRecord] = []
    for name, path in sources:
        sections.extend(collect_sections(name, path))

    if len(sections) < 3:
        raise ValueError("Not enough sections to generate hard questions.")

    random.shuffle(sections)
    llm = build_llm()

    easy_rows: List[Dict[str, str]] = []
    medium_rows: List[Dict[str, str]] = []
    hard_rows: List[Dict[str, str]] = []

    for section in sections:
        if len(easy_rows) < args.count:
            qa = generate_single_qa(llm, "easy", section)
            if qa:
                easy_rows.append({
                    "id": f"applied_easy_{len(easy_rows) + 1}",
                    "difficulty": "easy",
                    "source": section.source,
                    "section": section.section_id,
                    **qa,
                })
        if len(medium_rows) < args.count:
            qa = generate_single_qa(llm, "medium", section)
            if qa:
                medium_rows.append({
                    "id": f"applied_medium_{len(medium_rows) + 1}",
                    "difficulty": "medium",
                    "source": section.source,
                    "section": section.section_id,
                    **qa,
                })
        if len(easy_rows) >= args.count and len(medium_rows) >= args.count:
            break

    idx = 0
    while len(hard_rows) < args.count and idx + 2 < len(sections):
        triple = sections[idx:idx + 3]
        idx += 3
        qa = generate_hard_qa(llm, triple)
        if not qa:
            continue
        hard_rows.append({
            "id": f"applied_hard_{len(hard_rows) + 1}",
            "difficulty": "hard",
            "source": "|".join([sec.source for sec in triple]),
            "section": "|".join([sec.section_id for sec in triple]),
            **qa,
        })

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "applied_easy.jsonl", easy_rows)
    write_jsonl(output_dir / "applied_medium.jsonl", medium_rows)
    write_jsonl(output_dir / "applied_hard.jsonl", hard_rows)

    print("Generated datasets:")
    print(f"  easy: {len(easy_rows)}")
    print(f"  medium: {len(medium_rows)}")
    print(f"  hard: {len(hard_rows)}")


if __name__ == "__main__":
    main()
