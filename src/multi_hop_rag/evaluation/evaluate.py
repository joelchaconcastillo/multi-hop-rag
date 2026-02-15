"""Evaluate QA predictions against references."""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from ..config import Settings
from ..embedding import HuggingFaceEmbedder
from .metrics import rouge_l_f1, cosine_similarity


@dataclass
class EvaluationResult:
    """Aggregate evaluation metrics."""

    count: int
    rouge_l_mean: float
    rouge_l_std: float
    cosine_mean: float
    cosine_std: float
    llm_score_mean: float
    llm_score_std: float


def _load_jsonl(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _get_text(row: Dict[str, str], keys: Iterable[str]) -> Optional[str]:
    for key in keys:
        value = row.get(key)
        if value:
            return value
    return None


def evaluate_predictions(
    references_path: Path,
    predictions_path: Path,
    settings: Optional[Settings] = None,
    include_llm_score: bool = True,
) -> EvaluationResult:
    """Evaluate predictions and return aggregate metrics."""
    references = _load_jsonl(references_path)
    predictions = _load_jsonl(predictions_path)

    ref_map = {row["id"]: row for row in references if "id" in row}

    paired_refs: List[str] = []
    paired_preds: List[str] = []

    for row in predictions:
        row_id = row.get("id")
        if not row_id or row_id not in ref_map:
            continue
        reference = _get_text(ref_map[row_id], ("answer", "reference"))
        prediction = _get_text(row, ("prediction", "answer"))
        if not reference or not prediction:
            continue
        paired_refs.append(reference)
        paired_preds.append(prediction)

    if not paired_refs:
        return EvaluationResult(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    rouge_scores = [rouge_l_f1(ref, pred) for ref, pred in zip(paired_refs, paired_preds)]

    if settings is None:
        settings = Settings()
    embedder = HuggingFaceEmbedder(
        api_key=settings.huggingface_api_key,
        model_name=settings.huggingface_embedding_model,
    )
    ref_embeddings = embedder.embed_texts(paired_refs)
    pred_embeddings = embedder.embed_texts(paired_preds)

    cosine_scores = [
        cosine_similarity(ref_emb, pred_emb)
        for ref_emb, pred_emb in zip(ref_embeddings, pred_embeddings)
    ]

    rouge_std = statistics.stdev(rouge_scores) if len(rouge_scores) > 1 else 0.0
    cosine_std = statistics.stdev(cosine_scores) if len(cosine_scores) > 1 else 0.0

    llm_scores: List[float] = []
    if include_llm_score:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            temperature=0.0,
        )

        system_prompt = (
            "You are a strict grader. Score the prediction against the reference on a 0-1 scale. "
            "Return only a number between 0 and 1."
        )

        for reference, prediction in zip(paired_refs, paired_preds):
            user_prompt = (
                "Reference answer:\n"
                f"{reference}\n\n"
                "Prediction:\n"
                f"{prediction}\n\n"
                "Score (0 to 1):"
            )
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])
            match = re.search(r"(0(?:\.\d+)?|1(?:\.0+)?)", str(response.content))
            if match:
                llm_scores.append(float(match.group(1)))

    llm_mean = statistics.mean(llm_scores) if llm_scores else 0.0
    llm_std = statistics.stdev(llm_scores) if len(llm_scores) > 1 else 0.0

    return EvaluationResult(
        count=len(rouge_scores),
        rouge_l_mean=statistics.mean(rouge_scores),
        rouge_l_std=rouge_std,
        cosine_mean=statistics.mean(cosine_scores),
        cosine_std=cosine_std,
        llm_score_mean=llm_mean,
        llm_score_std=llm_std,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate QA predictions.")
    parser.add_argument("--references", required=True, help="Path to reference JSONL file.")
    parser.add_argument("--predictions", required=True, help="Path to predictions JSONL file.")
    parser.add_argument("--output", help="Optional output JSON path.")
    args = parser.parse_args()

    result = evaluate_predictions(Path(args.references), Path(args.predictions))
    payload = {
        "count": result.count,
        "rouge_l_mean": result.rouge_l_mean,
        "rouge_l_std": result.rouge_l_std,
        "cosine_mean": result.cosine_mean,
        "cosine_std": result.cosine_std,
        "llm_score_mean": result.llm_score_mean,
        "llm_score_std": result.llm_score_std,
    }

    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
