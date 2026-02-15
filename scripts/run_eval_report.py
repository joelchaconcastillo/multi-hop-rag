"""Run QA datasets through the pipeline and write an evaluation report."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from multi_hop_rag.pipeline import MultiHopRAGPipeline
from multi_hop_rag.evaluation import evaluate_predictions


def load_jsonl(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def generate_predictions(dataset_path: Path, output_path: Path) -> None:
    dataset = load_jsonl(dataset_path)
    pipeline = MultiHopRAGPipeline()

    predictions: List[Dict[str, str]] = []
    for idx, row in enumerate(dataset, start=1):
        question = row.get("question", "").strip()
        row_id = row.get("id")
        if not question or not row_id:
            continue
        response = pipeline.query(question)
        predictions.append({
            "id": row_id,
            "question": question,
            "prediction": response.get("answer", ""),
        })
        print(f"[{dataset_path.name}] {idx}/{len(dataset)}")

    write_jsonl(output_path, predictions)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run evaluation report for QA datasets.")
    parser.add_argument("--datasets-dir", required=True, help="Directory containing dataset JSONL files.")
    parser.add_argument("--predictions-dir", required=True, help="Directory to write predictions JSONL.")
    parser.add_argument("--report", required=True, help="Markdown report path.")
    parser.add_argument(
        "--only",
        help="Optional comma-separated list of dataset filenames to evaluate.",
    )
    args = parser.parse_args()

    datasets_dir = Path(args.datasets_dir)
    predictions_dir = Path(args.predictions_dir)
    report_path = Path(args.report)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    dataset_paths = sorted(
        {
            path
            for path in datasets_dir.glob("*.jsonl")
            if not path.name.endswith("_predictions.jsonl")
        }
    )

    if args.only:
        allowed = {name.strip() for name in args.only.split(",") if name.strip()}
        dataset_paths = [path for path in dataset_paths if path.name in allowed]

    results = []
    for dataset_path in dataset_paths:
        prediction_path = predictions_dir / dataset_path.name.replace(".jsonl", "_predictions.jsonl")
        generate_predictions(dataset_path, prediction_path)
        metrics = evaluate_predictions(dataset_path, prediction_path)
        results.append({
            "dataset": dataset_path.name,
            **asdict(metrics),
        })

    lines = [
        "# Evaluation Report",
        "",
        "| Dataset | Count | ROUGE-L mean | ROUGE-L std | Cosine mean | Cosine std | LLM score mean | LLM score std |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        lines.append(
            f"| {row['dataset']} | {row['count']} | {row['rouge_l_mean']:.4f} | "
            f"{row['rouge_l_std']:.4f} | {row['cosine_mean']:.4f} | {row['cosine_std']:.4f} | "
            f"{row['llm_score_mean']:.4f} | {row['llm_score_std']:.4f} |"
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
