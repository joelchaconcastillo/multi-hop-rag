# Evaluation Report

## Baseline Results

| Dataset | Count | ROUGE-L mean | ROUGE-L std | Cosine mean | Cosine std | LLM score mean | LLM score std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| applied_easy.jsonl | 30 | 0.0834 | 0.0492 | 0.6634 | 0.1958 | 0.4067 | 0.4495 |
| applied_medium.jsonl | 30 | 0.0889 | 0.0318 | 0.7212 | 0.1248 | 0.5467 | 0.4470 |
| applied_hard.jsonl | 30 | 0.1487 | 0.0283 | 0.8061 | 0.0710 | 0.8900 | 0.1539 |

## Improvements vs baseline

Baseline predictions archived at reports/predictions_versions/20260215-205423.

- applied_easy.jsonl: ROUGE-L +0.0004, Cosine +0.0215, LLM score +0.1866
- applied_hard.jsonl: ROUGE-L +0.0100, Cosine -0.0082, LLM score +0.0467
- applied_medium.jsonl: ROUGE-L +0.0029, Cosine +0.0336, LLM score +0.2300

## Current Results

| Dataset | Count | ROUGE-L mean | ROUGE-L std | Cosine mean | Cosine std | LLM score mean | LLM score std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| applied_easy.jsonl | 30 | 0.0838 | 0.0490 | 0.6849 | 0.1889 | 0.5933 | 0.4258 |
| applied_medium.jsonl | 30 | 0.0918 | 0.0372 | 0.7548 | 0.0949 | 0.7767 | 0.3451 |
| applied_hard.jsonl | 30 | 0.1587 | 0.0373 | 0.7979 | 0.0799 | 0.9367 | 0.0890 |
