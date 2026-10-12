
import csv
import gc
import json
import time
from pathlib import Path

import torch
from transformers import (
    pipeline,
    BertForSequenceClassification,
    BertTokenizer,
)

LABELS = ("negative", "neutral", "positive")

MODELS = [
    ("finbert_tone", "yiyanghkust/finbert-tone", "text-classification",
     {"label_0": "neutral", "label_1": "positive", "label_2": "negative"}),

    ("twitter_roberta", "cardiffnlp/twitter-roberta-base-sentiment-latest",
     "text-classification",
     {"label_0": "negative", "label_1": "neutral", "label_2": "positive"}),

    ("deberta_zeroshot", "MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
     "zero-shot-classification", {}),

    ("bart_mnli", "facebook/bart-large-mnli",
     "zero-shot-classification", {}),
]

# Load test sentences (run from project root)
with open("data/test_30.csv", encoding="utf-8") as f:
    inputs = list(csv.DictReader(f))

assert inputs, "No test sentences found"

out_dir = Path("results")
out_dir.mkdir(exist_ok=True)

device = 0 if torch.cuda.is_available() else -1
device_name = torch.cuda.get_device_name(0) if device == 0 else "CPU"
print("Device:", device_name)


def sync():
    # GPU operations are asynchronous; synchronize before timing.
    if device == 0:
        torch.cuda.synchronize()


def predict(clf, text, task, mapping):
    if task == "zero-shot-classification":
        result = clf(
            text,
            candidate_labels=list(LABELS),
            hypothesis_template="The financial sentiment of this statement is {}.",
            multi_label=False,
        )
        scores = dict(zip(result["labels"], result["scores"]))
    else:
        result = clf(text, top_k=None, truncation=True)
        if result and isinstance(result[0], list):
            result = result[0]

        scores = {
            mapping.get(x["label"].lower(), x["label"].lower()): x["score"]
            for x in result
        }

    return {label: float(scores[label]) for label in LABELS}


summary = []

with (out_dir / "dl_predictions.jsonl").open("w", encoding="utf-8") as f:
    with torch.inference_mode():
        for name, model_id, task, mapping in MODELS:
            print(f"\nRunning {name} on {len(inputs)} sentences")

            run_start = time.perf_counter()
            if name == "finbert_tone":
                model = BertForSequenceClassification.from_pretrained(model_id)
                tokenizer = BertTokenizer.from_pretrained(model_id)
                clf = pipeline(
                    task,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                )
            else:
                clf = pipeline(task, model=model_id, device=device)
            load_s = time.perf_counter() - run_start

            # Warm up model; exclude this from inference measurements.
            sync()
            warmup_start = time.perf_counter()
            predict(clf, inputs[0]["sentence"], task, mapping)
            sync()
            warmup_s = time.perf_counter() - warmup_start

            latencies = []

            for row in inputs:
                sync()
                start = time.perf_counter()

                scores = predict(clf, row["sentence"], task, mapping)

                sync()
                latency_s = time.perf_counter() - start
                latencies.append(latency_s)

                prediction = max(scores, key=scores.get)

                record = {
                    "id": row["id"],
                    "sentence": row["sentence"],
                    "model": model_id,
                    "predicted_label": prediction,
                    "confidence": scores.get(prediction),
                    "probabilities": scores,
                    "inference_seconds": round(latency_s, 6),
                    "device": device_name,
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            run_s = time.perf_counter() - run_start
            total_s = sum(latencies)

            summary.append({
                "model": model_id,
                "device": device_name,
                "samples": len(inputs),
                "load_seconds": round(load_s, 3),
                "warmup_seconds": round(warmup_s, 3),
                "inference_total_seconds": round(total_s, 3),
                "average_inference_ms": round(
                    total_s * 1000 / len(inputs), 2
                ),
                "model_run_wall_seconds": round(run_s, 3),
            })

            print(f"Average latency: {summary[-1]['average_inference_ms']} ms")

            # Free GPU memory before loading next model
            del clf
            gc.collect()
            if device == 0:
                torch.cuda.empty_cache()

(out_dir / "dl_summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)

print("\nCompleted! Results saved in results/")
