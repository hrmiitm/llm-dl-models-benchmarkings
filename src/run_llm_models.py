"""Zero-shot Financial PhraseBank benchmark using src/working_models.json."""
import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data/test_30.csv"
CONFIG = ROOT / "src/working_models.json"
OUT = ROOT / "results"

PROMPT = (
    "Classify the financial sentiment of the following news sentence from an "
    "investor's perspective. Output exactly one word: positive, negative, "
    "or neutral. Do not explain.\n\nSentence:\n"
)
VALID = {"negative", "neutral", "positive"}


def label_from(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.I | re.S).strip()
    if text.lower().strip(" \n\t`\"'.") in VALID:
        return text.lower().strip(" \n\t`\"'.")
    matches = set(re.findall(r"\b(?:negative|neutral|positive)\b", text.lower()))
    return matches.pop() if len(matches) == 1 else None


class ModelAccessError(RuntimeError):
    """An endpoint/model error that cannot be fixed by retrying another row."""


def call_model(session, model, endpoint, sentence, reasoning="low", timeout=120):
    prompt = PROMPT + sentence
    native = endpoint.endswith(":generateContent")
    if native:
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0, "maxOutputTokens": 1024,
                "thinkingConfig": {"thinkingLevel": reasoning.upper()},
            },
        }
    else:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "reasoning_effort": reasoning,
        }
        if "/groq/" in endpoint:
            payload.update({"max_completion_tokens": 512, "reasoning_format": "parsed"})
        else:
            payload["max_tokens"] = 1024
        if "/openrouter/" in endpoint:
            payload.pop("reasoning_effort")
            payload.update({"reasoning": {"effort": reasoning},
                            "provider": {"require_parameters": True}})

    start = time.perf_counter()
    for attempt in range(3):
        try:
            response = session.post(endpoint, json=payload, timeout=timeout)
        except requests.RequestException as exc:
            if attempt == 2:
                raise RuntimeError(str(exc)) from exc
            time.sleep(2 ** attempt)
            continue
        if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
            retry_after = response.headers.get("Retry-After", "")
            delay = float(retry_after) if retry_after.replace(".", "", 1).isdigit() else (60 if response.status_code == 429 else 2 ** attempt)
            time.sleep(min(delay, 60))
            continue
        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError(f"HTTP {response.status_code}: endpoint returned non-JSON data") from exc
        if not response.ok or (isinstance(data, dict) and data.get("error")):
            detail = json.dumps(data, ensure_ascii=False)[:2000]
            message = f"HTTP {response.status_code}: {detail}"
            if response.status_code in (400, 401, 402, 403, 404, 422):
                raise ModelAccessError(message)
            raise RuntimeError(message)
        if not isinstance(data, dict):
            raise RuntimeError("Endpoint returned an unexpected JSON response")
        break
    latency = time.perf_counter() - start

    if native:
        candidates = data.get("candidates") or []
        if not candidates:
            raise RuntimeError(f"Gemini returned no candidates: {json.dumps(data)[:1000]}")
        parts = candidates[0].get("content", {}).get("parts", [])
        raw = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        u = data.get("usageMetadata", {})
        tokens = (u.get("promptTokenCount"), u.get("candidatesTokenCount"),
                  u.get("totalTokenCount"))
    else:
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"Endpoint returned no choices: {json.dumps(data)[:1000]}")
        raw = choices[0].get("message", {}).get("content") or ""
        if isinstance(raw, list):
            raw = "".join(p.get("text", "") for p in raw if isinstance(p, dict))
        u = data.get("usage") or {}
        tokens = (u.get("prompt_tokens"), u.get("completion_tokens"),
                  u.get("total_tokens"))

    return raw, label_from(raw), latency, tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reasoning", choices=("low", "medium", "high"), default="low")
    parser.add_argument("--limit", type=int, help="Test only the first N sentences per model")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    if args.timeout <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("timeout and limit must be positive")
    load_dotenv(ROOT / ".env", override=False)
    token = os.getenv("LLMFOUNDRY_API_KEY") or os.getenv("LLMFOUNDRY_TOKEN")
    if not token:
        parser.error("Set LLMFOUNDRY_API_KEY in the project-root .env or environment")
    auth = token if ":" in token else f"{token}:llm-financial-benchmark"
    base_url = os.getenv("LLMFOUNDRY_BASE_URL", "https://llmfoundry.straivedemo.com").rstrip("/")
    with INPUT.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    with CONFIG.open(encoding="utf-8") as f:
        models = json.load(f)
    if not rows or any(not r.get("id") or not r.get("sentence") for r in rows):
        parser.error("Input CSV must contain nonempty id and sentence columns")
    if len({r["id"] for r in rows}) != len(rows):
        parser.error("Input CSV contains duplicate IDs")
    if args.limit:
        rows = rows[:args.limit]
    if not isinstance(models, list) or not models:
        parser.error("working_models.json must contain a nonempty model list")
    for spec in models:
        if not isinstance(spec, dict) or not isinstance(spec.get("model"), str) or not isinstance(spec.get("endpoints"), list) or not spec["endpoints"]:
            parser.error("Each model needs a name and a nonempty endpoints list")
        endpoint = spec["endpoints"][0]
        if not isinstance(endpoint, str) or not endpoint.startswith(base_url + "/"):
            parser.error(f"Model {spec['model']} endpoint must use LLMFOUNDRY_BASE_URL")
        if not (endpoint.endswith(":generateContent") or endpoint.endswith("/chat/completions")):
            parser.error(f"Unsupported endpoint for {spec['model']}: {endpoint}")
    with (ROOT / "data/answer_key_30.csv").open(encoding="utf-8", newline="") as f:
        ground_truth = {r["id"]: r["label"].strip().lower() for r in csv.DictReader(f)}
    if any(ground_truth.get(r["id"]) not in VALID for r in rows):
        parser.error("Every input ID must have a valid label in data/answer_key_30.csv")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    with requests.Session() as session, (args.output_dir / "llm_predictions.jsonl").open(
        "w", encoding="utf-8"
    ) as file:
        session.headers.update({"Authorization": f"Bearer {auth}",
                                "Content-Type": "application/json", "Cache-Control": "no-cache"})
        for spec in models:
            model = spec["model"]
            endpoint = spec["endpoints"][0]
            successful_latencies = []
            failures = 0
            token_totals = [0, 0, 0]
            token_observed = [False, False, False]
            predictions = {r["id"]: None for r in rows}
            model_start = time.perf_counter()
            permanent_error = None
            print(f"\n{model}: {len(rows)} sentences", flush=True)

            for i, row in enumerate(rows, start=1):
                result = {
                    "id": row["id"], "sentence": row["sentence"],
                    "model": model, "endpoint": endpoint,
                    "temperature": 0, "reasoning_effort": args.reasoning,
                    "predicted_label": None,
                    "confidence": None,
                    "inference_seconds": None,
                    "prompt_tokens": None, "completion_tokens": None,
                    "total_tokens": None, "raw_output": None, "error": None,
                }
                try:
                    if permanent_error:
                        raise ModelAccessError(permanent_error)
                    raw, label, elapsed, tokens = call_model(
                        session, model, endpoint, row["sentence"], args.reasoning, args.timeout
                    )
                    result.update({
                        "predicted_label": label,
                        "raw_output": raw,
                        "inference_seconds": round(elapsed, 6),
                        "prompt_tokens": tokens[0],
                        "completion_tokens": tokens[1],
                        "total_tokens": tokens[2],
                    })
                    predictions[row["id"]] = label
                    if label is None:
                        result["error"] = "No unambiguous sentiment label"
                        failures += 1
                    else:
                        successful_latencies.append(elapsed)
                    for j, v in enumerate(tokens):
                        if isinstance(v, int):
                            token_totals[j] += v
                            token_observed[j] = True
                except Exception as exc:
                    failures += 1
                    message = str(exc).replace(auth, "[REDACTED]").replace(token, "[REDACTED]")
                    result["error"] = message[:2000]
                    if isinstance(exc, ModelAccessError):
                        if permanent_error is None:
                            print(f"  Endpoint rejected the request; remaining rows will be marked failed: {message[:400]}", flush=True)
                        permanent_error = message
                file.write(json.dumps(result, ensure_ascii=False) + "\n")
                file.flush()
                print(f"  {i}/{len(rows)}  {result['predicted_label'] or 'ERROR'}", flush=True)

            # Score against labels kept out of all prompts.
            evaluation = [(ground_truth[k], v) for k, v in predictions.items()
                          if k in ground_truth]
            correct = sum(truth == pred for truth, pred in evaluation)
            accuracy = correct / len(rows)
            f1_scores = []
            for cls in sorted(VALID):
                tp = sum(t == cls and p == cls for t, p in evaluation)
                fp = sum(t != cls and p == cls for t, p in evaluation)
                fn = sum(t == cls and p != cls for t, p in evaluation)
                f1_scores.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)

            summary.append({
                "model": model,
                "endpoint": endpoint,
                "temperature": 0,
                "reasoning_effort": args.reasoning,
                "accuracy": round(accuracy, 4),
                "macro_f1": round(sum(f1_scores) / len(f1_scores), 4),
                "model_run_wall_seconds": round(time.perf_counter() - model_start, 4),
                "samples": len(rows),
                "successful": len(successful_latencies),
                "failed": failures,
                "total_inference_seconds": round(sum(successful_latencies), 4),
                "mean_latency_ms": round(
                    1000 * sum(successful_latencies) / len(successful_latencies), 2
                ) if successful_latencies else None,
                "prompt_tokens": token_totals[0] if token_observed[0] else None,
                "completion_tokens": token_totals[1] if token_observed[1] else None,
                "total_tokens": token_totals[2] if token_observed[2] else None,
            })

    (args.output_dir / "llm_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\nSaved {args.output_dir / 'llm_predictions.jsonl'} and {args.output_dir / 'llm_summary.json'}")


if __name__ == "__main__":
    main()
