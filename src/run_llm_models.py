"""Financial sentiment benchmark on OpenRouter, directly or via LLM Foundry."""
import argparse
import json
import os
import re
import time
from pathlib import Path
from urllib.parse import urlsplit

import requests
from dotenv import load_dotenv

from benchmark_common import ROOT, LABELS, environment, evaluate, finite_nonnegative, load_inputs, write_json, write_record

PROMPT = ("Classify the financial sentiment of the following news sentence from an "
          "investor's perspective. Output exactly one word: positive, negative, "
          "or neutral. Do not explain.\n\nSentence:\n")
VALID = set(LABELS)
RETRYABLE = {408, 429, 500, 502, 503, 504}


class ModelAccessError(RuntimeError):
    """A model or account failure that should skip remaining rows."""


def label_from(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.I | re.S).strip()
    cleaned = text.lower().strip(" \n\t`\"'.")
    if cleaned in VALID:
        return cleaned
    matches = set(re.findall(r"\b(?:negative|neutral|positive)\b", text.lower()))
    return matches.pop() if len(matches) == 1 else None


def reported_cost(data, headers):
    usage = data.get("usage") or {}
    for source, value in (("usage.cost", usage.get("cost")),
                          ("x-litellm-response-cost", headers.get("x-litellm-response-cost")),
                          ("cost_usd", data.get("cost_usd"))):
        cost = finite_nonnegative(value)
        if cost is not None:
            return cost, source
    return None, None


def cost_totals(attempts):
    missing = sum(a.get("api_cost_usd") is None for a in attempts)
    known = sum(a["api_cost_usd"] for a in attempts if a.get("api_cost_usd") is not None)
    return {"api_calls": len(attempts), "api_calls_without_cost": missing,
            "known_api_cost_usd": known,
            "api_cost_usd": known if attempts and not missing else None}


def call_model(session, spec, endpoint, sentence, reasoning="low", timeout=120,
               max_tokens=2048, retries=2, attempts=None, on_attempt=None):
    """Log every POST, including errors, before deciding whether to retry."""
    attempts = attempts if attempts is not None else []
    payload = {"model": spec["model"], "messages": [{"role": "user", "content": PROMPT + sentence}],
               "max_tokens": max_tokens}
    if spec.get("temperature") is not None:
        payload["temperature"] = spec["temperature"]
    if spec.get("reasoning", False):
        payload["reasoning"] = {"enabled": False} if reasoning == "none" else {"effort": reasoning}
    start = time.perf_counter()
    for index in range(retries + 1):
        tick = time.perf_counter()
        response = None
        data = {}
        error = None
        try:
            response = session.post(endpoint, json=payload, timeout=timeout)
            try:
                data = response.json()
            except ValueError:
                error = f"HTTP {response.status_code}: non-JSON response"
            if not isinstance(data, dict):
                data = {}
                error = "Unexpected JSON response"
            if not response.ok or data.get("error"):
                error = f"HTTP {response.status_code}: {json.dumps(data, ensure_ascii=False)[:1500]}"
        except requests.RequestException as exc:
            error = f"{type(exc).__name__}: request failed"
        cost, source = reported_cost(data, response.headers if response is not None else {})
        usage = data.get("usage") or {}
        entry = {"attempt": index + 1, "http_status": response.status_code if response is not None else None,
                 "inference_seconds": time.perf_counter() - tick, "api_cost_usd": cost,
                 "cost_source": source, "usage": usage, "generation_id": data.get("id"),
                 "response_model": data.get("model"), "provider": data.get("provider"), "error": error}
        attempts.append(entry)
        if on_attempt:
            on_attempt(entry)
        if error:
            permanent = response is not None and response.status_code in (400, 401, 402, 403, 404, 422)
            if permanent:
                raise ModelAccessError(error)
            retryable = response is None or response.status_code in RETRYABLE
            if retryable and index < retries:
                delay = finite_nonnegative(response.headers.get("Retry-After")) if response is not None else None
                time.sleep(min(delay if delay is not None else 2 ** index, 60))
                continue
            raise RuntimeError(error)
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("Endpoint returned no choices")
        choice = choices[0]
        raw = choice.get("message", {}).get("content") or ""
        if isinstance(raw, list):
            raw = "".join(p.get("text", "") for p in raw if isinstance(p, dict))
        return {"raw_output": raw, "predicted_label": label_from(raw),
                "inference_seconds": time.perf_counter() - start, "usage": usage,
                "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                "generation_id": data.get("id"), "provider": data.get("provider"),
                "response_model": data.get("model"), "finish_reason": choice.get("finish_reason")}


def connection(provider):
    # Explicit direct credentials win; Foundry is a compatibility bridge to OpenRouter.
    if provider == "auto":
        provider = "openrouter" if os.getenv("OPENROUTER_API_KEY") else "llmfoundry"
    if provider == "openrouter":
        token = os.getenv("OPENROUTER_API_KEY")
        endpoint = "https://openrouter.ai/api/v1/chat/completions"
    else:
        token = os.getenv("LLMFOUNDRY_API_KEY") or os.getenv("LLMFOUNDRY_TOKEN")
        endpoint = os.getenv("LLMFOUNDRY_BASE_URL", "https://llmfoundry.straivedemo.com").rstrip("/") + "/openrouter/v1/chat/completions"
    if not token:
        raise ValueError(f"Set {'OPENROUTER_API_KEY' if provider == 'openrouter' else 'LLMFOUNDRY_API_KEY'} in .env or the environment")
    auth = token if provider == "openrouter" or ":" in token else token + ":llm-financial-benchmark"
    if urlsplit(endpoint).scheme != "https":
        raise ValueError("API endpoint must use HTTPS")
    return provider, endpoint, auth


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("auto", "openrouter", "llmfoundry"), default="auto")
    parser.add_argument("--config", type=Path, default=ROOT / "src/working_models.json")
    parser.add_argument("--models", nargs="+", help="Exact model IDs from the config")
    parser.add_argument("--reasoning", choices=("none", "low", "medium", "high"), default="low")
    parser.add_argument("--input", type=Path, default=ROOT / "data/test_30.csv")
    parser.add_argument("--answer-key", type=Path, default=ROOT / "data/answer_key_30.csv")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--smoke", action="store_true", help="One sentence per model; nonzero exit if a model fails")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args(argv)
    if args.timeout <= 0 or args.max_tokens <= 0 or args.retries < 0 or (args.limit is not None and args.limit <= 0):
        parser.error("timeout, max-tokens and limit must be positive; retries must be nonnegative")
    if args.smoke and args.limit is None:
        args.limit = 1
    args.output_dir = args.output_dir or ROOT / "results" / ("llm_smoke_openrouter" if args.smoke else f"openrouter_{args.reasoning}")
    load_dotenv(ROOT / ".env", override=False)
    try:
        provider, endpoint, auth = connection(args.provider)
        rows, truth = load_inputs(args.input, args.answer_key, args.limit)
        specs = json.loads(args.config.read_text(encoding="utf-8"))
        if not isinstance(specs, list) or not specs:
            raise ValueError("Config must contain a nonempty list")
        for spec in specs:
            if not isinstance(spec, dict) or not isinstance(spec.get("model"), str) or not spec["model"]:
                raise ValueError("Every config entry needs a model ID")
            if spec.get("endpoints") != ["https://openrouter.ai/api/v1/chat/completions"]:
                raise ValueError("Config endpoints must use OpenRouter chat completions")
        if len({s["model"] for s in specs}) != len(specs):
            raise ValueError("Duplicate model IDs in config")
        if args.models:
            missing = set(args.models) - {s["model"] for s in specs}
            if missing:
                raise ValueError(f"Unknown models: {sorted(missing)}")
            specs = [s for s in specs if s["model"] in args.models]
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = environment(args.input, args.answer_key, args)
    env.update({"api_provider": provider, "endpoint": endpoint, "prompt": PROMPT, "model_config": specs})
    write_json(args.output_dir / "environment.json", env)
    summary = []
    with requests.Session() as session, (args.output_dir / "llm_predictions.jsonl").open("w", encoding="utf-8") as predictions_file, (args.output_dir / "llm_api_calls.jsonl").open("w", encoding="utf-8") as calls_file:
        session.headers.update({"Authorization": f"Bearer {auth}", "Content-Type": "application/json"})
        for spec in specs:
            print(f"\n{spec['model']}: {len(rows)} sentences via {provider}", flush=True)
            model_start = time.perf_counter()
            records, all_attempts = [], []
            permanent_error = None
            for row in rows:
                attempts = []
                result = {"id": row["id"], "sentence": row["sentence"], "model": spec["model"], "endpoint": endpoint,
                          "reasoning_effort": args.reasoning if spec.get("reasoning") else "none",
                          "temperature": spec.get("temperature"), "predicted_label": None,
                          "confidence": None, "raw_output": None, "inference_seconds": None,
                          "prompt_tokens": None, "completion_tokens": None, "total_tokens": None,
                          "error": None, "skipped": bool(permanent_error)}
                def log_attempt(entry):
                    safe = dict(entry)
                    if safe.get("error"):
                        safe["error"] = safe["error"].replace(auth, "[REDACTED]").replace(auth.split(":")[0], "[REDACTED]")
                        entry["error"] = safe["error"]
                    write_record(calls_file, safe | {"id": row["id"], "model": spec["model"], "endpoint": endpoint,
                                                   "reasoning_effort": result["reasoning_effort"]})
                tick = time.perf_counter()
                try:
                    if permanent_error:
                        raise ModelAccessError(permanent_error)
                    result.update(call_model(session, spec, endpoint, row["sentence"], args.reasoning,
                                             args.timeout, args.max_tokens, args.retries, attempts, log_attempt))
                    if result["predicted_label"] is None:
                        result["error"] = "No unambiguous sentiment label"
                except (RuntimeError, ValueError, TypeError) as exc:
                    result["error"] = str(exc).replace(auth, "[REDACTED]").replace(auth.split(":")[0], "[REDACTED]")[:2000]
                    if isinstance(exc, ModelAccessError):
                        permanent_error = result["error"]
                result["request_wall_seconds"] = time.perf_counter() - tick if attempts else None
                result.update(cost_totals(attempts))
                records.append(result)
                all_attempts.extend(attempts)
                write_record(predictions_file, result)
                print(f"  {row['id']}: {result['predicted_label'] or 'ERROR'}; cost={result['api_cost_usd']}", flush=True)
            latencies = [r["inference_seconds"] for r in records if r["inference_seconds"] is not None]
            item = {"model": spec["model"], "endpoint": endpoint, "reasoning_effort": args.reasoning,
                    **evaluate(records, truth), **cost_totals(all_attempts),
                    "model_run_wall_seconds": time.perf_counter() - model_start,
                    "total_inference_seconds": sum(latencies),
                    "mean_latency_ms": 1000 * sum(latencies) / len(latencies) if latencies else None}
            for key in ("prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens", "cached_tokens"):
                values = [a.get("usage", {}).get(key) for a in all_attempts] if key in ("prompt_tokens", "completion_tokens", "total_tokens") else [r.get(key) for r in records if not r["skipped"]]
                item[key] = sum(values) if values and all(isinstance(v, int) and not isinstance(v, bool) for v in values) else None
            summary.append(item)
            write_json(args.output_dir / "llm_summary.json", summary)
    print(f"\nSaved results in {args.output_dir}")
    return int(any(s["failed"] for s in summary))


if __name__ == "__main__":
    raise SystemExit(main())
