import argparse
import json
import os
import re
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

from benchmark_common import (
    ROOT,
    environment,
    evaluate,
    finite_nonnegative,
    load_inputs,
    write_json,
    write_record,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Pinned OpenRouter sentiment benchmark")
    parser.add_argument(
        "--reasoning", choices=["none", "low", "medium", "high"], default="low"
    )
    parser.add_argument(
        "--provider", choices=["auto", "openrouter", "llmfoundry"], default="llmfoundry"
    )
    for flag, options in {
        "models": {"nargs": "+"},
        "limit": {"type": int},
        "smoke": {"action": "store_true"},
        "output-dir": {"type": Path},
    }.items():
        parser.add_argument("--" + flag, **options)
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    load_dotenv(ROOT / ".env")
    direct = args.provider == "openrouter" or (
        args.provider == "auto" and bool(os.getenv("OPENROUTER_API_KEY"))
    )
    key = os.getenv("OPENROUTER_API_KEY" if direct else "LLMFOUNDRY_API_KEY")
    if not key:
        parser.error("Set OPENROUTER_API_KEY or LLMFOUNDRY_API_KEY in .env")
    endpoint = (
        "https://openrouter.ai/api/v1/chat/completions"
        if direct
        else os.getenv("LLMFOUNDRY_BASE_URL", "https://llmfoundry.straivedemo.com").rstrip("/")
        + "/openrouter/v1/chat/completions"
    )

    models = json.loads((ROOT / "src/working_models.json").read_text())
    if args.models and set(args.models) - {m["model"] for m in models}:
        parser.error("Unknown model ID")
    rows, truth = load_inputs(
        ROOT / "data/test_30.csv", ROOT / "data/answer_key_30.csv", args.limit
    )
    if args.smoke and args.limit is None:
        rows = list({truth[r["id"]]: r for r in reversed(rows)}.values())

    out = args.output_dir or ROOT / "results" / ("llm_smoke_pinned" if args.smoke else "")
    out.mkdir(parents=True, exist_ok=True)
    suffix, summaries = f"reasoning_{args.reasoning}", []
    write_json(
        out / f"environment_{suffix}.json",
        environment(ROOT / "data/test_30.csv", ROOT / "data/answer_key_30.csv", args),
    )

    with (
        requests.Session() as session,
        (out / f"llm_predictions_{suffix}.jsonl").open("w") as predictions,
        (out / f"llm_api_calls_{suffix}.jsonl").open("w") as calls,
    ):
        session.headers.update(
            {
                "Authorization": "Bearer "
                + (key if direct or ":" in key else key + ":llm-financial-benchmark"),
                "Cache-Control": "no-cache",
            }
        )
        for model in [m for m in models if not args.models or m["model"] in args.models]:
            pin = {
                "order": [model["provider"]],
                "only": [model["provider"]],
                "allow_fallbacks": False,
                "require_parameters": True,
            }
            meta = dict(
                model=model["model"],
                endpoint=endpoint,
                requested_provider=model["provider"],
                expected_provider=model["provider_name"],
                provider_routing=pin,
                temperature=model.get("temperature"),
                reasoning_effort=args.reasoning,
            )
            records, started = [], time.perf_counter()
            for row in rows:
                result, data, tick = (
                    meta | row | dict(predicted_label=None, error=None),
                    {},
                    time.perf_counter(),
                )
                payload = {
                    "model": model["model"],
                    "provider": pin,
                    "max_tokens": 2048,
                    "messages": [
                        {
                            "role": "user",
                            "content": "Classify financial sentiment from an investor's "
                            "perspective. Answer exactly positive, negative or neutral. "
                            "Sentence: " + row["sentence"],
                        }
                    ],
                }
                if model.get("temperature") is not None:
                    payload["temperature"] = model["temperature"]
                if model.get("reasoning"):
                    payload["reasoning"] = (
                        {"enabled": False}
                        if args.reasoning == "none"
                        else {"effort": args.reasoning}
                    )
                try:
                    response = session.post(endpoint, json=payload, timeout=120)
                    data = response.json()
                    data = data if isinstance(data, dict) else {}
                    response.raise_for_status()
                    if data.get("error") or data.get("provider") != model["provider_name"]:
                        raise ValueError(data.get("error") or "Provider mismatch")
                    raw = data["choices"][0]["message"]["content"] or ""
                    labels = set(
                        re.findall(
                            r"\b(negative|neutral|positive)\b",
                            re.sub(r"<think>.*?</think>", "", raw, flags=re.S | re.I).lower(),
                        )
                    )
                    result.update(
                        raw_output=raw,
                        predicted_label=next(iter(labels)) if len(labels) == 1 else None,
                    )
                    if result["predicted_label"] is None:
                        raise ValueError("Ambiguous sentiment label")
                except (
                    requests.RequestException,
                    ValueError,
                    KeyError,
                    IndexError,
                    TypeError,
                ) as exc:
                    result["error"] = str(exc).replace(key, "[REDACTED]")[:1000]

                usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
                cost = finite_nonnegative(usage.get("cost"))
                usage = usage | {"cost": cost}
                result.update(
                    inference_seconds=time.perf_counter() - tick,
                    usage=usage,
                    api_cost_usd=cost,
                    cost_source="usage.cost" if cost is not None else None,
                    provider=data.get("provider"),
                    provider_verified=data.get("provider") == model["provider_name"],
                    generation_id=data.get("id"),
                    **{
                        k: usage.get(k)
                        for k in ["prompt_tokens", "completion_tokens", "total_tokens"]
                    },
                )
                records.append(result)
                write_record(predictions, result)
                write_record(calls, result)
                print(
                    model["model"],
                    row["id"],
                    result["predicted_label"] or result["error"],
                    flush=True,
                )
            summaries.append(
                meta
                | evaluate(records, truth)
                | dict(
                    api_cost_usd=(
                        sum(r["api_cost_usd"] for r in records)
                        if all(r["api_cost_usd"] is not None for r in records)
                        else None
                    ),
                    provider=(
                        records[0]["provider"]
                        if all(r["provider"] == records[0]["provider"] for r in records)
                        else None
                    ),
                    provider_verified=all(r["provider_verified"] for r in records),
                    total_inference_seconds=sum(r["inference_seconds"] for r in records),
                    mean_latency_ms=(
                        1000 * sum(r["inference_seconds"] for r in records) / len(records)
                    ),
                    model_run_wall_seconds=time.perf_counter() - started,
                )
            )
            write_json(out / f"llm_summary_{suffix}.json", summaries)
    return int(any(s["failed"] for s in summaries))


if __name__ == "__main__":
    raise SystemExit(main())
