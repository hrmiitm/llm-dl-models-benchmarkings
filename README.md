# LLM Foundry endpoint and model probe

Run the sentiment benchmark with the API key in `.env`:

```bash
uv run python src/run_llm_models.py --reasoning low
```

Each run writes these files to `results/` (or `--output-dir`):

- `llm_predictions.jsonl`: predictions and per-sentence `api_cost_usd`, including retries.
- `llm_api_calls.jsonl`: one cost record per HTTP attempt, including failed requests.
- `llm_summary.json`: cost totals per model alongside benchmark metrics.

Cost uses reported USD values from `x-litellm-response-cost`, `usage.cost`, or
`cost_usd` when present. If the API response does not report cost, it stays `null`,
even when token counts are available. No cost estimates are calculated.

Unavailable cost is `null`. If any attempt has unknown cost, the sentence/model
`api_cost_usd` total is `null`; `known_api_cost_usd` gives the available subtotal,
and `api_calls_without_cost` counts unknown attempts. Skipped rows have zero API
calls and no cost value. Output files are replaced on each run.
