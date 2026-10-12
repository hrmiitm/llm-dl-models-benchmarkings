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

## Interactive results dashboard

`index.html` follows the counting benchmark's reasoning comparison layout. It loads the saved JSON/JSONL files and the answer-key CSV, then computes its scores in browser JavaScript. No benchmark scores are embedded in the HTML or assets.

From the project root:

```bash
python scripts/build_results_manifest.py
python -m http.server 8000 --bind 127.0.0.1
```

Open **http://127.0.0.1:8000/**. HTTP serving is required for `fetch`; do not open the HTML with a `file://` URL.

The dashboard offers accuracy, macro F1, balanced accuracy, macro precision/recall, error rate, answer success rate and saved confidence; cost/latency/timing axes; linear/log scales; model and reasoning filters; overall model averages; a confusion matrix; and CSV export.

`results/manifest.json` lists source paths only. Rebuild it after adding or renaming result files, then press **Refresh data**. Smoke-test outputs are excluded. Raw predictions take priority over rounded summary metrics. Missing/invalid predictions count as errors; default plots use only complete runs matching the answer key. Duplicate files for the same configuration use the newest file set without pooling duplicate predictions.

Overall LLM values are means across selected available reasoning levels, for one same-size run. They are not an ensemble, and repeated sentences are not independent new test examples.

### Cost provenance

Provider references live in `results/pricing.json`, with sources and verification dates. Saved `api_cost_usd`, `cost_usd`, `total_cost_usd` or `usage.cost` charges take priority when complete. Known subtotals with unknown attempts are not presented as full API charges.

When a charge is absent, the default dashboard view estimates standard paid token cost from the official provider rates. Billed output/thinking is estimated as `total_tokens - prompt_tokens`; recorded completion tokens are a fallback only when total usage is unavailable. All input is assumed uncached; proxy fees, discounts, taxes and any unreported retry usage are excluded. These estimates change only the dashboard view; raw result files remain untouched. **Saved API charges only** disables that fallback. Unknown costs remain unknown.

DL compute cost is recorded inference seconds / 3,600 × the selected hourly rate. Azure/AWS T4 presets and a custom rate are available. GPU/host rental scenarios based on Colab timings are not invoices or performance measurements on those clouds. Model loading, warmup and idle time are excluded.

The current reference rates were checked on 2026-10-12. Gemini 3.8 Flash's promotional rate expires on 2026-12-31; update the pricing file when it expires. The AWS preset is a published historical reference, not a live quote.

### Verification

```bash
node --test tests/benchmark-data.test.mjs
node --check assets/dashboard.js
```

The page requires no JavaScript package installation or build step. It makes no model API calls and does not read `.env`.
