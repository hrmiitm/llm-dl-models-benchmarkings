# Financial sentiment: LLM and DL benchmarks

Compare OpenRouter LLMs and local Hugging Face classifiers on the same balanced
30-sentence Financial PhraseBank sample (10 negative, 10 neutral, 10 positive).
`data/test_30.csv` contains inputs only. The separate answer key is used after
inference, never in prompts. Both CSVs are committed, so cloning is enough to get
the benchmark data. This small sample is a functionality benchmark, not evidence
of generalization; finance models may have seen Financial PhraseBank in training.

## Clone and run LLMs

Python 3.12+:

```bash
git clone https://github.com/hrmiitm/llm-dl-models-benchmarkings.git
cd llm-dl-models-benchmarkings
python -m pip install .
cp .env.example .env
# Set OPENROUTER_API_KEY in .env, then:
python src/run_llm_models.py --provider openrouter --smoke
python src/run_llm_models.py --provider openrouter --reasoning low
```

With uv, use `uv run python src/run_llm_models.py --smoke` instead. Default
`--provider auto` uses a direct OpenRouter key if set, otherwise the existing
LLM Foundry key and its `/openrouter/v1/chat/completions` bridge. You can explicitly
select `--provider llmfoundry`. Credentials stay in `.env` or environment variables.

The config, `src/working_models.json`, contains:

- `anthropic/claude-sonnet-5.5`
- `openai/gpt-6-luna`
- `qwen/qwen3-14b` (small open-weight model)
- `google/gemini-3.1-flash-lite` (light hosted Google model)

Their IDs and supported parameters were checked against the
[OpenRouter catalog](https://openrouter.ai/api/v1/models) on 2026-10-13.
Sonnet and Luna omit temperature because their catalog does not advertise it.
Reasoning uses OpenRouter's `reasoning` object. `--reasoning none` requests disabled
reasoning; providers may still perform mandatory reasoning. Saved usage is the
provider's report, even when its reasoning counts differ from completion counts.

Select models, input files or an output directory:

```bash
python src/run_llm_models.py --models openai/gpt-6-luna qwen/qwen3-14b --limit 3
python src/run_llm_models.py --reasoning medium
python src/run_llm_models.py --reasoning high
```

Each run saves `llm_predictions.jsonl`, `llm_api_calls.jsonl`, `llm_summary.json`
and `environment.json`. Predictions include latency, model/provider, generation
ID, finish reason, raw answer, full usage, token counts, cached/reasoning tokens
and reported API cost. `llm_api_calls.jsonl` records every HTTP attempt, including
failed requests. Costs come from `usage.cost`, `x-litellm-response-cost` or
`cost_usd`; no estimates are inserted into runner output. See
[OpenRouter usage accounting](https://openrouter.ai/docs/cookbook/administration/usage-accounting).

A sentence/model's total `api_cost_usd` includes retries, and becomes `null` when
any attempt has unknown cost. `known_api_cost_usd` retains the known subtotal;
`api_calls_without_cost` counts unknown attempts. No-call skipped rows have
`api_calls=0` and null cost. Summary token totals include attempts and stay null
when any attempted call is missing that token count. Timing includes successful
responses, including invalid answers; model wall time also includes retries and
errors. Permanent access errors skip the remaining rows for that model. Failed or
missing predictions count as errors in accuracy and F1, not discarded samples.

Default outputs are `results/openrouter_low/`, `openrouter_medium/`, or
`openrouter_high/`; smoke outputs go to `results/llm_smoke_openrouter/`. Repeating
a configuration replaces its output files. Both runners return a nonzero exit
code if any selected model fails to return a valid prediction.

## Colab T4: commands only

In Colab, select **Runtime → Change runtime type → T4 GPU**, then run:

```python
!git clone https://github.com/hrmiitm/llm-dl-models-benchmarkings.git
%cd /content/llm-dl-models-benchmarkings
!pip install -q -r requirements-colab.txt
!python src/run_dl_models.py --device cuda --require-t4 --smoke
!python src/run_dl_models.py --device cuda --require-t4 --output-dir results/colab_dl_new
!zip -qr benchmark_results.zip results/colab_dl_new
```

Download the results:

```python
from google.colab import files
files.download('benchmark_results.zip')
```

The requirements preserve Colab's installed CUDA-enabled PyTorch when compatible.
No notebooks, data preparation or API keys are needed for DL inference.

Local DL setup and model selection:

```bash
python -m pip install '.[dl]'
python src/run_dl_models.py --list-models
python src/run_dl_models.py --device cuda --smoke
python src/run_dl_models.py --models finbert finbert_tone deberta_base_zeroshot
# CPU works too, but is slower:
python src/run_dl_models.py --device cpu --models finbert --smoke
# Or with uv:
uv run --extra dl python src/run_dl_models.py --device cuda
```

Seven three-class models: FinBERT tone, ProsusAI FinBERT, Twitter RoBERTa,
multilingual DistilBERT, DeBERTa zero-shot base and large, and BART MNLI.
FinBERT's two variants have different label orders, which the runner validates.
The [FinBERT config](https://huggingface.co/ProsusAI/finbert/blob/main/config.json)
and [FinBERT tone config](https://huggingface.co/yiyanghkust/finbert-tone/blob/main/config.json)
provide their mappings. Zero-shot models use the same three candidates and
financial sentiment hypothesis. Inference is batch size 1, one model loaded at a
time. CUDA defaults to float16 (supported by T4); CPU uses float32. Override with
`--dtype float32` to compare precision. First runs download model weights.

DL outputs: predictions with confidence and validated probabilities, summary and
evaluation with accuracy/macro F1/balanced accuracy/per-class metrics/confusion
matrix, and environment metadata. They include actual GPU name, dtype, resolved
model revision, package versions, seed, input SHA256 hashes, GPU peak allocated
memory, load time, warmup time and synchronized inference latency. Loading and
warmup are excluded from inference timing. `--require-t4` refuses a different GPU;
local smoke evidence is explicitly labeled RTX 4050, not T4. To pin a model, select
one with `--models` and pass its HF commit hash through `--revision`.

## Interactive dashboard

```bash
python scripts/build_results_manifest.py
python -m http.server 8000 --bind 127.0.0.1
```

Open **http://127.0.0.1:8000/** and press **Refresh data** after adding results.
The dashboard reads saved files and computes scores using the answer-key CSV;
HTTP serving is required. Smoke outputs are excluded from the manifest. Missing
or invalid predictions count as errors; default plots use complete runs matching
the answer key. Duplicate configurations use the newest saved run.

Provider references live in `results/pricing.json`. Complete saved API charges
have priority. When charges are unavailable, the dashboard can estimate costs
only for models with known reference rates; **Saved API charges only** disables
estimates. The old pricing file describes older runs and is not a current price
quote for the new model set. Raw outputs are never changed by dashboard estimates.
DL cost scenarios use recorded inference hours × selected hourly rate; they exclude
load, warmup and idle time. Cloud rental scenarios are estimates, not invoices.

## Verification

```bash
python -m unittest discover -s tests -p 'test_*.py'
node --test tests/benchmark-data.test.mjs
node --check assets/dashboard.js
python src/run_llm_models.py --smoke
python src/run_dl_models.py --smoke --device cuda
```

On 2026-10-13, all four configured LLMs passed a one-sentence OpenRouter smoke
through the LLM Foundry bridge, with reported cost and token metadata for every
call. Saved evidence is in `results/llm_smoke_openrouter/` (including the checked
catalog). All seven DL models passed on an RTX 4050 in float16; evidence is in
`results/dl_smoke_final/`. T4 execution is configured but was not measured locally.

Saved smoke results test loading and valid output, not 30-sample model accuracy.
To regenerate the original deterministic sample, install `.[data]` and run
`python src/data_prepare.py` (this replaces the CSVs).
