# LLM and DL sentiment benchmarks

Benchmark financial sentiment on 30 balanced Financial PhraseBank sentences.
Inputs and the separate answer key are included in `data/`.
Uses [uv](https://docs.astral.sh/uv/getting-started/installation/) and Python 3.12+.

## Setup

```bash
git clone https://github.com/hrmiitm/llm-dl-models-benchmarkings.git
cd llm-dl-models-benchmarkings
uv sync --locked
cp .env.example .env
```

`uv` manages `.venv` and dependencies from `uv.lock`; no activation is needed.

## Run LLMs

Set `LLMFOUNDRY_API_KEY` in `.env` for the default Foundry → OpenRouter gateway.
Test before running all 30 sentences:

```bash
uv run python src/run_llm_models.py --smoke
uv run python src/run_llm_models.py --reasoning low
uv run python src/run_llm_models.py --reasoning medium
uv run python src/run_llm_models.py --reasoning high
```

For direct OpenRouter, set `OPENROUTER_API_KEY` and add `--provider openrouter`:

```bash
uv run python src/run_llm_models.py --provider openrouter --reasoning low
```

Models and provider pins are defined in [working_models.json](src/working_models.json).
Fallback is disabled, and returned providers are checked. Each sentence gets one
request, without automatic retries.

Run a specific model or save a separate run:

```bash
uv run python src/run_llm_models.py --models google/gemini-3.8-flash --limit 3
uv run python src/run_llm_models.py --reasoning low --output-dir results/my_run
```

Outputs include the sentiment label, confidence, reported cost, tokens, provider
and inference time. LLM confidence is a model-reported estimate from 0 to 1, not
a calibrated guarantee that the answer is correct:

```text
results/llm_predictions_reasoning_low.jsonl
results/llm_summary_reasoning_low.json
results/llm_api_calls_reasoning_low.jsonl
results/environment_reasoning_low.json
```

Other reasoning levels use their own suffix. Repeating the same run overwrites
its files. Smoke results go to `results/llm_smoke_pinned/`. Missing cost stays null;
inference time includes the HTTP request and response processing.

## Run DL models on Colab T4

Select **Runtime → Change runtime type → T4 GPU**, then run:

```python
!git clone https://github.com/hrmiitm/llm-dl-models-benchmarkings.git
%cd /content/llm-dl-models-benchmarkings
!pip install -q uv
import sys
!uv pip install --system --python {sys.executable} -r requirements-colab.txt
!uv run --no-project {sys.executable} src/run_dl_models.py --device cuda --require-t4 --smoke
!uv run --no-project {sys.executable} src/run_dl_models.py --device cuda --require-t4 --output-dir /content/colab_dl_new --archive /content/colab_dl_new.zip
```

Colab uses its existing Python runtime and compatible CUDA PyTorch. Nine models
cover FinBERT, RoBERTa, DistilBERT, DeBERTa and BART; list them with `--list-models`.
Models run one at a time in float16 on CUDA. GPU-synchronized inference timing
includes tokenization and postprocessing; model loading and three warmup passes
are measured separately. Outputs also include class probabilities, confidence,
GPU name/memory, dtype, model revisions, package versions and scores.

Download predictions, scores and timing:

```python
from google.colab import files
files.download('/content/colab_dl_new.zip')
```

After downloading, **manually import the Colab results** into your local repo:

```bash
unzip /path/to/colab_dl_new.zip -d results/
uv run python scripts/build_results_manifest.py
```

Use a new output folder/archive name for another run. The archive contains the
result files and input/answer-key snapshots. The DL runner never uploads or commits
results; use this Colab workflow for the benchmark. Local commands are for checks:

```bash
uv sync --locked --extra dl
uv run --extra dl python src/run_dl_models.py --list-models
uv run --extra dl python src/run_dl_models.py --device cuda --smoke --output-dir /tmp/dl_smoke_check
```

Use `--device cpu` without a GPU. First runs download model weights.
Default full-run output: `results/dl/`; smoke output: `results/dl_smoke/`.

## View results

```bash
uv run python scripts/build_results_manifest.py
uv run python -m http.server 8000 --bind 127.0.0.1
```

Open **http://127.0.0.1:8000/**. Rebuild the manifest after adding results, then
click **Refresh data**. Smoke runs are excluded. Dashboard cost estimates are
separate from reported API charges.

## Checks

```bash
uv run python -m unittest discover -s tests -p 'test_*.py'
node --test tests/benchmark-data.test.mjs
```
