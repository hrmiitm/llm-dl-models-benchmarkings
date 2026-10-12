# LLM and DL sentiment benchmarks

Benchmark financial sentiment on 30 balanced Financial PhraseBank sentences.
Inputs and the separate answer key are included in `data/`.
Requires Python 3.12+.

## Setup

```bash
git clone https://github.com/hrmiitm/llm-dl-models-benchmarkings.git
cd llm-dl-models-benchmarkings
python -m venv .venv
source .venv/bin/activate
pip install .
cp .env.example .env
```

## Run LLMs

Set `OPENROUTER_API_KEY` in `.env`, then test before running all 30 sentences:

```bash
python src/run_llm_models.py --smoke
python src/run_llm_models.py --reasoning low
python src/run_llm_models.py --reasoning medium
python src/run_llm_models.py --reasoning high
```

Alternatively, set `LLMFOUNDRY_API_KEY` to use Foundry's OpenRouter endpoint.
Models and provider pins are defined in [working_models.json](src/working_models.json).
Fallback is disabled, and returned providers are checked. Each sentence gets one
request, without automatic retries.

Run a specific model or save a separate run:

```bash
python src/run_llm_models.py --models google/gemini-3.8-flash --limit 3
python src/run_llm_models.py --reasoning low --output-dir results/my_run
```

Outputs include predictions, reported cost, tokens, provider and inference time:

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
!pip install -q -r requirements-colab.txt
!python src/run_dl_models.py --device cuda --require-t4 --smoke
!python src/run_dl_models.py --device cuda --require-t4 --output-dir results/colab_dl_new
```

Download predictions, scores and timing:

```python
!zip -qr benchmark_results.zip results/colab_dl_new
from google.colab import files
files.download('benchmark_results.zip')
```

For local DL runs:

```bash
pip install '.[dl]'
python src/run_dl_models.py --list-models
python src/run_dl_models.py --device cuda --smoke
python src/run_dl_models.py --device cuda
```

Use `--device cpu` without a GPU. First runs download model weights.
Default full-run output: `results/dl/`; smoke output: `results/dl_smoke/`.

## View results

```bash
python scripts/build_results_manifest.py
python -m http.server 8000 --bind 127.0.0.1
```

Open **http://127.0.0.1:8000/**. Rebuild the manifest after adding results, then
click **Refresh data**. Smoke runs are excluded. Dashboard cost estimates are
separate from reported API charges.

## Checks

```bash
python -m unittest discover -s tests -p 'test_*.py'
node --test tests/benchmark-data.test.mjs
```
