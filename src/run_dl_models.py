"""Reproducible financial sentiment inference for a Colab T4 or local CPU/CUDA."""
import argparse
import gc
import importlib.metadata
import math
import random
import shutil
import zipfile
import time
from pathlib import Path

from benchmark_common import ROOT, LABELS, environment, evaluate, load_inputs, write_json, write_record

# Three-way sentiment only: binary classifiers are not comparable to this dataset.
MODELS = [
    {"name": "finbert_tone", "model": "yiyanghkust/finbert-tone", "task": "text-classification", "mapping": {"label_0": "neutral", "label_1": "positive", "label_2": "negative"}},
    {"name": "finbert", "model": "ProsusAI/finbert", "task": "text-classification", "mapping": {}},
    {"name": "twitter_roberta", "model": "cardiffnlp/twitter-roberta-base-sentiment-latest", "task": "text-classification", "mapping": {"label_0": "negative", "label_1": "neutral", "label_2": "positive"}},
    {"name": "distilbert_multilingual", "model": "lxyuan/distilbert-base-multilingual-cased-sentiments-student", "task": "text-classification", "mapping": {}},
    {"name": "deberta_base_zeroshot", "model": "MoritzLaurer/deberta-v3-base-zeroshot-v2.0", "task": "zero-shot-classification", "mapping": {}},
    {"name": "deberta_zeroshot", "model": "MoritzLaurer/deberta-v3-large-zeroshot-v2.0", "task": "zero-shot-classification", "mapping": {}},
    {"name": "distilbert_zeroshot", "model": "typeform/distilbert-base-uncased-mnli", "task": "zero-shot-classification", "mapping": {}},
    {"name": "deberta_mnli", "model": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", "task": "zero-shot-classification", "mapping": {}},
    {"name": "bart_mnli", "model": "facebook/bart-large-mnli", "task": "zero-shot-classification", "mapping": {}},
]
HYPOTHESIS = "The financial sentiment of this statement is {}."


def predict(clf, text, task, mapping, max_length=512):
    if task == "zero-shot-classification":
        result = clf(text, candidate_labels=list(LABELS), hypothesis_template=HYPOTHESIS,
                     multi_label=False)
        scores = dict(zip(result["labels"], result["scores"]))
    else:
        result = clf(text, top_k=None, truncation=True, max_length=max_length)
        if result and isinstance(result[0], list):
            result = result[0]
        scores = {mapping.get(x["label"].lower(), x["label"].lower()): float(x["score"]) for x in result}
    if set(scores) != set(LABELS):
        raise ValueError(f"Expected exactly three sentiment classes, got {sorted(scores)}")
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in scores.values()) or abs(sum(scores.values()) - 1) > 1e-3:
        raise ValueError("Invalid probability distribution")
    return {label: float(scores[label]) for label in LABELS}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/test_30.csv")
    parser.add_argument("--answer-key", type=Path, default=ROOT / "data/answer_key_30.csv")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--archive", type=Path, help="Write a zip for manual download and import into results/")
    parser.add_argument("--warmup", type=int, default=3, help="Untimed warmup predictions per model")
    parser.add_argument("--models", nargs="+", help="Model aliases or Hugging Face IDs")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--require-t4", action="store_true", help="Fail unless the selected GPU is a T4")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16"), default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--smoke", action="store_true", help="One sentence per model; fails if any model fails")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--revision", default="main", help="HF revision; a commit hash pins one selected model")
    args = parser.parse_args(argv)
    if args.list_models:
        for spec in MODELS:
            print(f"{spec['name']}: {spec['model']}")
        return 0
    if (args.limit is not None and args.limit <= 0) or args.max_length <= 0 or args.warmup < 1:
        parser.error("limit, max-length and warmup must be positive")
    if args.smoke and args.limit is None:
        args.limit = 1
    args.output_dir = args.output_dir or ROOT / "results" / ("dl_smoke" if args.smoke else "dl")
    specs = MODELS
    if args.models:
        unknown = set(args.models) - {x for s in MODELS for x in (s["name"], s["model"])}
        if unknown:
            parser.error(f"Unknown models: {sorted(unknown)}; use --list-models")
        specs = [s for s in MODELS if s["name"] in args.models or s["model"] in args.models]
    try:
        inputs, truth = load_inputs(args.input, args.answer_key, args.limit)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    # Imports are lazy so listing models and validating data do not initialize CUDA.
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertForSequenceClassification, BertTokenizer, pipeline
    from transformers import __version__ as transformers_version
    from huggingface_hub import hf_hub_download
    cuda = args.device != "cpu" and torch.cuda.is_available()
    if args.device == "cuda" and not cuda:
        parser.error("CUDA is unavailable; select a GPU runtime or use --device cpu")
    device_name = torch.cuda.get_device_name(0) if cuda else "CPU"
    if args.require_t4 and (not cuda or "T4" not in device_name):
        parser.error(f"T4 required, selected device is {device_name}")
    if args.dtype == "float16" and not cuda:
        parser.error("float16 requires CUDA; use float32 on CPU")
    dtype = torch.float16 if args.dtype == "float16" or (args.dtype == "auto" and cuda) else torch.float32
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    def sync():
        if cuda:
            torch.cuda.synchronize()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = environment(args.input, args.answer_key, args)
    env.update({"gpu": device_name, "torch": torch.__version__, "transformers": importlib.metadata.version("transformers"),
                "cuda": torch.version.cuda, "dtype": str(dtype), "model_config": specs,
                "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory if cuda else None,
                "peak_gpu_memory_metric": "torch.cuda.max_memory_allocated; includes resident model, excludes reserved and other-process memory",
                "hypothesis_template": HYPOTHESIS, "batch_size": 1, "timing": "synchronized end-to-end pipeline per sentence, including tokenization, transfers and postprocessing; excludes load and warmup",
                "warmup_iterations": args.warmup, "gpu_compute_capability": list(torch.cuda.get_device_capability(0)) if cuda else None})
    write_json(args.output_dir / "environment.json", env)
    print(f"Device: {device_name}; dtype: {dtype}", flush=True)
    summary = []
    with (args.output_dir / "dl_predictions.jsonl").open("w", encoding="utf-8") as file, torch.inference_mode():
        for spec in specs:
            clf = model = tokenizer = None
            records, latencies = [], []
            load_seconds = warmup_seconds = None
            load_error = None
            revision = None
            tick = time.perf_counter()
            print(f"\nRunning {spec['name']} ({spec['model']})", flush=True)
            try:
                # finbert-tone's legacy config omits model_type; use its explicit tokenizer.
                config_path = Path(hf_hub_download(spec["model"], "config.json", revision=args.revision))
                revision = next((p.name for p in config_path.parents if p.parent.name == "snapshots"), None)
                if not revision:
                    raise RuntimeError("Unable to resolve the Hugging Face snapshot revision")
                tokenizer_class = BertTokenizer if spec["name"] == "finbert_tone" else AutoTokenizer
                tokenizer = tokenizer_class.from_pretrained(spec["model"], revision=revision)
                model_class = BertForSequenceClassification if spec["name"] == "finbert_tone" else AutoModelForSequenceClassification
                tokenizer.model_max_length = min(args.max_length, tokenizer.model_max_length)
                dtype_arg = {"dtype" if int(transformers_version.split(".")[0]) >= 5 else "torch_dtype": dtype}
                model = model_class.from_pretrained(spec["model"], revision=revision, **dtype_arg)
                model.eval()
                clf = pipeline(spec["task"], model=model, tokenizer=tokenizer, device=0 if cuda else -1)
                sync()
                load_seconds = time.perf_counter() - tick
                warmup_start = time.perf_counter()
                for _ in range(args.warmup):
                    predict(clf, inputs[0]["sentence"], spec["task"], spec["mapping"], args.max_length)
                sync()
                warmup_seconds = time.perf_counter() - warmup_start
            except Exception as exc:
                load_error = f"Model load/warmup failed: {type(exc).__name__}: {exc}"[:2000]
                print(load_error, flush=True)
            if cuda:
                torch.cuda.reset_peak_memory_stats()
            for row in inputs:
                record = {"id": row["id"], "sentence": row["sentence"], "model": spec["model"],
                          "model_alias": spec["name"], "model_revision": revision, "device": device_name,
                          "dtype": str(dtype), "predicted_label": None, "confidence": None,
                          "confidence_source": "classifier_probability",
                          "probabilities": None, "inference_seconds": None, "error": load_error}
                if not load_error:
                    try:
                        sync()
                        start = time.perf_counter()
                        scores = predict(clf, row["sentence"], spec["task"], spec["mapping"], args.max_length)
                        sync()
                        elapsed = time.perf_counter() - start
                        prediction = max(scores, key=scores.get)
                        record.update({"predicted_label": prediction, "confidence": scores[prediction],
                                       "probabilities": scores, "inference_seconds": elapsed})
                        latencies.append(elapsed)
                    except Exception as exc:
                        record["error"] = f"{type(exc).__name__}: {exc}"[:2000]
                records.append(record)
                write_record(file, record)
                print(f"  {row['id']}: {record['predicted_label'] or 'ERROR'}", flush=True)
            summary.append({"model": spec["model"], "model_alias": spec["name"], "model_revision": revision,
                            "device": device_name, "dtype": str(dtype), **evaluate(records, truth),
                            "load_seconds": load_seconds, "warmup_seconds": warmup_seconds,
                            "warmup_iterations": args.warmup, "batch_size": 1,
                            "inference_total_seconds": sum(latencies),
                            "average_inference_ms": 1000 * sum(latencies) / len(latencies) if latencies else None,
                            "model_run_wall_seconds": time.perf_counter() - tick,
                            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if cuda else None,
                            "error": load_error})
            write_json(args.output_dir / "dl_summary.json", summary)
            # Release all references before moving on to the next model.
            del clf, model, tokenizer
            gc.collect()
            if cuda:
                torch.cuda.empty_cache()
    write_json(args.output_dir / "dl_evaluation.json", summary)
    if args.archive:
        data_dir = args.output_dir / "data"
        data_dir.mkdir(exist_ok=True)
        for source, name in ((args.input, "test_30.csv"), (args.answer_key, "answer_key_30.csv")):
            destination = data_dir / name
            if source.resolve() != destination.resolve():
                shutil.copyfile(source, destination)
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.archive, "w", zipfile.ZIP_DEFLATED) as archive:
            for name in ("dl_predictions.jsonl", "dl_summary.json", "dl_evaluation.json", "environment.json",
                         "data/test_30.csv", "data/answer_key_30.csv"):
                archive.write(args.output_dir / name, Path(args.output_dir.name) / name)
        print(f"Archive ready for manual download: {args.archive}")
    print(f"\nSaved results in {args.output_dir}")
    return int(any(s["failed"] for s in summary))


if __name__ == "__main__":
    raise SystemExit(main())
