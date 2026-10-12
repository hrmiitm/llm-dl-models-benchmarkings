"""Shared validation, scoring and reproducibility metadata for both runners."""
import csv
import hashlib
import json
import math
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ("negative", "neutral", "positive")


def load_inputs(input_path, answer_path, limit=None):
    with Path(input_path).open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    with Path(answer_path).open(encoding="utf-8", newline="") as file:
        answers = list(csv.DictReader(file))
    for name, entries in (("input", rows), ("answer key", answers)):
        if not entries or any(not r.get("id") or not r.get("sentence", "").strip() for r in entries):
            raise ValueError(f"{name} needs nonempty id and sentence columns")
        if len({r["id"] for r in entries}) != len(entries):
            raise ValueError(f"{name} contains duplicate IDs")
    truth = {r["id"]: r for r in answers}
    for row in rows:
        answer = truth.get(row["id"])
        if not answer or answer.get("label") not in LABELS:
            raise ValueError(f"Input ID {row['id']} has no valid answer")
        if row["sentence"] != answer["sentence"]:
            raise ValueError(f"Sentence mismatch for ID {row['id']}")
    return rows[:limit] if limit else rows, {k: r["label"] for k, r in truth.items()}


def evaluate(records, truth):
    pairs = [(truth[r["id"]], r.get("predicted_label")) for r in records]
    per_class = {}
    for label in LABELS:
        tp = sum(t == label and p == label for t, p in pairs)
        fp = sum(t != label and p == label for t, p in pairs)
        fn = sum(t == label and p != label for t, p in pairs)
        per_class[label] = {"precision": tp / (tp + fp) if tp + fp else 0,
                            "recall": tp / (tp + fn) if tp + fn else 0,
                            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
                            "support": sum(t == label for t, _ in pairs)}
    n = len(pairs)
    successful = sum(p in LABELS for _, p in pairs)
    return {"samples": n, "successful": successful, "failed": n - successful,
            "accuracy": sum(t == p for t, p in pairs) / n if n else None,
            "macro_f1": sum(x["f1"] for x in per_class.values()) / 3,
            "balanced_accuracy": sum(x["recall"] for x in per_class.values()) / 3,
            "per_class": per_class,
            "confusion_matrix": {t: {p: sum(a == t and (b == p if p != "invalid" else b not in LABELS)
                                                for a, b in pairs)
                                          for p in (*LABELS, "invalid")} for t in LABELS}}


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_record(file, record):
    file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    file.flush()


def environment(input_path, answer_path, args):
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (ROOT / "src").glob("*.py")}
    return {"started_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(), "platform": platform.platform(),
            "git_revision": revision, "source_sha256": source_hashes, "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "inputs": {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (input_path, answer_path)}}


def finite_nonnegative(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if not isinstance(value, bool) and math.isfinite(number) and number >= 0 else None
