# uv run --extra data python src/data_prepare.py
import csv
import random
import zipfile
from pathlib import Path
from huggingface_hub import hf_hub_download

# Download original Financial PhraseBank dataset
archive = hf_hub_download(
    repo_id="takala/financial_phrasebank",
    filename="data/FinancialPhraseBank-v1.0.zip",
    repo_type="dataset",
)

# Read only the 100% annotator agreement subset
with zipfile.ZipFile(archive) as z:
    content = z.read(
        "FinancialPhraseBank-v1.0/Sentences_AllAgree.txt"
    ).decode("iso-8859-1")

rows = []
for i, line in enumerate(content.splitlines()):
    sentence, label = line.rsplit("@", 1)
    rows.append({
        "id": i,
        "sentence": sentence.strip(),
        "label": label.strip()
    })

assert len(rows) == 2264

# Create output directory at project root
out = Path(__file__).resolve().parent.parent / "data"
out.mkdir(exist_ok=True)

def save_csv(filename, data, columns):
    with (out / filename).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns,
                                extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(data)

# Complete dataset
save_csv("all_2264.csv", rows, ["id", "sentence", "label"])

# Balanced random sample: 10 sentences per class
rng = random.Random(42)
sample = []

for label in ["negative", "neutral", "positive"]:
    group = [r for r in rows if r["label"] == label]
    sample.extend(rng.sample(group, 10))

rng.shuffle(sample)

# Model input: labels hidden
save_csv("test_30.csv", sample, ["id", "sentence"])

# Ground truth: evaluate after prediction
save_csv("answer_key_30.csv", sample,
         ["id", "sentence", "label"])

print(f"Total sentences: {len(rows)}")
print(f"Sampled sentences: {len(sample)}")
print(f"Saved files in: {out}")
