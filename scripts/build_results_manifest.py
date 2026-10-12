#!/usr/bin/env python3
"""Index sources only; scores are computed by the browser from original files."""
import datetime as dt
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
CONFIG_FILES = {'manifest.json', 'formats.json', 'cloud_rates.json'}
files = []
for path in sorted(RESULTS.rglob('*')):
    parts = path.relative_to(RESULTS).parts
    if not path.is_file() or any('smoke' in p.lower() or p.startswith('.') for p in parts):
        continue
    if path.suffix not in {'.json', '.jsonl', '.csv'} or path.name in CONFIG_FILES:
        continue
    if 'api_calls' in path.stem or path.stem.startswith('environment') or 'data' in parts[:-1]:
        continue  # Duplicate API logs, runtime metadata and dataset snapshots are not prediction runs.
    files.append({'path': path.relative_to(ROOT).as_posix(),
                  'modified_at': dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc).isoformat()})
manifest = {'schema_version': 2, 'ground_truth': 'data/answer_key_30.csv', 'files': files}
RESULTS.mkdir(exist_ok=True)
(RESULTS / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(f'Indexed {len(files)} result sources; smoke runs and duplicate API logs excluded.')
