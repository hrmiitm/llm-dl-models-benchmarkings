#!/usr/bin/env python3
"""Index benchmark source files; the browser calculates all result metrics."""
import datetime as dt
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
PREFIXES = ('llm_summary', 'llm_predictions', 'llm_api_calls', 'dl_summary', 'dl_predictions', 'dl_evaluation', 'environment')
EXCLUDED = {'llm_smoke', 'llm-smoke', 'smoke', 'smoke-test', 'initial-run', 'decision_models', 'extended'}
files = []
for path in sorted(RESULTS.rglob('*')):
    if not path.is_file() or any(part in EXCLUDED or 'smoke' in part.lower() for part in path.relative_to(RESULTS).parts[:-1]):
        continue
    if path.suffix not in {'.json', '.jsonl'} or not any(path.stem == stem or path.stem.startswith(stem + '_') for stem in PREFIXES):
        continue
    files.append({'path': path.relative_to(ROOT).as_posix(),
                  'modified_at': dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc).isoformat()})
manifest = {'schema_version': 1, 'generated_at': dt.datetime.now(dt.timezone.utc).isoformat(),
            'ground_truth': 'data/answer_key_30.csv', 'files': files}
RESULTS.mkdir(exist_ok=True)
(RESULTS / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(f'Indexed {len(files)} benchmark source files; smoke tests excluded.')
