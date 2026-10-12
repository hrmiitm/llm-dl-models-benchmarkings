"""Repair failed saved predictions only; retain every API attempt for audit."""
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

from benchmark_common import ROOT, LABELS, evaluate, finite_nonnegative, load_inputs
from run_llm_models import predict_one

META = ('model', 'endpoint', 'requested_provider', 'expected_provider',
        'provider_routing', 'temperature', 'reasoning_effort')


def failed(row):
    confidence = finite_nonnegative(row.get('confidence'))
    return bool(row.get('error')) or row.get('predicted_label') not in LABELS or confidence is None or confidence > 1


def atomic_write(path, text):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(path)


def update_summary(summary, records, calls, truth):
    costs = [finite_nonnegative(r.get('api_cost_usd')) for r in records]
    all_costs = [finite_nonnegative(r.get('api_cost_usd')) for r in calls]
    confidence = [r.get('confidence') for r in records]
    total_seconds = sum(r['inference_seconds'] for r in records)
    prior_wall = summary.get('model_run_wall_seconds', 0)
    new_rows = [r for r in records if r.get('replaced_attempt')]
    # Only newly added repair time is added to cumulative model wall time.
    prior_repairs = summary.get('repaired_samples', 0)
    added_time = sum(r['inference_seconds'] for r in new_rows) - summary.get('repair_inference_seconds', 0)
    return summary | evaluate(records, truth) | {
        'api_cost_usd': sum(costs) if all(c is not None for c in costs) else None,
        'cost_basis': 'retained_prediction_attempts',
        'all_attempts_api_cost_usd': sum(all_costs) if all(c is not None for c in all_costs) else None,
        'api_attempts': len(calls), 'unpriced_api_attempts': sum(c is None for c in all_costs),
        'provider': records[0]['provider'] if len({r['provider'] for r in records}) == 1 else None,
        'provider_verified': all(r.get('provider_verified') is True for r in records),
        'mean_confidence': sum(confidence) / len(records) if all(not failed(r) for r in records) else None,
        'confidence_samples': sum(c is not None for c in confidence),
        'total_inference_seconds': total_seconds, 'mean_latency_ms': 1000 * total_seconds / len(records),
        'model_run_wall_seconds': prior_wall + max(0, added_time),
        'repair_inference_seconds': sum(r['inference_seconds'] for r in new_rows),
        'repaired_samples': max(prior_repairs, len(new_rows)),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=ROOT / 'results')
    parser.add_argument('--dry-run', action='store_true', help='List failures without API calls or writes')
    parser.add_argument('--timeout', type=int, default=180)
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    rows, truth = load_inputs(ROOT / 'data/test_30.csv', ROOT / 'data/answer_key_30.csv')
    inputs = {r['id']: r for r in rows}
    models = {m['model']: m for m in json.loads((ROOT / 'src/working_models.json').read_text())}
    pending = []
    for path in sorted(args.results_dir.rglob('llm_predictions*.jsonl')):
        if any('smoke' in p.lower() for p in path.parts):
            continue
        lines = path.read_text().splitlines()
        records = [json.loads(line) for line in lines]
        indices = [i for i, r in enumerate(records) if failed(r)]
        if indices:
            summaries = path.with_name(path.name.replace('predictions', 'summary').replace('.jsonl', '.json'))
            calls = path.with_name(path.name.replace('predictions', 'api_calls'))
            if not summaries.exists() or not calls.exists():
                raise ValueError(f'Missing companion summary/API log for {path}')
            for model_id in {records[i]['model'] for i in indices}:
                group = [r for r in records if r['model'] == model_id]
                if len(group) != len(inputs) or {r['id'] for r in group} != set(inputs) or len({json.dumps({k: r[k] for k in META}, sort_keys=True) for r in group}) != 1:
                    raise ValueError('Repair requires one complete matching test-set configuration per model')
                if sum(r['model'] == model_id for r in json.loads(summaries.read_text())) != 1:
                    raise ValueError('Repair requires exactly one companion summary per model')
            for i in indices:
                r = records[i]
                if r['model'] not in models or r['id'] not in inputs or r['sentence'] != inputs[r['id']]['sentence']:
                    raise ValueError('Saved failed case does not match input or registered model')
                print(f"Pending: {r['model']} / {r['reasoning_effort']} / ID {r['id']}", flush=True)
            pending.append((path, lines, records, indices, summaries, calls))
    if args.dry_run or not pending:
        print(f'{sum(len(p[3]) for p in pending)} failed cases; no requests made.')
        return 0

    load_dotenv(ROOT / '.env')
    endpoints = {'https://openrouter.ai/api/v1/chat/completions': ('OPENROUTER_API_KEY', True),
                 os.getenv('LLMFOUNDRY_BASE_URL', 'https://llmfoundry.straivedemo.com').rstrip('/') + '/openrouter/v1/chat/completions': ('LLMFOUNDRY_API_KEY', False)}
    for _, _, records, indices, _, _ in pending:
        for i in indices:
            if records[i]['endpoint'] not in endpoints or not os.getenv(endpoints[records[i]['endpoint']][0]):
                parser.error('Saved endpoint is not configured or its API key is missing')

    remaining = 0
    with requests.Session() as session:
        for path, lines, records, indices, summary_path, calls_path in pending:
            calls = [json.loads(line) for line in calls_path.read_text().splitlines()]
            summaries = json.loads(summary_path.read_text())
            changed = set()
            for i in indices:
                old = records[i]
                env_name, direct = endpoints[old['endpoint']]
                key = os.environ[env_name]
                session.headers.update({'Authorization': 'Bearer ' + (key if direct or ':' in key else key + ':llm-financial-benchmark'), 'Cache-Control': 'no-cache'})
                model = models[old['model']] | {'provider': old['requested_provider'], 'provider_name': old['expected_provider'], 'temperature': old['temperature']}
                meta = {k: old[k] for k in META}
                result = predict_one(session, model, inputs[old['id']], old['reasoning_effort'], old['endpoint'], key, meta, timeout=args.timeout)
                result['rerun_at_utc'] = datetime.now(timezone.utc).isoformat()
                result['rerun_source_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), ROOT / 'src/run_llm_models.py')}
                calls.append(result.copy())
                if not failed(result):
                    result['replaced_attempt'] = old
                    records[i] = result
                    lines[i] = json.dumps(result, ensure_ascii=False, allow_nan=False)
                    print(f"Repaired: {old['model']} / {old['reasoning_effort']} / ID {old['id']} → {result['predicted_label']}; cost={result['api_cost_usd']}; provider={result['provider']}", flush=True)
                else:
                    remaining += 1
                    print(f"Still failed: {old['model']} / {old['reasoning_effort']} / ID {old['id']}: {result['error']}", flush=True)
                changed.add(old['model'])
                # Save each attempted repair immediately. Existing successful lines stay byte-for-byte intact.
                atomic_write(calls_path, ''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in calls))
                atomic_write(path, '\n'.join(lines) + '\n')
                for j, summary in enumerate(summaries):
                    if summary['model'] in changed:
                        summaries[j] = update_summary(summary, [r for r in records if r['model'] == summary['model']], [r for r in calls if r['model'] == summary['model']], truth)
                atomic_write(summary_path, json.dumps(summaries, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    return int(remaining > 0)


if __name__ == '__main__':
    raise SystemExit(main())
