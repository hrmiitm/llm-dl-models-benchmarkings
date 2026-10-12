# Add results to the dashboard

1. Put the full run's JSON, JSONL or CSV in `results/` (a subfolder is fine).
2. Run `uv run python scripts/build_results_manifest.py` and click **Refresh results**.
3. If the schema is unrecognized, add a mapping in `results/formats.json`.

The manifest indexes files, not scores. The browser reads original files, parses
arrays, individual objects, or envelopes such as `predictions`, `summaries`,
`records`, `results` and `data`. Smoke folders, `environment*`, `api_calls*`,
config files and `data/` snapshots are excluded. API logs duplicate current
prediction records and must not be added as another run.

## Predictions (preferred)

Use `model`, `model_kind` (`llm` or `dl`), `id`, `predicted_label` and optional
`sentence`. IDs must match the entire answer key and provided sentences must
match too. Labels are `negative`, `neutral`, `positive`. Include `confidence`
(0–1), `api_cost_usd` (USD per example), `inference_seconds`, `reasoning_effort`,
`requested_provider`, `endpoint` and `temperature` when available. DL records
can include `device` and `model_revision`.

Accuracy is rescored against `data/answer_key_30.csv`. Any failure, invalid label,
missing/duplicate/foreign ID, or mismatched sentence excludes the whole run from
both graphs. A mean confidence requires valid confidence on every example.
Missing costs remain unknown; a known subtotal is never treated as a total.
Zero cost is allowed on the linear axis, but cannot appear on a log axis.

## Summaries and custom field names

An aggregate must have accuracy, the full sample count, and either `failed: 0`
or `successful` equal to that count. It must also declare the same `test_set`
path as the manifest, or list every test ID in `dataset_ids`. This is a declaration,
not independent verification: the tooltip labels such data as a saved aggregate.
`confidence_samples`, when provided, must cover the full test set.

Example adapter for a new exporter; replace the paths/fields with your own:

```json
{
  "adapters": [{
    "pattern": "results/vendor/*.json",
    "family": "llm",
    "role": "summary",
    "test_set": "data/answer_key_30.csv",
    "fields": {
      "model": "model_name",
      "samples": "n",
      "failed": "failures",
      "accuracy": "metrics.score",
      "confidence": "metrics.certainty",
      "cost": "billing.usd",
      "totalSeconds": "timing.total_ms"
    },
    "units": {"accuracy": "percent", "confidence": "percent", "totalSeconds": "ms"}
  }]
}
```

Mapping keys: `model`, `kind`, `effort`, `id`, `sentence`, `label`, `confidence`,
`accuracy`, `cost`, `seconds`, `totalSeconds`, `wallSeconds`, `samples`,
`successful`, `failed`, `provider`, `endpoint`, `device`, `temperature`, `revision`.
Fields support dotted paths; patterns support `*`. Units default to fractions,
seconds and USD. Use explicit `percent` for 0–100 data and `ms` for milliseconds.
An optional `aliases` object adds common field-name lists globally. The first
matching adapter applies; an adapter's `role` is optional for mixed envelopes.

Existing prediction/summary/evaluation filenames join by their shared stem and
folder; envelopes in a single file join too. Model, family, reasoning, provider
pin/routing, endpoint, temperature, revision, device and dtype distinguish configurations.
For duplicate exports of the same configuration, the latest whole run wins using
file modification times in the manifest. Keep only the intended comparison runs
in `results/`, or remove old exports before rebuilding; runs are never pooled.

## Cost and confidence

LLM cost sums saved charges for the retained prediction attempts. A repaired
row includes its original `replaced_attempt`; API logs retain all requests, and
summaries separately report all-attempt cost (null when any charge is unknown).
Mean latency also covers retained predictions, while cumulative model wall time
includes the observed earlier attempt and repair request time. DL cost = recorded seconds / 3,600 × selected
USD/hour; T4 presets require a recorded T4 device. Choose inference-only or
`model_run_wall_seconds` (includes measured loading/warmup/inference), or map
`wallSeconds`. Provisioning, idle time, ancillary charges and cloud billing
minimums are excluded. Google GPU-only pricing excludes the CPU/RAM host.
Rate sources/date are in `results/cloud_rates.json`; update them when needed.

The confidence graph accepts only complete LLM configurations with confidence.
It compares mean reported confidence to accuracy; this is not an ECE/reliability
diagram and does not establish probability calibration. Select Detail for an
80–100% view, and Full for both axes at 0–100%. Missing intermediate reasoning
levels are not bridged. Legend buttons hide models; points support mouse, touch
and keyboard focus. Hovering an exact overlap lists every coincident configuration.
