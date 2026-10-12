import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import temp_failed_rerun as repair
from benchmark_common import load_inputs, ROOT


class RepairTests(unittest.TestCase):
    def fixture(self, directory):
        rows, truth = load_inputs(ROOT / 'data/test_30.csv', ROOT / 'data/answer_key_30.csv')
        meta = dict(model='google/gemini-3.8-flash', endpoint='https://llmfoundry.straivedemo.com/openrouter/v1/chat/completions', requested_provider='google-vertex/global', expected_provider='Google', temperature=None, reasoning_effort='low', provider_routing={'order':['google-vertex/global'],'only':['google-vertex/global'],'allow_fallbacks':False,'require_parameters':True})
        records = [meta | row | dict(predicted_label=truth[row['id']], confidence=.9, api_cost_usd=.01, inference_seconds=1, provider='Google', provider_verified=True, error=None) for row in rows]
        records[0].update(predicted_label=None, confidence=None, api_cost_usd=None, inference_seconds=120, error='timeout')
        predictions = directory / 'llm_predictions_reasoning_low.jsonl'
        calls = directory / 'llm_api_calls_reasoning_low.jsonl'
        summary = directory / 'llm_summary_reasoning_low.json'
        text = ''.join(json.dumps(r) + '\n' for r in records)
        predictions.write_text(text); calls.write_text(text)
        summary.write_text(json.dumps([meta | dict(model_run_wall_seconds=150)]))
        return predictions, calls, summary, records, truth

    def test_only_failed_rows_are_replaced_and_successes_are_byte_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            path, calls, summary, records, truth = self.fixture(Path(directory))
            original_lines = path.read_text().splitlines()
            new = records[0] | dict(predicted_label=truth[records[0]['id']], confidence=.95, api_cost_usd=.02, inference_seconds=2, error=None)
            with patch.dict(os.environ, {'LLMFOUNDRY_API_KEY':'fake-key'}), patch.object(repair,'predict_one',return_value=new.copy()) as request:
                self.assertEqual(repair.main(['--results-dir',directory]),0)
                request.assert_called_once()
                self.assertEqual(request.call_args.args[2]['id'],records[0]['id'])
            lines = path.read_text().splitlines()
            self.assertEqual(lines[1:],original_lines[1:])
            self.assertEqual(json.loads(lines[0])['replaced_attempt'],records[0])
            self.assertEqual(len(calls.read_text().splitlines()),31)
            score = json.loads(summary.read_text())[0]
            self.assertEqual(score['failed'],0); self.assertEqual(score['accuracy'],1)
            self.assertAlmostEqual(score['api_cost_usd'],.31)
            self.assertEqual(score['total_inference_seconds'],31)
            self.assertEqual(score['model_run_wall_seconds'],152)
            self.assertEqual(score['unpriced_api_attempts'],1)
            self.assertIsNone(score['all_attempts_api_cost_usd'])
            self.assertAlmostEqual(score['mean_confidence'],(.95+29*.9)/30)
            with patch.object(repair,'predict_one') as request:
                self.assertEqual(repair.main(['--results-dir',directory]),0)
                request.assert_not_called()

    def test_unsuccessful_retry_is_logged_without_overwriting_any_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            path,calls,summary,records,_ = self.fixture(Path(directory))
            before = path.read_bytes()
            with patch.dict(os.environ,{'LLMFOUNDRY_API_KEY':'fake-key'}),patch.object(repair,'predict_one',return_value=records[0].copy()):
                self.assertEqual(repair.main(['--results-dir',directory]),1)
            self.assertEqual(path.read_bytes(),before)
            self.assertEqual(len(calls.read_text().splitlines()),31)
            self.assertEqual(json.loads(summary.read_text())[0]['failed'],1)

    def test_dry_run_makes_no_calls_or_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            path,calls,summary,_,_ = self.fixture(Path(directory))
            before = {p:p.read_bytes() for p in (path,calls,summary)}
            with patch.object(repair,'predict_one') as request:
                self.assertEqual(repair.main(['--results-dir',directory,'--dry-run']),0)
                request.assert_not_called()
            self.assertEqual(before,{p:p.read_bytes() for p in before})

    def test_partial_test_set_is_rejected_before_calling_the_api(self):
        with tempfile.TemporaryDirectory() as directory:
            path,_,_,_,_ = self.fixture(Path(directory))
            path.write_text(path.read_text().splitlines()[0]+'\n')
            with patch.object(repair,'predict_one') as request:
                with self.assertRaisesRegex(ValueError,'complete matching test-set'):
                    repair.main(['--results-dir',directory])
                request.assert_not_called()

    def test_valid_but_wrong_answer_is_kept_without_repeating_until_correct(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _, summary, records, truth = self.fixture(Path(directory))
            wrong = next(label for label in repair.LABELS if label != truth[records[0]['id']])
            new = records[0] | dict(predicted_label=wrong, confidence=.95, api_cost_usd=.02, inference_seconds=2, error=None)
            with patch.dict(os.environ, {'LLMFOUNDRY_API_KEY':'fake-key'}), patch.object(repair,'predict_one',return_value=new) as request:
                self.assertEqual(repair.main(['--results-dir',directory]),0)
                self.assertEqual(repair.main(['--results-dir',directory]),0)
                request.assert_called_once()
            self.assertEqual(json.loads(path.read_text().splitlines()[0])['predicted_label'],wrong)
            self.assertAlmostEqual(json.loads(summary.read_text())[0]['accuracy'],29/30)
