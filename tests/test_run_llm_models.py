import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import run_llm_models as runner
from benchmark_common import evaluate, load_inputs
from run_dl_models import predict


def response(body, status=200):
    result = requests.Response()
    result.status_code = status
    result._content = json.dumps({'provider': 'OpenAI', **body}).encode()
    return result


class RunnerTests(unittest.TestCase):
    def run_benchmark(self, responses, extra=()):
        session = Mock()
        session.post.side_effect = responses
        with tempfile.TemporaryDirectory() as temp:
            with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'fake-key'}), patch.object(runner.requests, 'Session') as factory:
                factory.return_value.__enter__.return_value = session
                code = runner.main(['--models', 'openai/gpt-6-luna', '--limit', str(len(responses)), '--output-dir', temp, *extra])
            effort = extra[extra.index('--reasoning') + 1] if '--reasoning' in extra else 'low'
            records = [json.loads(x) for x in Path(temp, f'llm_predictions_reasoning_{effort}.jsonl').read_text().splitlines()]
            summary = json.loads(Path(temp, f'llm_summary_reasoning_{effort}.json').read_text())[0]
            calls = [json.loads(x) for x in Path(temp, f'llm_api_calls_reasoning_{effort}.jsonl').read_text().splitlines()]
        return code, records, summary, calls, session

    def test_pin_reasoning_cost_and_latency(self):
        body = {'id': 'gen-1', 'choices': [{'message': {'content': 'positive'}}],
                'usage': {'cost': 0.001, 'prompt_tokens': 10, 'completion_tokens': 2, 'total_tokens': 12}}
        code, rows, summary, calls, session = self.run_benchmark([response(body)], ['--reasoning', 'high'])
        payload = session.post.call_args.kwargs['json']
        self.assertEqual(payload['provider'], {'order': ['openai'], 'only': ['openai'], 'allow_fallbacks': False, 'require_parameters': True})
        self.assertEqual(payload['reasoning'], {'effort': 'high'})
        self.assertNotIn('temperature', payload)
        self.assertEqual(session.headers.update.call_args.args[0]['Authorization'], 'Bearer fake-key')
        self.assertEqual(code, 0)
        self.assertTrue(rows[0]['provider_verified'])
        self.assertEqual(rows[0]['generation_id'], 'gen-1')
        self.assertEqual(rows[0]['total_tokens'], 12)
        self.assertEqual(rows[0]['api_cost_usd'], summary['api_cost_usd'])
        self.assertEqual(rows[0]['inference_seconds'], summary['total_inference_seconds'])
        self.assertEqual(len(calls), 1)

    def test_provider_mismatch_retains_cost_without_claiming_expected_provider(self):
        code, rows, summary, _, _ = self.run_benchmark([response({'provider': 'Azure', 'usage': {'cost': 0.01}})])
        self.assertEqual(code, 1)
        self.assertIsNone(rows[0]['predicted_label'])
        self.assertEqual(rows[0]['provider'], 'Azure')
        self.assertFalse(rows[0]['provider_verified'])
        self.assertEqual(summary['provider'], 'Azure')
        self.assertEqual(summary['api_cost_usd'], 0.01)

    def test_http_failure_has_one_call_and_preserves_reported_cost(self):
        code, rows, summary, calls, session = self.run_benchmark([response({'error': 'busy', 'usage': {'cost': 0.002}}, 503)])
        self.assertEqual(code, 1)
        session.post.assert_called_once()
        self.assertIsNone(rows[0]['predicted_label'])
        self.assertEqual(summary['api_cost_usd'], 0.002)
        self.assertEqual(len(calls), 1)

    def test_network_failure_is_recorded_and_next_sentence_runs(self):
        code, rows, summary, calls, _ = self.run_benchmark([requests.Timeout('fake-key'), response({'choices': [{'message': {'content': 'positive'}}], 'usage': {'cost': 0.001}})])
        self.assertEqual(code, 1)
        self.assertNotIn('fake-key', rows[0]['error'])
        self.assertEqual(rows[1]['predicted_label'], 'positive')
        self.assertIsNone(summary['api_cost_usd'])
        self.assertEqual(len(calls), 2)

    def test_unknown_cost_and_reported_zero_are_distinct(self):
        for usage, expected in (({}, None), ({'cost': 0}, 0), ({'cost': float('nan')}, None)):
            code, rows, summary, _, _ = self.run_benchmark([response({'choices': [{'message': {'content': 'positive'}}], 'usage': usage})])
            self.assertEqual(code, 0)
            self.assertEqual(rows[0]['api_cost_usd'], expected)
            self.assertEqual(summary['api_cost_usd'], expected)

    def test_ambiguous_output_fails_and_thoughts_are_excluded(self):
        for raw, expected in (('positive or negative', None), ('<think>negative</think>Positive.', 'positive')):
            code, rows, _, _, _ = self.run_benchmark([response({'choices': [{'message': {'content': raw}}]})])
            self.assertEqual(rows[0]['predicted_label'], expected)
            self.assertEqual(code, int(expected is None))

    def test_malformed_response_does_not_crash_benchmark(self):
        invalid = requests.Response()
        invalid.status_code = 200
        invalid._content = b'[]'
        code, rows, _, _, _ = self.run_benchmark([invalid])
        self.assertEqual(code, 1)
        self.assertIsNotNone(rows[0]['error'])

    def test_none_reasoning_is_sent_explicitly(self):
        _, _, _, _, session = self.run_benchmark([response({'choices': [{'message': {'content': 'positive'}}]})], ['--reasoning', 'none'])
        self.assertEqual(session.post.call_args.kwargs['json']['reasoning'], {'enabled': False})

    def test_invalid_predictions_count_as_errors(self):
        metrics = evaluate([{'id': '1', 'predicted_label': None}, {'id': '2', 'predicted_label': 'positive'}], {'1': 'negative', '2': 'positive'})
        self.assertEqual(metrics['accuracy'], 0.5)
        self.assertEqual(metrics['failed'], 1)
        self.assertEqual(metrics['confusion_matrix']['negative']['invalid'], 1)

    def test_answer_key_sentence_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            inp, ans = Path(temp, 'input.csv'), Path(temp, 'answer.csv')
            inp.write_text('id,sentence\n1,hello\n')
            ans.write_text('id,sentence,label\n1,other,neutral\n')
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                load_inputs(inp, ans)

    def test_dl_label_mapping_and_invalid_probabilities(self):
        clf = Mock(return_value=[{'label': 'LABEL_0', 'score': 0.1}, {'label': 'LABEL_1', 'score': 0.8}, {'label': 'LABEL_2', 'score': 0.1}])
        mapping = {'label_0': 'neutral', 'label_1': 'positive', 'label_2': 'negative'}
        self.assertEqual(predict(clf, 'text', 'text-classification', mapping)['positive'], 0.8)
        clf.return_value[0]['score'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'probability'):
            predict(clf, 'text', 'text-classification', mapping)


if __name__ == '__main__':
    unittest.main()
