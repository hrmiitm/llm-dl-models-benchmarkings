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


def response(body, status=200, headers=None):
    result = requests.Response()
    result.status_code = status
    result._content = json.dumps(body).encode()
    result.headers.update(headers or {})
    return result


class RunnerTests(unittest.TestCase):
    spec = {'model': 'openai/gpt-6-luna', 'reasoning': True, 'temperature': None}

    def test_openrouter_payload_and_reported_usage(self):
        session = Mock()
        session.post.return_value = response({'id': 'gen-1', 'provider': 'OpenAI', 'model': self.spec['model'],
            'choices': [{'message': {'content': 'positive'}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 20, 'completion_tokens': 4, 'total_tokens': 24, 'cost': 0.001,
                      'completion_tokens_details': {'reasoning_tokens': 2}}})
        attempts = []
        result = runner.call_model(session, self.spec, 'https://openrouter.ai/api/v1/chat/completions', 'Sales increased.', 'high', attempts=attempts)
        payload = session.post.call_args.kwargs['json']
        self.assertNotIn('temperature', payload)
        self.assertNotIn('reasoning_effort', payload)
        self.assertEqual(payload['reasoning'], {'effort': 'high'})
        self.assertNotIn('usage', payload)
        self.assertEqual(result['predicted_label'], 'positive')
        self.assertEqual(result['reasoning_tokens'], 2)
        self.assertEqual(result['generation_id'], 'gen-1')
        self.assertEqual(runner.cost_totals(attempts)['api_cost_usd'], 0.001)

    def test_retry_cost_includes_failed_attempt(self):
        session = Mock()
        session.post.side_effect = [response({'error': 'overloaded', 'usage': {'cost': 0.002}}, 503, {'Retry-After': '4'}),
            response({'choices': [{'message': {'content': 'negative'}}], 'usage': {'cost': 0.003}})]
        attempts = []
        with patch.object(runner.time, 'sleep') as sleep:
            result = runner.call_model(session, self.spec, 'https://openrouter.ai/api/v1/chat/completions', 'text', attempts=attempts)
        sleep.assert_called_once_with(4)
        self.assertEqual(result['predicted_label'], 'negative')
        self.assertEqual(runner.cost_totals(attempts)['api_cost_usd'], 0.005)
        self.assertEqual(len(attempts), 2)

    def test_unknown_and_zero_cost_are_distinct(self):
        self.assertEqual(runner.reported_cost({'usage': {'cost': 0}}, {}), (0, 'usage.cost'))
        self.assertEqual(runner.reported_cost({'usage': {'cost': float('nan')}}, {}), (None, None))
        totals = runner.cost_totals([{'api_cost_usd': 0.1}, {'api_cost_usd': None}])
        self.assertIsNone(totals['api_cost_usd'])
        self.assertEqual(totals['known_api_cost_usd'], 0.1)
        self.assertEqual(totals['api_calls_without_cost'], 1)
        self.assertIsNone(runner.cost_totals([])['api_cost_usd'])

    def test_network_failure_is_logged_as_unknown_cost(self):
        session = Mock()
        session.post.side_effect = requests.Timeout('sensitive details')
        attempts = []
        with self.assertRaisesRegex(RuntimeError, 'Timeout'):
            runner.call_model(session, self.spec, 'https://openrouter.ai/api/v1/chat/completions', 'text', retries=0, attempts=attempts)
        self.assertEqual(len(attempts), 1)
        self.assertIsNone(attempts[0]['http_status'])
        self.assertNotIn('sensitive', attempts[0]['error'])

    def test_permanent_failure_logs_cost_without_retry(self):
        session = Mock()
        session.post.return_value = response({'error': {'message': 'Key limit exceeded'}}, 403)
        attempts = []
        with self.assertRaisesRegex(runner.ModelAccessError, 'Key limit exceeded'):
            runner.call_model(session, self.spec, 'https://openrouter.ai/api/v1/chat/completions', 'text', attempts=attempts)
        session.post.assert_called_once()
        self.assertEqual(len(attempts), 1)

    def test_ambiguous_or_missing_answers_are_not_predictions(self):
        session = Mock()
        session.post.return_value = response({'usage': {'cost': 0.1}})
        with self.assertRaisesRegex(RuntimeError, 'no choices'):
            runner.call_model(session, self.spec, 'https://openrouter.ai/api/v1/chat/completions', 'text')
        self.assertIsNone(runner.label_from('positive or negative'))
        self.assertEqual(runner.label_from('<think>negative</think>Positive.'), 'positive')

    def test_connection_does_not_modify_direct_key(self):
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test-key'}, clear=True):
            provider, endpoint, auth = runner.connection('auto')
        self.assertEqual(provider, 'openrouter')
        self.assertEqual(auth, 'test-key')
        self.assertEqual(endpoint, 'https://openrouter.ai/api/v1/chat/completions')

    def test_model_rejection_skips_rows_and_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as temp:
            session = Mock()
            session.post.return_value = response({'error': 'no access'}, 403)
            with patch.object(runner, 'connection', return_value=('openrouter', 'https://openrouter.ai/api/v1/chat/completions', 'fake-token')), patch.object(runner.requests, 'Session') as factory:
                factory.return_value.__enter__.return_value = session
                code = runner.main(['--models', 'openai/gpt-6-luna', '--limit', '2', '--output-dir', temp])
            records = [json.loads(x) for x in Path(temp, 'llm_predictions.jsonl').read_text().splitlines()]
            summary = json.loads(Path(temp, 'llm_summary.json').read_text())[0]
            self.assertEqual(code, 1)
            session.post.assert_called_once()
            self.assertTrue(records[1]['skipped'])
            self.assertEqual(records[1]['api_calls'], 0)
            self.assertEqual(summary['failed'], 2)
            self.assertIsNone(summary['api_cost_usd'])

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
