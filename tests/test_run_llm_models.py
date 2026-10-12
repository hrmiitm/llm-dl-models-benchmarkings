import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import requests

path = Path(__file__).resolve().parents[1] / 'src/run_llm_models.py'
spec = importlib.util.spec_from_file_location('run_llm_models', path)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def response(body, status=200, headers=None):
    result = requests.Response()
    result.status_code = status
    result._content = json.dumps(body).encode()
    result.headers.update(headers or {})
    return result


class RunnerTests(unittest.TestCase):
    def test_groq_request_fits_quota_and_sets_reasoning(self):
        session = Mock()
        session.post.return_value = response({'choices': [{'message': {'content': 'positive'}}]})
        raw, label, _, _ = runner.call_model(
            session, 'qwen/qwen3.8-27b', 'https://proxy/groq/openai/v1/chat/completions', 'Sales increased.', 'high')
        payload = session.post.call_args.kwargs['json']
        self.assertEqual(payload['max_completion_tokens'], 512)
        self.assertNotIn('max_tokens', payload)
        self.assertEqual(payload['temperature'], 0)
        self.assertEqual(payload['reasoning_effort'], 'high')
        self.assertEqual(label, 'positive')

    def test_native_gemini_uses_native_thinking_and_excludes_thoughts(self):
        session = Mock()
        session.post.return_value = response({
            'candidates': [{'content': {'parts': [{'text': 'negative', 'thought': True}, {'text': 'neutral'}]}}],
            'usageMetadata': {'promptTokenCount': 20, 'candidatesTokenCount': 2, 'totalTokenCount': 30},
        })
        raw, label, _, tokens = runner.call_model(
            session, 'gemini-3.8-flash', 'https://proxy/gemini/v1beta/models/gemini-3.8-flash:generateContent', 'Board meeting.', 'medium')
        config = session.post.call_args.kwargs['json']['generationConfig']
        self.assertEqual(config['thinkingConfig']['thinkingLevel'], 'MEDIUM')
        self.assertEqual(raw, 'neutral')
        self.assertEqual(label, 'neutral')
        self.assertEqual(tokens, (20, 2, 30))

    def test_permanent_failure_is_not_retried_and_retains_reason(self):
        session = Mock()
        session.post.return_value = response({'error': {'message': 'Key limit exceeded'}}, 403)
        with self.assertRaisesRegex(runner.ModelAccessError, 'Key limit exceeded'):
            runner.call_model(session, 'example', 'https://proxy/openrouter/v1/chat/completions', 'text')
        session.post.assert_called_once()

    def test_transient_failure_honors_retry_after(self):
        session = Mock()
        session.post.side_effect = [response({'error': 'overloaded'}, 503, {'Retry-After': '4'}),
                                    response({'choices': [{'message': {'content': 'negative'}}]})]
        with patch.object(runner.time, 'sleep') as sleep:
            self.assertEqual(runner.call_model(session, 'example', 'https://proxy/gemini/v1beta/openai/chat/completions', 'text')[1], 'negative')
        sleep.assert_called_once_with(4)
        self.assertEqual(session.post.call_count, 2)

    def test_missing_candidates_and_ambiguous_labels_are_not_predictions(self):
        session = Mock()
        session.post.return_value = response({'promptFeedback': {'blockReason': 'OTHER'}})
        with self.assertRaisesRegex(RuntimeError, 'no candidates'):
            runner.call_model(session, 'example', 'https://proxy/gemini/example:generateContent', 'text')
        self.assertIsNone(runner.label_from('positive or negative'))
        self.assertEqual(runner.label_from('<think>negative</think>Positive.'), 'positive')


if __name__ == '__main__':
    unittest.main()
