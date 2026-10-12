import json
import math
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import run_decision_models as runner
from benchmark_common import ROOT


def answer(choice='positive', scores=None, native=.75):
    return {'answers': {'sentiment': {'type': 'choice', 'choice': choice,
            'probabilities': scores or {'negative':0, 'neutral':.1, 'positive':.9}, 'confidence': native}}}


class DecisionTests(unittest.TestCase):
    def test_winning_probability_and_native_confidence_are_distinct(self):
        label, probabilities, confidence, native = runner.parse_prediction(answer())
        self.assertEqual(label,'positive'); self.assertEqual(confidence,.9)
        self.assertEqual(native,.75); self.assertEqual(confidence,probabilities[label])

    def test_invalid_probabilities_or_nonwinning_choice_are_rejected(self):
        for data in [answer(choice='neutral'), answer(scores={'negative':0,'neutral':0,'positive':.8}),
                     answer(scores={'negative':0,'neutral':-.1,'positive':1.1}),
                     answer(scores={'negative':0,'positive':1}), answer(native=True),
                     answer(scores={'negative':0,'neutral':0,'positive':float('nan')})]:
            with self.assertRaises((ValueError,KeyError)):
                runner.parse_prediction(data)

    def test_unknown_cost_does_not_become_zero_or_partial_total(self):
        rows=[{'id':'1','predicted_label':'positive','confidence':.9,'native_confidence':.7,'api_cost_usd':None,'inference_seconds':1},
              {'id':'2','predicted_label':'positive','confidence':.9,'native_confidence':.7,'api_cost_usd':0,'inference_seconds':2}]
        summary=runner.summarize('example',rows,{'1':'positive','2':'negative'},'endpoint')
        self.assertIsNone(summary['api_cost_usd']); self.assertEqual(summary['known_api_cost_usd'],0)
        self.assertEqual(summary['missing_cost_samples'],1); self.assertEqual(summary['accuracy'],.5)
        self.assertEqual(summary['mean_latency_ms'],1500)

    def test_smoke_payload_contains_no_answer_key_and_records_response_details(self):
        session=Mock(); responses=[]
        for model in runner.MODELS:
            response=requests.Response(); response.status_code=200
            response._content=json.dumps(answer() | {'model':model,'provider':'Returned provider','id':'id-'+model,'usage':{'cost':0,'input_tokens':10,'output_tokens':2}}).encode()
            responses.append(response)
        session.post.side_effect=responses
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'data').mkdir()
            for name in ('test_30.csv','answer_key_30.csv'): shutil.copyfile(ROOT/'data'/name,root/'data'/name)
            with patch.object(runner,'ROOT',root),patch.dict(os.environ,{'LLMFOUNDRY_API_KEY':'fake-key'}),patch.object(runner.requests,'Session') as factory:
                factory.return_value.__enter__.return_value=session
                self.assertEqual(runner.main(['--smoke']),0)
            self.assertEqual(session.post.call_count,3)
            for call in session.post.call_args_list:
                payload=call.kwargs['json']; self.assertEqual(set(payload),{'model','state','questions'})
                self.assertEqual(set(payload['state']),{'sentence'})
                self.assertNotIn('label',payload['state']); self.assertEqual(payload['questions']['sentiment'],runner.QUESTION)
            out=next((root/'results/decision_models').iterdir()); self.assertIn('smoke',out.name)
            records=[json.loads(line) for line in (out/'llm_predictions.jsonl').read_text().splitlines()]
            calls=[json.loads(line) for line in (out/'llm_api_calls.jsonl').read_text().splitlines()]
            self.assertEqual(len(records),3)
            for r,c in zip(records,calls):
                self.assertEqual(r['confidence'],.9); self.assertEqual(r['native_confidence'],.75)
                self.assertEqual(r['api_cost_usd'],0); self.assertEqual(r['total_tokens'],12)
                self.assertEqual(r['provider'],'Returned provider'); self.assertEqual(r['http_status'],200)
                self.assertEqual(r['response_model'],r['model']); self.assertTrue(math.isfinite(r['inference_seconds']))
                self.assertIn('raw_response',c)
