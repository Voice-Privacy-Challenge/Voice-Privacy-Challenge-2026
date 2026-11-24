import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from .. import encoder, textnorm
from ..evaluate_text_attacker import evaluate_text_attacker
from ..head import Head, load_head
from .fakes import FakeModel, FakeTokenizer, save_head
from utils import parse_yaml

CONFIG = """
data_dir: {root}/data
exp_dir: {root}/exp
anon_data_suffix: '{suffix}'
datasets:
  - name: toy_dev
    data: toy_dev
    enrolls: [_enrolls]
    trials: [_trials_mixed]
privacy:
  text_attacker:
    dataset_name: [toy_dev]
    backbone: fake
    backbone_revision: none
    evaluation:
      results_dir: !ref <exp_dir>/text_attacker/results
      asr_text_dir: !ref <exp_dir>/asr
      trial_text: {trial_text}
      hyp_norm: head
      eval_batchsize: 2
      heads:
        toy:
          head_dir: !ref <exp_dir>/heads/toy
"""


class TextNormTest(unittest.TestCase):
    def test_modes(self):
        self.assertEqual(textnorm.normalize("Mr. Smith’s 2 dogs, well-known!", 'upper_letters'), "MR SMITH'S DOGS WELL KNOWN")
        self.assertEqual(textnorm.normalize("L’élève, well-known 3x²", 'lower_letters'), "l'élève well-known x")
        self.assertEqual(textnorm.normalize("As it is, 1.", 'none'), "As it is, 1.")


class EvaluationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        files = {'data/toy_dev_enrolls/text': "s1-a THE SHIP SAILED NORTH\ns1-b THE SHIP SANK\ns2-a BREAD AND BUTTER\ns2-b\n",
                 'data/toy_dev_enrolls/utt2spk': "s1-a s1\ns1-b s1\ns2-a s2\ns2-b s2\n",
                 'data/toy_dev_trials_mixed/text': "t1 THE SHIP SAILED\nt2 BREAD AND CAKE\n",
                 'data/toy_dev_trials_mixed/trials': "s1 t1 target\ns2 t1 nontarget\ns1 t2 nontarget\ns2 t2 target\n",
                 'exp/asr/toy_dev_asr/text': "t1 the ship sailed, 2 times\nt2 bread and cake\n",
                 'exp/asr/toy_dev_asr_anon/text': "t1 the sheep sailed\nt2\n"}
        for relative, content in files.items():
            (self.root / relative).parent.mkdir(parents=True, exist_ok=True)
            (self.root / relative).write_text(content, encoding='utf-8')
        torch.manual_seed(0)
        save_head(self.root / 'exp/heads/toy', Head(hidden=16, bottleneck=8, embedding=4),
                  {'name': 'toy', 'max_length': 96, 'hyp_norm': 'upper_letters'})
        with mock.patch.object(encoder.AutoTokenizer, 'from_pretrained', return_value=FakeTokenizer()), \
                mock.patch.object(encoder.AutoModel, 'from_pretrained', return_value=FakeModel()):
            self.encoder = encoder.PooledEncoder(backbone='fake')

    def tearDown(self):
        self.tmp.cleanup()

    def params(self, suffix='', trial_text='[ref, asr_orig]'):
        (self.root / 'config.yaml').write_text(CONFIG.format(root=self.root, suffix=suffix, trial_text=trial_text))
        return parse_yaml(self.root / 'config.yaml')

    def run_attacker(self, **config):
        params = self.params(**config)
        # the package exports the function under the name of its module
        with mock.patch.object(sys.modules[evaluate_text_attacker.__module__], 'load_encoder', return_value=self.encoder):
            return evaluate_text_attacker(eval_datasets=[('toy_dev_enrolls', 'toy_dev_trials_mixed')],
                                          eval_data_dir=params['data_dir'], params=params['privacy']['text_attacker'],
                                          device='cpu', anon_data_suffix=params['anon_data_suffix'])

    def test_reference_and_original_asr(self):
        results = self.run_attacker().set_index('trial_text')
        self.assertEqual(sorted(results.index), ['asr_orig', 'ref'])
        out = self.root / 'exp/text_attacker/results'
        scores = (out / 'toy_dev_enrolls-toy_dev_trials_mixed/scores').read_text().splitlines()
        self.assertEqual([line.split()[:2] for line in scores], [['s1', 't1'], ['s2', 't1'], ['s1', 't2'], ['s2', 't2']])
        eer = float((out / 'toy_dev_enrolls-toy_dev_trials_mixed/EER').read_text())
        self.assertAlmostEqual(100 * eer, results.loc['ref', 'EER'], places=3)
        self.assertTrue((out / 'toy_dev_enrolls-toy_dev_asr/scores').is_file())
        self.assertTrue((out / 'results.csv').is_file())

    def test_anonymized_condition(self):
        results = self.run_attacker(suffix='_anon', trial_text='[ref, asr_orig, asr_anon]').set_index('trial_text')
        self.assertEqual(sorted(results.index), ['asr_anon', 'asr_orig', 'ref'])
        self.assertTrue((self.root / 'exp/text_attacker/results/toy_dev_enrolls-toy_dev_asr_anon/scores').is_file())

    def test_asr_transcripts_are_normalized_to_the_reference_convention(self):
        (self.root / 'exp/asr/toy_dev_asr/text').write_text("t1 the ship sailed\nt2 Bread, and cake!\n")
        self.run_attacker()
        out = self.root / 'exp/text_attacker/results'
        self.assertEqual((out / 'toy_dev_enrolls-toy_dev_asr/scores').read_text(),
                         (out / 'toy_dev_enrolls-toy_dev_trials_mixed/scores').read_text())

    def test_head_round_trip(self):
        head, meta = load_head(self.root / 'exp/heads/toy')
        self.assertEqual((meta['name'], meta['hidden'], meta['embedding']), ('toy', 16, 4))
        emb = head.embed(torch.ones(2, 16))
        np.testing.assert_allclose(emb.norm(dim=-1).detach().numpy(), 1, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
