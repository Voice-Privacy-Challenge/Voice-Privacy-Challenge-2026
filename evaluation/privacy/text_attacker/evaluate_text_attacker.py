"""Evaluation of the text attacker.

Enrollment always uses reference transcripts. The trial text is one of:
    ref        reference transcripts (the ``text`` file of the data directory)
    asr_orig   ASR transcripts of the original trial audio
    asr_anon   ASR transcripts of the anonymized trial audio
The ASR transcripts are the ``text`` files that the ASR evaluation of run_evaluation.py writes.
"""
import pandas as pd

from ..asv.asv import ASV
from .encoder import PooledEncoder
from .extraction import TextExtraction
from .head import load_head


class TextASV(ASV):
    """Trial scoring and EER of the ASV evaluation (cosine), on the embeddings of a TextExtraction."""

    def __init__(self, extractor, score_save_dir):
        # ASV.__init__ builds an audio SpeakerExtraction; only the cosine scoring is reused
        self.extractor = extractor
        self.score_save_dir = score_save_dir
        self.distance = 'cosine'


def load_encoder(params, device):
    return PooledEncoder(backbone=params['backbone'], revision=params['backbone_revision'], device=device)


def load_attacker(params, dataset_name, encoder, force_compute=False):
    """TextASV for a dataset; its head is chosen by the dataset prefix (libri, en, de, ...)."""
    eval_params = params['evaluation']
    head, meta = load_head(eval_params['heads'][dataset_name.split('_')[0]]['head_dir'], device=encoder.device)
    score_save_dir = eval_params['results_dir'] / 'cosine_out'
    extractor = TextExtraction(encoder=encoder, head=head, max_length=meta['max_length'],
                               hyp_norm=meta['hyp_norm'] if eval_params['hyp_norm'] == 'head' else eval_params['hyp_norm'],
                               asr_text_dir=eval_params['asr_text_dir'], results_dir=score_save_dir / 'emb_text',
                               batch_size=eval_params['eval_batchsize'], force_compute=force_compute)
    score_save_dir.mkdir(parents=True, exist_ok=True)
    return TextASV(extractor=extractor, score_save_dir=score_save_dir)


def trial_text_dir(trial_source, eval_data_dir, trial, asr_text_dir, anon_data_suffix):
    """Directory whose ``text`` file holds the trial transcripts of a condition."""
    if trial_source == 'ref':
        return eval_data_dir / trial
    dataset_name = '_'.join(trial.split('_')[:2])
    return asr_text_dir / f'{dataset_name}_asr{anon_data_suffix if trial_source == "asr_anon" else ""}'


def evaluate_text_attacker(eval_datasets, eval_data_dir, params, device, anon_data_suffix, force_compute=False):
    save_dir = params['evaluation']['results_dir']
    encoder = load_encoder(params, device)
    results = []

    for enroll, trial in eval_datasets:
        asv = load_attacker(params, trial, encoder, force_compute=force_compute)
        for trial_source in params['evaluation']['trial_text']:
            test_dir = trial_text_dir(trial_source, eval_data_dir, trial, params['evaluation']['asr_text_dir'],
                                      anon_data_suffix)
            # as in the ASV evaluation, the trial list is always the one of the original trial directory
            EER = asv.eer_compute(enrol_dir=eval_data_dir / enroll, test_dir=test_dir,
                                  trial_runs_file=eval_data_dir / trial / 'trials')
            EER = round(EER * 100, 3)

            print(f'{enroll}-{test_dir.name}: {trial_source}-EER={EER}')
            trials_info = trial.split('_')
            gender = trials_info[3]
            if 'common' in trial:
                gender += '_common'

            results.append({'dataset': trials_info[0], 'split': trials_info[1], 'gender': gender,
                            'trial_text': trial_source, 'EER': EER})

    results_df = pd.DataFrame(results)
    print(results_df)
    results_df.to_csv(save_dir / f'results{anon_data_suffix}.csv')
    return results_df
