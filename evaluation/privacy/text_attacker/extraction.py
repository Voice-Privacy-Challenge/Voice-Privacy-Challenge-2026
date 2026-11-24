"""Speaker embeddings from the ``text`` file of a data directory.

TextExtraction stands in for the SpeakerExtraction of the ASV evaluation, so
that the scoring of ``evaluation.privacy.asv.asv.ASV`` runs on transcripts.
"""
from pathlib import Path

import torch

from ..asv.speaker_embeddings import SpeakerEmbeddings
from . import textnorm
from utils import read_kaldi_format, setup_logger

logger = setup_logger(__name__)


class TextExtraction:

    def __init__(self, encoder, head, max_length, hyp_norm, asr_text_dir, results_dir, batch_size=32,
                 force_compute=False):
        """
        Args:
            encoder, head: PooledEncoder and Head
            max_length: number of tokens a transcript is truncated to
            hyp_norm: normalization of the transcripts under ``asr_text_dir`` (see textnorm.py)
            asr_text_dir: results directory of the ASR evaluation
            results_dir: where the embeddings of each data directory are stored
        """
        self.encoder = encoder
        self.head = head
        self.max_length = max_length
        self.hyp_norm = hyp_norm
        self.asr_text_dir = Path(asr_text_dir)
        self.results_dir = results_dir
        self.batch_size = batch_size
        self.force_compute = force_compute
        self.extracted = {}  # data directory -> embeddings of this run

    def read_text(self, dataset_path):
        """Transcripts of a data directory; ASR transcripts are normalized, reference transcripts are not."""
        text = read_kaldi_format(dataset_path / 'text', values_as_string=True)
        if self.asr_text_dir in dataset_path.parents:
            text = {utt: textnorm.normalize(transcript, self.hyp_norm) for utt, transcript in text.items()}
        return text

    @torch.inference_mode()
    def extract_speakers(self, dataset_path, emb_level='utt'):
        if dataset_path not in self.extracted:
            results_dir = self.results_dir / dataset_path.name / 'utt-level'
            embeddings = SpeakerEmbeddings(vec_type='text', emb_level='utt')
            if (results_dir / 'speaker_vectors.pt').exists() and not self.force_compute:
                logger.info('No speaker extraction necessary; load existing embeddings instead...')
                embeddings.load_vectors(results_dir)
            else:
                text = self.read_text(dataset_path)
                logger.info(f'Extract embeddings of {len(text)} utterances')
                utts = list(text)
                pooled = self.encoder.pool(list(text.values()), max_length=self.max_length, batch_size=self.batch_size)
                vectors = self.head.embed(torch.from_numpy(pooled).to(self.encoder.device)).cpu()
                speakers = utts  # the directories of the ASR evaluation have no utt2spk
                if emb_level == 'spk':
                    utt2spk = read_kaldi_format(dataset_path / 'utt2spk')
                    speakers = [utt2spk[utt] for utt in utts]
                embeddings.set_vectors(identifiers=utts, vectors=vectors, speakers=speakers, genders=[''] * len(utts))
                embeddings.save_vectors(results_dir)
            self.extracted[dataset_path] = embeddings.convert_to_spk_level() if emb_level == 'spk' else embeddings
        return self.extracted[dataset_path]
