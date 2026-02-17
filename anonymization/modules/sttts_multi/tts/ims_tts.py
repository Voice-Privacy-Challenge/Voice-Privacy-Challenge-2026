import torch
import resampy
import numpy as np

from .IMSToucan.InferenceInterfaces.AnonToucanTTS import AnonToucanTTS
from utils import setup_logger

logger = setup_logger(__name__)

class ImsTTS:

    def __init__(self, hifigan_path, fastspeech_path, device, embedding_path=None, output_sr=16000, lang='en'):
        self.device = device
        self.output_sr = output_sr

        self.model = AnonToucanTTS(device=self.device, vocoder_model_path=hifigan_path, tts_model_path=fastspeech_path,
                                   emb_model_path=embedding_path, language=lang)

    def read_text(self, text, speaker_embedding, text_is_phones=True, language='en', duration=None, pitch=None,
                  energy=None, start_silence=None, end_silence=None):
        if pitch is not None:
            pitch = pitch.transpose(0, 1)
        if energy is not None:
            energy = energy.transpose(0, 1)

        self.model.set_language(language)
        speaker_embedding = speaker_embedding.to(self.device)
        self.model.default_utterance_embedding = speaker_embedding
        wav, sr = self.model(text=text, input_is_phones=text_is_phones, durations=duration, pitch=pitch, energy=energy)

        # TODO: this is not an ideal solution, but must work for now
        i = 0
        while wav.shape[0] < (sr // 2):  # 0.5 s
            # sometimes, the speaker embedding is so off that it leads to a practically empty audio
            # then, we need to sample a new embedding
            if i > 0 and i % 10 == 0:
                mask = torch.zeros(speaker_embedding.shape[0]).float().random_(-40, 40).to(self.device)
            else:
                mask = torch.zeros(speaker_embedding.shape[0]).float().random_(-2, 2).to(self.device)
            speaker_embedding = speaker_embedding * mask
            self.model.default_utterance_embedding = speaker_embedding.to(self.device)
            wav, sr = self.model(text=text, input_is_phones=text_is_phones, durations=duration, pitch=pitch, energy=energy)
            i += 1
            if i > 30:
                break
        if i > 0:
            logger.info(f'Synthesized utt in {i} takes')

        # start and end silence are computed for 16000, so we have to adapt this to different output sr
        factor = sr / 16000
        if start_silence is not None:
            start_sil = np.zeros([int(start_silence * factor)])
            wav = np.concatenate((start_sil, wav), axis=0)
        if end_silence is not None:
            end_sil = np.zeros([int(end_silence * factor)])
            wav = np.concatenate((wav, end_sil), axis=0)

        if self.output_sr != sr:
            wav = resampy.resample(wav, sr, self.output_sr)

        return wav
