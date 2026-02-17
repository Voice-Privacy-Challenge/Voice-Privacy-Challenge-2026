from pathlib import Path
import numpy as np
import torch
import logging
import torch.nn.functional as F
from speechbrain.pretrained import EncoderClassifier


class SpeechBrainVectors:

    def __init__(self, vec_type, device, model_path: Path = None):

        self.device = device
        self.vec_type = vec_type
        if model_path is not None: # and model_path.exists():
            model_path = Path(model_path).absolute()
            if model_path.is_file():
                savedir = model_path.parent
            else:
                savedir = model_path
        logging.info(f"Loading {savedir}")

    
        if vec_type == 'ecapa':
            if model_path.exists():
                source = str(model_path)
            else:
                source = 'speechbrain/spkrec-ecapa-voxceleb'
            self.extractor = EncoderClassifier.from_hparams(
                    source=source,
                    savedir=str(savedir),
                    run_opts={'device': self.device}
            )
  
        else:
            if model_path is None:
                model_path = Path('')
            print("Model Path not found")

       
    def extract_vector(self, audio, sr, wav_path=None):

        if self.vec_type == 'sslecapa':
            filtered_logits=None
            audio = torch.tensor(np.trim_zeros(audio.cpu().numpy()))
            if len(audio.shape) == 1:
                wavs = audio.unsqueeze(0)
                if len(wavs.shape) == 1:
                    wavs = wavs.unsqueeze(0)
                wav_lens = torch.ones(wavs.shape[0])
                wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
                fbank_features = self.fbank_computer(wavs)
                fbank_features = self.normalizer(fbank_features, wav_lens)
            ssl_features = self.wavlm_.extract_features([wav_path])
            vec = self.extractor(fbank_features, ssl_features)

            if vec.dim() == 1:
                return F.normalize(vec, dim=0)
            else:
                return F.normalize(vec, dim=1).squeeze()

        elif self.vec_type == 'ecapa':
            audio = torch.tensor(np.trim_zeros(audio.cpu().numpy()))
            if len(audio.shape) == 1:
                audio = audio.unsqueeze(0)
            return self.extractor.encode_batch(wavs=audio).squeeze()

