from pathlib import Path
import re
import numpy as np
import torch
from evaluation.privacy.asv.ecapa_model_wavlm_24layer import ECAPA_TDNN_test
import logging
from speechbrain.processing.features import InputNormalization
from speechbrain.lobes.features import Fbank
import torch.nn.functional as F
import soundfile as sf
import pandas as pd
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import os
from evaluation.privacy.asv.WavLM import WavLM, WavLMConfig
from torch.nn.utils.rnn import pad_sequence
from typing import List, Union, Tuple, Dict
from collections import OrderedDict
from hyperpyyaml import load_hyperpyyaml
from speechbrain.pretrained import EncoderClassifier

class WavLMFeatureExtractor:
    """
    WavLM feature extractor that searches for audio files across
    Kaldi-style data directories and performs batched inference.
    """

    def __init__(self, model_path: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading WavLM model to {self.device}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.cfg = WavLMConfig(checkpoint['cfg'])
        self.model = WavLM(self.cfg)
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(self.device)
        self.model.eval()
        print("Extractor is ready.")

   
    def _load_audio(self, file_path: str):
        """Loads and pre-processes a single audio file."""
        audio_input, sr = sf.read(file_path)
        audio_input = torch.from_numpy(audio_input).float()

        # Ensure 1D audio (T,)
        if audio_input.dim() > 1:
            # Select first channel if stereo
            audio_input = audio_input[0, :]
        
        # Original code had layer norm here
        if self.cfg.normalize:
            audio_input = F.layer_norm(audio_input, audio_input.shape)
        return audio_input, sr

    def extract_features(
        self, 
        audio_paths: Union[str, List[str], Tuple[str, ...]]
    ) -> Dict[str, Union[List[torch.Tensor], List[str]]]:
        if isinstance(audio_paths, str):
            audio_paths = [audio_paths]

        audio_inputs = []
        for path in audio_paths:
            audio, _ = self._load_audio(path)
            audio_inputs.append(audio)
        padded_batch = pad_sequence(audio_inputs, batch_first=True)
        padded_batch = padded_batch.to(self.device)

        # 3. Run inference ONCE on the entire batch
        with torch.no_grad():
            final_features, layer_results_list = self.model.extract_features(
                padded_batch,
                output_layer=self.model.cfg.encoder_layers,
                ret_layer_results=True
            )
        layer_reps = [x for x, _ in final_features[1][1:]] 
        layer_reps = torch.stack(layer_reps, dim=0)
        layer_reps = layer_reps.permute(2, 0, 1, 3)
        del final_features, layer_results_list, padded_batch,audio_inputs
        return layer_reps



class SpeechBrainVectors:

    def __init__(self, vec_type, device, model_path: Path = None):

        self.device = device
        self.vec_type = vec_type
        if model_path is not None and model_path.exists():
            model_path = Path(model_path).absolute()
            if model_path.is_file():
                savedir = model_path.parent
            else:
                savedir = model_path
        logging.info(f"Loading {savedir}")


        if vec_type == 'ecapa_ssl':
            # Load model and processor
            # breakpoint()
            # self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            # self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
            # self.model.to(self.device)
            # self.model.eval()
            
            self.extractor = ECAPA_TDNN_test().to(device)
            checkpoint = torch.load(model_path / "embedding_model.ckpt", map_location=device)
            state_dict = checkpoint.get('state_dict', checkpoint)
            # strip module. prefix
            new_state = { k[len('module.'):] if k.startswith('module.') else k : v
                        for k, v in state_dict.items() }
            res = self.extractor.load_state_dict(new_state, strict=True)
            print(res)  # shows missing/unexpected
            self.extractor.eval()

            self.fbank_computer = Fbank(
                n_mels=80,         # Use the global constant
                left_frames=0, # Use the global constant
                right_frames=0, # Use the global constant
                deltas=False          # Use the global constant
                ).to(device).eval()

           
            self.wavlm_ = WavLMFeatureExtractor(
                model_path / "WavLM-Large.pt",
                device=self.device,
            )
            self.normalizer = InputNormalization(norm_type='sentence', std_norm=False).to(device)

    
        elif vec_type == 'ecapa':
            self.extractor = EncoderClassifier.from_hparams(
                    source=str(model_path),
                    savedir=str(savedir),
                    run_opts={'device': self.device}
            )
  
        else:
            if model_path is None:
                model_path = Path('')
            print("Model Path not found")

       
    def extract_vector(self, audio, sr, wav_path=None):

        if self.vec_type == 'ecapa_ssl':
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

