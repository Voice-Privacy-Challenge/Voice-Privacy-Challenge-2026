# This code is based on
# https://github.com/speechbrain/speechbrain/blob/develop/recipes/VoxCeleb/SpeakerRec/train_speaker_embeddings.py

import os
import re
import torch
import copy
from typing import List, Tuple, Dict, Union
from speechbrain.lobes.models.ECAPA_TDNN import *
from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
from speechbrain.utils.data_utils import download_file
from speechbrain.utils.distributed import run_on_main
import pandas as pd
from .libri_prepare import prepare_libri  # noqa
from .asv_dataset import ASVDatasetGenerator
from ..ecapa_model_wavlm_24layer import ECAPA_TDNN_test
from typing import Union, List
import sys
import soundfile as sf
from ..WavLM import WavLM, WavLMConfig
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

class WavLMFeatureExtractor:
    """
    WavLM feature extractor that searches for audio files across
    predefined folders and performs true batched inference.
    """

    def __init__(self, model_path: str, device: str = None, INPUT_FOLDERS: list = None, max_length: float = 10.0):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.INPUT_FOLDERS = INPUT_FOLDERS
        print(f"Loading WavLM model to {self.device}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.cfg = WavLMConfig(checkpoint['cfg'])
        self.model = WavLM(self.cfg)
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(self.device)
        self.model.eval()
        self.max_length = max_length  # seconds; >max_length: random crop, <max_length: pad
        print(f"Extractor is ready (max_length={max_length}s).")

    def _find_audio_path(self, audio_name: str) -> str:
        """
        Searches predefined folders for the audio file.
        (Original implementation)
        """
        file_name = f"{audio_name}.wav"
        # breakpoint()
        for folder in self.INPUT_FOLDERS:
            full_path = os.path.join(folder, file_name)
            if os.path.isfile(full_path):
                
                return full_path
        raise FileNotFoundError(f"Audio file '{file_name}' not found in any predefined folder.")

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

    def _crop_or_pad(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        """Crop to max_length if longer (random segment), pad if shorter."""
        max_samples = int(self.max_length * sr)
        n = audio.shape[0]
        if n > max_samples:
            start = torch.randint(0, n - max_samples + 1, (1,)).item()
            return audio[start : start + max_samples]
        elif n < max_samples:
            return F.pad(audio, (0, max_samples - n), mode="constant", value=0)
        return audio

    def extract_features(
        self,
        audio_names: Union[str, List[str], Tuple[str, ...]] = None,
        audio_paths: Union[str, List[str], Tuple[str, ...]] = None
    ) -> Dict[str, Union[List[torch.Tensor], List[str]]]:
        """Extract WavLM features. Prefer audio_paths (from CSV wav column) for robustness across all languages."""
        if audio_paths is not None:
            paths = [audio_paths] if isinstance(audio_paths, str) else list(audio_paths)
        elif audio_names is not None:
            names = [audio_names] if isinstance(audio_names, str) else list(audio_names)
            paths = [self._find_audio_path(n) for n in names]
        else:
            raise ValueError("Either audio_names or audio_paths must be provided")

        audio_inputs = []
        for path in paths:
            audio, sr = self._load_audio(path)
            audio = self._crop_or_pad(audio, sr)
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

def normalize_uttid(uttids: Union[str, List[str]]) -> Union[str, List[str]]:
    """
    Normalize utterance IDs for WavLM file lookup:
    1. Remove trailing '_start_end' suffix (e.g. _0.0_3.0) to get base utt_id.
    2. Convert double dashes '--' to single '-' for Libri-style ids.

    E.g. DEP_9e4Ipo3C_0.0_3.0 -> DEP_9e4Ipo3C (Japanese), 1234--5678--9012_0.0_3.0 -> 1234-5678-9012 (Libri)
    """
    if isinstance(uttids, str):
        # Strip trailing _num_num (chunk suffix)
        base = re.sub(r"_[0-9.]+_[0-9.]+$", "", uttids)
        return base.replace("--", "-")
    elif isinstance(uttids, list):
        return [normalize_uttid(u) for u in uttids]
    else:
        raise TypeError(f"Expected str or list of str, got {type(uttids)}")





class SpeakerBrain(sb.core.Brain):
    """Class for speaker embedding training"
    """

    def frame_shuffle_fast(self, wavs, frame_len=400, frame_shift=160):
        #25ms frame with 10ms shift
        #Frame length=0.025×16000=400 samples
        #Frame shift=0.010×16000=160 samples
        B, T = wavs.shape
        num_frames = 1 + (T - frame_len) // frame_shift

        # Frame using unfold: [B, num_frames, frame_len]
        framed = wavs.unfold(dimension=1, size=frame_len, step=frame_shift)

        # Shuffle frame order
        torch.manual_seed(42)
        perm = torch.randperm(num_frames)
        framed_shuffled = framed[:, perm, :]  # [B, num_frames, frame_len]

        # Vectorized overlap-add reconstruction
        # We compute the indices where each frame contributes
        recon = torch.zeros((B, T), device=self.device)
        ones = torch.ones((B, num_frames, frame_len), device=self.device)

        idx = (
            torch.arange(0, num_frames * frame_shift, frame_shift, device=self.device)
            .unsqueeze(1) + torch.arange(frame_len, device=self.device)
        )  # [num_frames, frame_len]
        idx = idx.unsqueeze(0).expand(B, -1, -1)  # [B, num_frames, frame_len]

        # Flatten all
        recon = recon.scatter_add(1, idx.reshape(B, -1), framed_shuffled.reshape(B, -1))
        
        # Normalize by how many times each sample was added (overlap count)
        overlap = torch.zeros((B, T), device=self.device)
        overlap = overlap.scatter_add(1, idx.reshape(B, -1), ones.reshape(B, -1))
        recon = recon / torch.clamp(overlap, min=1.0)
        return recon

    def compute_forward(self, batch, stage):
        """Computation pipeline based on a encoder + speaker classifier.
        Data augmentation and environmental corruption are applied to the
        input speech.
        """
        batch = batch.to(self.device)
        wavs, lens = batch.sig
       
        # Add waveform augmentation if specified.
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
           wavs, lens = self.hparams.wav_augment(wavs, lens)
            
            
        # Feature extraction and normalization
        feats = self.modules.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, lens)

        # Embeddings + speaker classifier: use wav paths from CSV (robust for cn/ja/en/es/de/fr)
        wav_paths = batch.wav if hasattr(batch, "wav") and batch.wav is not None else None
        if wav_paths is not None:
            get_wav2vec2features = self.wavlm_extractor.extract_features(audio_paths=wav_paths)
        else:
            uttid = normalize_uttid(batch.id)
            get_wav2vec2features = self.wavlm_extractor.extract_features(audio_names=uttid)
        embeddings = self.modules.embedding_model(feats, get_wav2vec2features)
        self.emb = embeddings
        outputs = self.modules.classifier(embeddings)
        
        return outputs, lens
    
    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using speaker-id as label. (Modified: safe unpacking + logging)"""
        predictions, lens = predictions

        uttid = batch.id  # list of full uttid strings normally
        
        spkid, _ = batch.spk_id_encoded
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            spkid = self.hparams.wav_augment.replicate_labels(spkid)
            

        loss_aam = self.hparams.compute_cost(predictions, spkid, lens)
        loss =  loss_aam

        if stage == sb.Stage.TRAIN and hasattr(self.hparams.lr_annealing, "on_batch_end"):
            self.hparams.lr_annealing.on_batch_end(self.optimizer)

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(uttid, predictions, spkid, lens)

        return loss
    
    # def on_stage_start(self, stage, epoch=None):
    #     # Unfreeze all parameters first (reset state)
    #     for module in [self.modules.compute_features, self.modules.mean_var_norm,
    #                        self.modules.embedding_model, self.modules.classifier]:
    #         for p in module.parameters():
    #             p.requires_grad = True
        
    #     # --- Embedding Model Freeze Logic for Epoch 1 ---
    #     # In epoch 1, we want to freeze embedding_model but DDP doesn't allow requires_grad=False.
    #     if stage == sb.Stage.TRAIN and epoch == 1:
    #         print("[FREEZE LOGIC] Epoch 1: Saving initial embedding_model state_dict...")
    #         print("[FREEZE LOGIC] Embedding model will be reset after each batch (effectively frozen).")
    #         print("[FREEZE LOGIC] Only compute_features, mean_var_norm, and classifier will be updated.")
            
    #         # Get the state dict - handle DDP wrapper
    #         if hasattr(self.modules.embedding_model, 'module'):
    #             # DDP wrapped
    #             state_dict = self.modules.embedding_model.module.state_dict()
    #         else:
    #             state_dict = self.modules.embedding_model.state_dict()
            
    #         # Deep copy to ensure independent copy
    #         self._frozen_embedding_state = copy.deepcopy(state_dict)
            
    #         # Save to disk as backup
    #         frozen_path = os.path.join(self.hparams.output_folder, "frozen_embedding_model.ckpt")
    #         torch.save(self._frozen_embedding_state, frozen_path)
    #         print(f"[FREEZE LOGIC] Saved frozen embedding state to: {frozen_path}")
        
    #     elif stage == sb.Stage.TRAIN and epoch >= 2:
    #         # Clear frozen state - embedding model will train normally
    #         if hasattr(self, '_frozen_embedding_state') and self._frozen_embedding_state is not None:
    #             print(f"[FREEZE LOGIC] Epoch {epoch}: Clearing frozen state. Embedding model will now train normally.")
    #             self._frozen_embedding_state = None

    #     if stage != sb.Stage.TRAIN:
    #         self.error_metrics = self.hparams.error_stats()
    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of an epoch."""
        # Unfreeze all parameters first (reset state)
        for module in [self.modules.compute_features, self.modules.mean_var_norm,
                           self.modules.embedding_model, self.modules.classifier]:
            for p in module.parameters():
                p.requires_grad = True
        
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()


    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ErrorRate"] = self.error_metrics.summarize("average")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            
            self.checkpointer.save_and_keep_only(
                num_to_keep=7,
                meta={"ErrorRate": stage_stats["ErrorRate"]},
                min_keys=["ErrorRate"],
                name=epoch
            )
            # self.checkpointer.save_checkpoint(
            #     meta={"epoch": epoch, "loss": stage_loss},
            #     name=f"epoch_{epoch}_latest",
            # )
            
    # def on_fit_batch_end(self, batch, outputs, loss, should_step):
    #     """Called at the end of each fit_batch.
        
    #     V2: Resets embedding_model to frozen state after each optimizer step in epoch 1.
    #     This effectively "freezes" the embedding model while still allowing gradients
    #     to flow (required for DDP).
    #     """
    #     # Call parent implementation if it exists
    #     if hasattr(super(), 'on_fit_batch_end'):
    #         super().on_fit_batch_end(batch, outputs, loss, should_step)
        
    #     # Only reset embedding model during epoch 1 after optimizer step
    #     if should_step and hasattr(self, '_frozen_embedding_state') and self._frozen_embedding_state is not None:
    #         # Reset embedding model weights to frozen state
    #         frozen_state = self._frozen_embedding_state
            
    #         # Handle DDP wrapper
    #         if hasattr(self.modules.embedding_model, 'module'):
    #             # DDP wrapped - load into the inner module
    #             self.modules.embedding_model.module.load_state_dict(frozen_state, strict=True)
    #         else:
    #             self.modules.embedding_model.load_state_dict(frozen_state, strict=True)
            
    #         # Debug print (every 100 steps to reduce log spam)
    #         if self.step >=250:
    #             self._frozen_embedding_state = None
    #             print(f"[FREEZE LOGIC] Step {self.step}: Reset embedding_model to frozen state.")
                
                
    def fit_batch(self, batch):
        """Fit one batch, override to do multiple updates.

        The default implementation depends on a few methods being defined
        with a particular behavior:

        * ``compute_forward()``
        * ``compute_objectives()``
        * ``optimizers_step()``

        Also depends on having optimizers passed at initialization.

        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for training. Default implementation assumes
            this batch has two elements: inputs and targets.

        Returns
        -------
        detached loss
        """
        should_step = (self.step % self.grad_accumulation_factor) == 0
        self.on_fit_batch_start(batch, should_step)

        with self.no_sync(not should_step):
            with self.training_ctx:
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            scaled_loss = self.scaler.scale(
                loss / self.grad_accumulation_factor
            )
            self.check_loss_isfinite(scaled_loss)
            scaled_loss.backward()

        if should_step:
            self.optimizers_step()

        self.on_fit_batch_end(batch, outputs, loss, should_step)
        return loss.detach().cpu()



def _convert_to_yaml(overrides):
    # convert dict to yaml for overrides
    yaml_string = ""
    for key in overrides:
        yaml_string += str(key) +': ' +str(overrides[key]) + '\n'
    return yaml_string.strip()

        
def train_asv_speaker_embeddings_ssl_ft(config_file, hparams_file, run_opts):
    # breakpoint()
    overrides = _convert_to_yaml(hparams_file)
    
    with open(config_file) as f:
        hparams = load_hyperpyyaml(f, overrides)
    
    
    run_on_main(
        prepare_libri,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["save_folder"],
            "splits": ["train", "dev"],
            "split_ratio": [90, 10],
            "num_utt": hparams["num_utt"],
            "num_spk": hparams["num_spk"],
            "seg_dur": hparams["sentence_len"],
            "skip_prep": hparams["skip_prep"],
            "utt_selected_ways": hparams["utt_selected_ways"],
        },
    )
    
    
    asv_dataset_gen = ASVDatasetGenerator(hparams)
    train_data, valid_data = asv_dataset_gen.dataio_prep()
    
    sb.core.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=config_file,
        overrides=overrides,
    )
    
    
    
    sb.utils.distributed.ddp_init_group(run_opts)
    run_opts["find_unused_parameters"] = True
    device = run_opts.get("device", "cpu")
    ecapa_ssl = ECAPA_TDNN_test().to(device)
    
    #Checking for pretrained path
    if os.path.exists(hparams['pretrained_path']):
        print(f"Loading pretrained model from: {hparams['pretrained_path'] }")
        try:
            checkpoint = torch.load(hparams['pretrained_path'] , map_location=device)
            state_dict = checkpoint.get('state_dict', checkpoint)
            # strip module. prefix
            state_dict = { k[len('module.'):] if k.startswith('module.') else k : v
                        for k, v in state_dict.items() }
            
            missing, unexpected = ecapa_ssl.load_state_dict(state_dict, strict=True)
            print(f"Pretrained model loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
            if len(missing) > 0:
                print(f"Missing keys details: {missing}")
        except Exception as e:
            print(f"ERROR: Failed to load pretrained model: {e}")
    
    hparams['modules']['embedding_model'] = ecapa_ssl
    hparams['checkpointer'].add_recoverable('embedding_model', ecapa_ssl)
    # Init model
    speaker_brain = SpeakerBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    INPUT_FOLDERS = [
        hparams['data_folder']+'/wav',
    ]
    max_length = hparams.get("wavlm_max_length", 3.0)  # seconds
    wavlm_ = WavLMFeatureExtractor(
        hparams["pretrained_wavlm_model"],
        device=device,
        INPUT_FOLDERS=INPUT_FOLDERS,
        max_length=max_length,
    )
    speaker_brain.wavlm_extractor = wavlm_

    speaker_brain.fit(
        speaker_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )
