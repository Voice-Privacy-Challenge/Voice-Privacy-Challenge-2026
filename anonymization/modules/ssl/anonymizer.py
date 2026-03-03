import os
import shutil
import warnings
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import resampy
from typing import Optional, Union, Tuple, Dict, List
from kaldiio import ReadHelper
from librosa.util import normalize as librosa_normalize
import fairseq
from fairseq.data import Dictionary
from speechbrain.lobes.features import Fbank
from speechbrain.processing.features import InputNormalization
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import amfm_decompy.basic_tools as basic
import amfm_decompy.pYAAPT as pYAAPT

# Local utilities
from .utils.softhubert import ContentExtractor
from .utils.spk import SpeakerEmbeddingExtractor, SpeakerPoolManager, GenderDetector
from .utils.hifigan import Vocoder
from .utils.f0 import F0Extractor

# ----------------------------- Constants -----------------------------
TARGET_SR = 16000
F0_HOP_SIZE = 160
SSL_HOP_SIZE = 320

SOURCE_DIR = "exp/ssl_models/provided/"
HUBERT_MODEL_PATH = "exp/ssl_models/hubert_base_ls960.pt"
SOFT_MODEL_PATH = os.path.join(SOURCE_DIR, "HuBERT_soft/soft_model.pt")
ECAPA_CKPT_PATH = os.path.join(SOURCE_DIR, "ECAPA-TDNN/embedding_model.ckpt")
HIFIGAN_CKPT_PATH = os.path.join(SOURCE_DIR, "HiFi-GAN/libri_tts_clean_100_fbank_xv_ssl_freeze/g_00100000")

# Pooling / dataset paths
XVECTOR_SCP_PATH = "exp/ssl_models/spk_embedding/libritts_train_other_500/xvector.scp"
GENDER_MAP_PATH = "exp/ssl_models/spk2gender"


class SelectionBasedAnonymizationPipeline:
    """
    Enhanced gender-aware kNN speaker-pooling VC pipeline
    """
    def __init__(
        self,
        device: str = "cpu",
        xvector_scp_path: str = XVECTOR_SCP_PATH,
        gender_map_path: str = GENDER_MAP_PATH,
        default_world_size: int = 50,
        default_region_size: int = 10,
        default_flag_proximity: str = "random",
        default_flag_cross_gender: bool = False,
        default_gender_pool: bool = False,
        default_avg: str = "mean",
        target_sr: int = TARGET_SR,
        ssl_hop: int = SSL_HOP_SIZE,
        f0_hop: int = F0_HOP_SIZE,
        print_debug: bool = False,
    ):
        self.device = device
        self.target_sr = int(target_sr)
        self.ssl_hop = int(ssl_hop)
        self.f0_hop = int(f0_hop)
        self.default_world_size = int(default_world_size)
        self.default_region_size = int(default_region_size)
        self.default_flag_proximity = default_flag_proximity
        self.default_flag_cross_gender = bool(default_flag_cross_gender)
        self.default_gender_pool = bool(default_gender_pool)
        self.default_avg = default_avg.lower()
        self.cached_speaker_vector = None
        self.anonymizer_cache_state = False
        self.skip_caching = False
        self.print_debug = bool(print_debug)

        self._require_file(HUBERT_MODEL_PATH, "HuBERT model")
        self._require_file(SOFT_MODEL_PATH, "HuBERT soft model")
        self._require_file(ECAPA_CKPT_PATH, "ECAPA checkpoint")
        self._require_file(HIFIGAN_CKPT_PATH, "HiFi-GAN checkpoint")
        self._require_file(xvector_scp_path, "xvector scp")
        self._initialize_components(xvector_scp_path, gender_map_path)

    @staticmethod
    def _require_file(path: str, desc: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing {desc} at: {path}")
    
    def _initialize_components(self, xvector_scp_path: str, gender_map_path: str):
        """Initialize all components"""
        # Required for fairseq (PyTorch 2.1+)
        import argparse
        if hasattr(torch.serialization, 'add_safe_globals'):
            torch.serialization.add_safe_globals([argparse.Namespace, Dictionary])
        
        # Initialize components
        self.content_extractor = ContentExtractor(
            hubert_model_path=HUBERT_MODEL_PATH,
            soft_model_path=SOFT_MODEL_PATH,
            device=self.device
        )
        
        self.speaker_extractor = SpeakerEmbeddingExtractor(
            ecapa_ckpt_path=ECAPA_CKPT_PATH,
            device=self.device
        )
        
        # self.gender_detector = GenderDetector(device=self.device)
        self.gender_detector = None
        self.speaker_pool = SpeakerPoolManager(
            xvector_scp_path=xvector_scp_path,
            gender_map_path=gender_map_path
        )
        
        self.vocoder = Vocoder(
            hifigan_ckpt_path=HIFIGAN_CKPT_PATH,
            device=self.device
        )
        
        # F0 extractor
        self.f0_extractor = F0Extractor()
    
   
    def _preprocess_audio(self, input_array: np.ndarray, sample_rate: int) -> torch.Tensor:
        """Preprocess audio for feature extraction"""
        x = np.asarray(input_array).squeeze()
        
        if x.size == 0:
            raise ValueError("Empty input audio.")
        
        # Convert to float32 if needed
        if x.dtype.kind in {"i", "u"}:
            x = x.astype(np.float32) / 32768.0
        elif x.dtype != np.float32:
            x = x.astype(np.float32)
        
        # Resample
        if sample_rate != self.target_sr:
            x = resampy.resample(x, sample_rate, self.target_sr)
        
        # Normalize
        x = librosa_normalize(x) * 0.95
        
        # Convert to tensor
        audio = torch.from_numpy(x).float().unsqueeze(0)
        
        # Ensure length is multiple of SSL hop
        if audio.size(1) < self.ssl_hop:
            raise ValueError(f"Audio too short ({audio.size(1)} samples) for SSL hop {self.ssl_hop}.")
        
        ssl_len = audio.size(1) // self.ssl_hop
        audio = audio[:, : ssl_len * self.ssl_hop]
        
        return audio
    
    @torch.inference_mode()
    def process_utterance(
        self,
        input_array: np.ndarray,
        sample_rate: int = 16000,
        world_size: Optional[int] = None,
        region_size: Optional[int] = None,
        flag_proximity: Optional[str] = None,
        flag_cross_gender: Optional[bool] = None,
        gender_pool: Optional[bool] = None,
        avg: Optional[str] = None,
        cache_query: bool = False,
        return_dict: bool = False,
        return_neighbors: bool = False,
    ) -> Union[np.ndarray, dict]:
        """
        Main pipeline execution for a single utterance
        """
        # Use defaults if not provided
        world_size = world_size if world_size is not None else self.default_world_size
        region_size = region_size if region_size is not None else self.default_region_size
        flag_proximity = flag_proximity if flag_proximity is not None else self.default_flag_proximity
        flag_cross_gender = flag_cross_gender if flag_cross_gender is not None else self.default_flag_cross_gender
        gender_pool = gender_pool if gender_pool is not None else self.default_gender_pool
        avg = (avg or self.default_avg).lower()
        
        # Validate parameters
        if flag_proximity not in {"farthest", "nearest", "random"}:
            raise ValueError("flag_proximity must be 'farthest', 'nearest' or 'random'")
        if world_size <= 0:
            raise ValueError("world_size must be >= 1")
        if region_size <= 0:
            raise ValueError("region_size must be >= 1")
        if region_size > world_size:
            print(f"[WARN] region_size ({region_size}) > world_size ({world_size}). Using region_size = world_size.")
            region_size = world_size
        
        # Preprocess audio
        audio_tensor = self._preprocess_audio(input_array, sample_rate)
        
        # Detect gender
        detected_gender = None
        if gender_pool:
            if self.gender_detector is None:
                # Lazy load if requested but not initialized
                from .utils.spk import GenderDetector
                self.gender_detector = GenderDetector(device=self.device)
            detected_gender = self.gender_detector.detect(input_array, sample_rate)
            if self.print_debug:
                print(f"Detected gender: {detected_gender}")
        else:
            if self.print_debug:
                print("Gender detection skipped (gender_pool=False)")
        
        if self.print_debug:
            print(f"Using world_size={world_size}, region_size={region_size}, flag_proximity={flag_proximity}")
            print(f"Cross-gender: {flag_cross_gender}, Gender-pool: {gender_pool}")
        
        # Determine gender for pooling
        gender_for_pooling = None
        if gender_pool and detected_gender:
            if flag_cross_gender:
                gender_for_pooling = "m" if detected_gender == "f" else "f"
            else:
                gender_for_pooling = detected_gender
        
        # Extract features
        f0_np = self.f0_extractor.extract(audio_tensor.numpy(), rate=self.target_sr, interp=False)
        f0 = torch.as_tensor(f0_np, dtype=torch.float32)
        
        if np.isnan(f0.numpy()).any():
            f0 = torch.nan_to_num(f0, nan=0.0)
        
        T = f0.shape[-1]  # Number of frames
        
        # Extract content features
        content = self.content_extractor.extract(audio_tensor.to(self.device))
        content = F.layer_norm(content, content.shape).transpose(2, 1)
        content = F.pad(content, (0, 1), "replicate").to(torch.float32)
        
        # Enhanced speaker selection
        metadata_available = False
        if not self.anonymizer_cache_state or self.skip_caching:
            if self.print_debug:
                print("Enhanced speaker selection processing")
            
            # Extract query speaker embedding
            query_vec = self.speaker_extractor.extract(audio_tensor)
            query_vec_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
            
            # Get appropriate pool view
            pool_keys, X, Xn = self.speaker_pool.get_pool_view(gender_for_pooling, gender_pool)
            
            if Xn.shape[0] == 0:
                raise ValueError("Empty xvector pool view.")
            
            # Enhanced speaker selection
            selected_idx, selected_sims = self._select_speakers_enhanced(
                query_vec_norm, pool_keys, X, Xn, world_size, region_size, flag_proximity
            )
            
            selected_vecs = X[selected_idx]
            selected_keys = pool_keys[selected_idx]
            
            # Averaging strategies
            pooled = np.mean(selected_vecs, axis=0)
   
            # Normalize pooled vector
            pooled = pooled / (np.linalg.norm(pooled) + 1e-8)
            
            # Prepare speaker tensor
            speaker = torch.from_numpy(pooled.astype(np.float32)).unsqueeze(0).unsqueeze(-1)
            speaker = F.layer_norm(speaker, speaker.shape)
            
            # Cache speaker vector
            self.cached_speaker_vector = speaker
            self.anonymizer_cache_state = True
            metadata_available = True
            
        else:
            # Use cached speaker vector
            speaker = self.cached_speaker_vector
        

        # Align features
        content_up = self.vocoder.feature_upsample(content, T)
        speaker_up = self.vocoder.feature_upsample(speaker.to(device=content_up.device, dtype=content_up.dtype), T)
        
        # Ensure f0 is on the correct device and has correct shape [1, 1, T]
        f0_tensor = f0.to(device=content_up.device, dtype=content_up.dtype, non_blocking=True)
        if f0_tensor.ndim == 2:
            f0_tensor = f0_tensor.unsqueeze(0)
        
        # Concatenate features
        feats = torch.cat([content_up, f0_tensor, speaker_up], dim=1)
        
        # Get vocoder dtype
        vdtype = next(self.vocoder.generator.parameters()).dtype
        feats = feats.to(device=self.device, dtype=vdtype, non_blocking=True)
        
        # Synthesize audio
        audio_synth = self.vocoder.synthesize(feats)
        audio_out = audio_synth.detach().to("cpu").numpy().squeeze().astype(np.float32)
        
        # Prepare output
        if return_dict:
            if metadata_available:
                output = {
                    "audio": audio_out,
                    "world_size": world_size,
                    "region_size": region_size,
                    "flag_proximity": flag_proximity,
                    "flag_cross_gender": flag_cross_gender,
                    "gender_pool": gender_pool,
                    "detected_gender": detected_gender,
                    "gender_for_pooling": gender_for_pooling,
                    "similarities": selected_sims if metadata_available else None,
                    "neighbors": selected_keys if (return_neighbors and metadata_available) else None,
                    "pooled_speaker": pooled if metadata_available else None,
                }
            else:
                output = {
                    "audio": audio_out,
                    "world_size": world_size,
                    "region_size": region_size,
                    "flag_proximity": flag_proximity,
                    "flag_cross_gender": flag_cross_gender,
                    "gender_pool": gender_pool,
                    "detected_gender": detected_gender,
                    "gender_for_pooling": gender_for_pooling,
                    "similarities": None,
                    "neighbors": None,
                    "pooled_speaker": None,
                    "note": "Metadata not available in cached mode",
                }
            return output
        
        return audio_out

    @torch.inference_mode()
    def run(
        self,
        dataset_path: str,
        results_dir: str,
        anon_level: str = "utterance",
        settings: dict = {},
        force_compute: bool = False,
        **kwargs
    ):
        """
        Process a dataset
        """
        import soundfile as sf
        from pathlib import Path
        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(x): return x

        # Prepare paths
        dataset_path = Path(dataset_path)
        results_dir = Path(results_dir)
        ds_name = dataset_path.name
        
        # Add suffix to output directory if provided
        suffix = settings.get("anon_suffix", "")
        out_folder = results_dir / f"{ds_name}{suffix}/wav"  # Output folder is wav/
        out_folder.mkdir(parents=True, exist_ok=True)

        # Merge settings into kwargs for process_utterance
        run_kwargs = {
            "world_size": settings.get("world_size"),
            "region_size": settings.get("region_size"),
            "flag_proximity": settings.get("flag_proximity"),
            "flag_cross_gender": settings.get("flag_cross_gender"),
            "gender_pool": settings.get("gender_pool"),
            "avg": settings.get("avg"),
        }
        # Update with explicitly passed kwargs
        run_kwargs.update({k: v for k, v in kwargs.items() if v is not None})

        # Look for wav.scp
        scp_file = dataset_path / "wav.scp"
        if not scp_file.exists():
            # If no scp, just look for wav files
            wav_files = list(dataset_path.glob("*.wav"))
            if not wav_files:
                raise FileNotFoundError(f"No wav.scp or wav files found in {dataset_path}")
            data_to_process = [(f.stem, str(f)) for f in wav_files]
        else:
            with open(scp_file, "r") as f:
                data_to_process = []
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        data_to_process.append((parts[0], parts[1]))

        print(f"Processing {len(data_to_process)} utterances from {dataset_path}...")
        print(f"Anonymization level: {anon_level}")

        current_spk = None
        suffix = settings.get("anon_suffix", "")
        n_skip_exist = 0
    

        for utt_id, wav_path in tqdm(data_to_process):
            # Determine speaker ID for caching
            # VPC format is usually spkID-uttID
            spk_id = utt_id.split('-')[0] if '-' in utt_id else utt_id

            # When force_compute=False: skip if output already exists
            out_path = out_folder / f"{utt_id}.wav"
            if out_path.exists() and not force_compute:
                n_skip_exist += 1
                continue

            # Load audio
            audio, sr = sf.read(wav_path)
            
            # Cache management
            if anon_level == "utterance" or anon_level == "random_per_utt":
                self.reset_cache()
            elif (anon_level == "speaker" or anon_level == "constant") and spk_id != current_spk:
                self.reset_cache()
                current_spk = spk_id
            
            # Run anonymization
            try:
                anon_audio = self.process_utterance(audio, sample_rate=sr, **run_kwargs)
                
                # Save audio
                sf.write(out_path, anon_audio, samplerate=self.target_sr)
            except Exception as e:
                print(f"\nError processing {utt_id}: {e}")

        if n_skip_exist or n_skip_copy:
            print(f"Skipped: {n_skip_exist} (already exist), {n_skip_copy} (copied from trials_f/m)")
    
    def _select_speakers_enhanced(
        self,
        query_vec_norm: np.ndarray,
        pool_keys: np.ndarray,
        X: np.ndarray,
        Xn: np.ndarray,
        world_size: int,
        region_size: int,
        flag_proximity: str = "farthest"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Enhanced speaker selection with farthest/nearest/random + sampling
        """
        # Compute similarities
        sims = Xn @ query_vec_norm  # (N,)
        
        # Select based on proximity flag
        if flag_proximity == "random":
            # Completely random selection from the pool
            all_idx = np.arange(Xn.shape[0])
            k_eff = min(region_size, Xn.shape[0])
            selected_idx = np.random.choice(all_idx, k_eff, replace=False)
            return selected_idx, sims[selected_idx]

        if flag_proximity == "farthest":
            # Select farthest (most different) speakers
            k_eff = min(world_size, Xn.shape[0])
            top_idx = np.argpartition(sims, k_eff)[:k_eff]  # Get k_eff smallest similarities
            top_idx = top_idx[np.argsort(sims[top_idx])]  # Sort ascending
        else:  # "nearest"
            # Select nearest (most similar) speakers
            k_eff = min(world_size, Xn.shape[0])
            top_idx = np.argpartition(sims, -k_eff)[-k_eff:]  # Get k_eff largest similarities
            top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]  # Sort descending
        
        # Random sampling from selected top-k speakers
        if region_size < len(top_idx):
            random.seed()  # Use current time as seed
            selected_idx = random.sample(list(top_idx), region_size)
        else:
            selected_idx = top_idx
        
        return np.array(selected_idx), sims[selected_idx]

    def reset_cache(self):
        """Reset pipeline caches"""
        self.cached_speaker_vector = None
        self.anonymizer_cache_state = False
        #print("Pipeline cache reset")
