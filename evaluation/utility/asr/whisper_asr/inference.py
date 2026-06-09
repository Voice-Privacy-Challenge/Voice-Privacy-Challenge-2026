import sys
import re
from utils import save_kaldi_format
from copy import deepcopy
from speechbrain.utils.metric_stats import ErrorRateStats
import tqdm
import torch
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from utils import read_kaldi_format

# Punctuation removed before WER so ref/hyp match VPC normalized ground-truth transcripts.
_NON_WORD_PUNCT = re.compile(r'[.,!?;:»«""()\[\]…—–\-]')


class ASRDataset(torch.utils.data.Dataset):
    def __init__(self, wav_scp_file, asr_model):
        self.data = []
        for utt_id, wav_file in read_kaldi_format(wav_scp_file).items():
            #wav, sr = torchaudio.load(str(wav_file))
            #wav = asr_model.load_audio(wav_file)
            #wav_len = len(wav.squeeze())
            self.data.append((utt_id, wav_file))

        # Sort the data based on audio length
        #self.data = sorted(data, key=lambda x: x[2], reverse=True)

    def __getitem__(self, idx):
        utt_id, wav_file = self.data[idx]
        return utt_id, wav_file

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):  ## make them all the same length with zero padding
        utt_ids, wav_files = zip(*batch)
       
        return utt_ids, wav_files



class InferenceWhisperASR:

    """
    Drop-in-ish replacement for your SpeechBrain ASR inference class, but using openai-whisper.

    Notes:
    - Whisper expects 16kHz audio. For tensor batches, this assumes `inputs` are already
      16kHz waveforms in float (typically [-1, 1]) and `lengths` are sample counts.
    - For file-based transcription, Whisper handles loading/resampling internally via whisper.load_audio().
    """

    def __init__(
        self,
        model_path: str = "openai/whisper-large-v3",
        device: str = "cuda",
    ):
        self.device = device
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.language_map: dict = {
            "en": "en",
            "de": "de",
            "fr": "fr",
            "es": "es",
            "zh": "cn",
            "ja": "ja",
        }

        self.asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        self.asr_model.to(device)
        processor = AutoProcessor.from_pretrained(model_path)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.asr_model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
            return_timestamps=False,  # Disable timestamps to improve batch processing efficiency
        )



    def _normalize_transcript(self, text):
        """Match VPC ground-truth format: lowercase, no punctuation, collapsed spaces."""
        text = text.strip().lower()
        text = _NON_WORD_PUNCT.sub('', text)
        return ' '.join(text.split())

    def plain_text_key(self, path):
        tokens = []  # key: token_list
        for token in path:
            # For Chinese text with pinyin format (e.g., "深 shen1 交 jiao1"), 
            # extract only Chinese characters (remove pinyin)
            cleaned_token = self._extract_chinese_chars(token.strip())
            
            # Check if text contains Chinese characters
            has_chinese = bool(re.search(r'[\u4e00-\u9fff]', cleaned_token))
            
            if has_chinese:
                # For Chinese text, split by character (character-level evaluation)
                # Remove all spaces and split into individual characters
                cleaned_token = cleaned_token.replace(' ', '')
                tokens.append(list(cleaned_token) if cleaned_token else [])
            else:
                # For non-Chinese text, split by spaces (word-level evaluation)
                cleaned_token = self._normalize_transcript(cleaned_token)
                tokens.append(cleaned_token.split(' ') if cleaned_token else [])
        return tokens
    
    def _extract_chinese_chars(self, text):
        """Extract Chinese characters from text that may contain pinyin.
        
        Example: "深 shen1 交 jiao1 所 suo3" -> "深交所"
        """
        # Pattern to match Chinese characters (CJK Unified Ideographs)
        chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
        # Find all Chinese character sequences and join them
        chinese_chars = chinese_pattern.findall(text)
        return ''.join(chinese_chars) if chinese_chars else text

    def transcribe_audios(self, data, out_file):
        texts = {}
        for batch in tqdm.tqdm(data):
            utt_ids, wav_files = batch
            wav_files = list(wav_files)
            #print(wav_files[0])
            language = None
            wav_path_str = str(wav_files[0]).lower()
            # Extract language from path segments
            # Handle two path formats:
            # 1. corpora/en/dev/... or corpora/cn/test/... (language as separate segment)
            # 2. data/en_dev_trials_f_ssl/wav/... (language embedded in directory name)
            path_segments = re.split(r'[/\\]', wav_path_str)
            for segment in path_segments:
                # First check if segment exactly matches a language code
                for k, v in self.language_map.items():
                    if segment == k or segment == v:
                        language = k
                        break
                if language:
                    break
                
                # If no exact match, check if segment starts with language code (e.g., "en_dev_trials_f_ssl")
                if not language:
                    for k, v in self.language_map.items():
                        # Check if segment starts with language code followed by underscore or end of string
                        if segment.startswith(f'{k}_') or segment.startswith(f'{v}_'):
                            language = k
                            break
                    if language:
                        break
            
            generate_kwargs = {"language": language} if language else {}
            #print(f"Decoding language: {language}")
            
            predicts = self.pipe(wav_files, batch_size=len(wav_files), generate_kwargs=generate_kwargs)
            for i, utt_id in enumerate(utt_ids):
                texts[deepcopy(utt_id)] = str(predicts[i]["text"])

        out_file.parent.mkdir(exist_ok=True, parents=True)
        save_kaldi_format(texts, out_file)
        return texts

    def compute_wer(self, ref_texts, hyp_texts, out_file):
        wer_stats = ErrorRateStats()

        ids = []
        predicted = []
        targets = []
        for utt_id, ref in ref_texts.items():
            ids.append(utt_id)
            targets.append(ref)
            predicted.append(hyp_texts[utt_id])

        wer_stats.append(ids=ids, predict=self.plain_text_key(predicted), target=self.plain_text_key(targets))
        out_file.parent.mkdir(exist_ok=True, parents=True)

        with open(out_file, 'w') as f:
            wer_stats.write_stats(f)

        return wer_stats





