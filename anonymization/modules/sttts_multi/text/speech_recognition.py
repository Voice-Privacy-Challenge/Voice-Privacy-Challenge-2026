from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import time
from torch.multiprocessing import set_start_method
from itertools import cycle, repeat
import numpy as np
from pathlib import Path
import torch
import soundfile
import resampy
import re

from .text import Text
from .recognition.whisper import WhisperASR
from utils import read_kaldi_format, setup_logger

set_start_method('spawn', force=True)
logger = setup_logger(__name__)


class ASRDataset(torch.utils.data.Dataset):
    def __init__(self, utt2spk, wav_scp, already_recognized_utts, utterance_list):
        self.utterances = []
        for utt, spk in utt2spk.items():
            if utt not in wav_scp:
                continue
            if utt in already_recognized_utts:
                continue
            if utterance_list and utt not in utterance_list:
                continue
            if utt in wav_scp:
                self.utterances.append((utt, spk, wav_scp[utt]))

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt, spk, wav_path = self.utterances[idx]
        speech, rate = soundfile.read(wav_path)
        if rate != 16000:
            speech = resampy.resample(speech, rate, 16000)
        return {'raw': speech, 'sampling_rate': 16000, 'utt': utt, 'spk': spk}


def run_process(params):
    utt2spk, wav_scp, already_recognized_utts, utterance_list, asr_model, out_dir, save_intermediate, \
        lang, sleep, job_id = params
    time.sleep(sleep)
    asr_dataset = ASRDataset(utt2spk=utt2spk, wav_scp=wav_scp, already_recognized_utts=already_recognized_utts,
                             utterance_list=utterance_list)
    return asr_model.recognize_speech_of_dataset(asr_dataset, out_dir=out_dir, save_intermediate=save_intermediate,
                                                 job_id=job_id, lang=lang)


class SpeechRecognition:

    def __init__(self, devices, settings, results_dir=None, save_intermediate=True, force_compute=False):
        self.devices = devices
        self.save_intermediate = save_intermediate
        self.force_compute = force_compute if force_compute else settings.get('force_compute_recognition', False)
        self.n_processes = len(self.devices)
        self.gold_langs = settings.get('gold_langs', False)

        self.model_hparams = settings

        if results_dir:
            self.results_dir = results_dir
        elif 'results_path' in settings:
            self.results_dir = settings['results_path']
        elif 'results_dir' in settings:
            self.results_dir = settings['results_dir']
        else:
            if self.save_intermediate:
                raise ValueError('Results dir must be specified in parameters or settings!')

        if settings.get('recognizer', None) is None:  # use gold text
            self.asr_models = None
            self.is_phones = False
            self.save_intermediate = False
        else:
            self.asr_models = [self._create_model_instance(hparams=self.model_hparams, device=device)
                              for device in self.devices]
            self.is_phones = (self.asr_models[0].output == 'phones')

    def recognize_speech(self, dataset_path, dataset_name=None, utterance_list=None):
        dataset_name = dataset_name if dataset_name else dataset_path.name
        dataset_results_dir = self.results_dir / dataset_name if self.save_intermediate else Path('')

        if self.asr_models is None:
            return self._load_gold_transcripts(dataset_path)

        utt2spk = read_kaldi_format(dataset_path / 'utt2spk')
        texts = Text(is_phones=self.is_phones)

        if (dataset_results_dir / 'text').exists() and not self.force_compute:
            # if the text created from this ASR model already exists for this dataset and a computation is not
            # forced, simply load the text
            texts.load_text(in_dir=dataset_results_dir)

        if len(texts) == len(utt2spk):
            logger.info('No speech recognition necessary; load existing text instead...')
        else:
            if len(texts) > 0:
                logger.info(f'No speech recognition necessary for {len(texts)} of {len(utt2spk)} utterances')
            # otherwise, recognize the speech
            dataset_results_dir.mkdir(exist_ok=True, parents=True)
            logger.info(f'Recognize speech of {len(utt2spk) - len(texts)} utterances...')
            wav_scp = read_kaldi_format(dataset_path / 'wav.scp')
            if self.gold_langs:
                # Each dataset contains audios in only one language, we can read this language from the path name
                lang = self._extract_lang_from_path(list(wav_scp.values())[0])
            else:
                lang = None

            save_intermediate = self.save_intermediate and not utterance_list
            start = time.time()
            if self.n_processes == 1:
                params = [utt2spk, wav_scp, texts.utterances, utterance_list, self.asr_models[0],
                          dataset_results_dir, save_intermediate, lang, 0, None]
                new_texts = [run_process(params)]
            else:
                sleeps = [10 * i for i in range(self.n_processes)]
                utt2spk_jobs = [{k: v for k, v in list(utt2spk.items())[i::self.n_processes]}
                                for i in range(self.n_processes)]
                params = zip(utt2spk_jobs, # utterances to recognize
                             repeat(wav_scp), # wav paths
                             repeat(texts.utterances), # already recognized utterances
                             repeat(utterance_list), # sub list of utterances
                             self.asr_models, # asr model
                             repeat(dataset_results_dir),  # out_dir
                             repeat(save_intermediate),  # whether to save intermediate results
                             repeat(lang),  # whether to use gold languages or recognize lang from audio
                             sleeps, # avoid starting all processes at same time
                             list(range(self.n_processes))) # job_id
                new_texts = process_map(run_process, params, max_workers=self.n_processes)


            end = time.time()
            total_time = round(end - start, 2)
            logger.info(f'Total time for speech recognition: {total_time} seconds ({round(total_time / 60, 2)} minutes / '
                  f'{round(total_time / 60 / 60, 2)} hours)')

            texts = self._combine_texts(main_text_instance=texts, additional_text_instances=new_texts)
            if save_intermediate:
                texts.save_text(out_dir=dataset_results_dir)
                self._remove_temp_files(out_dir=dataset_results_dir)

        return texts

    def _extract_lang_from_path(self, wav_path):
        # Taken from evaluation/utility/asr/whisper_asr/inference.py
        language_map = {
            "en": "en",
            "de": "de",
            "fr": "fr",
            "es": "es",
            "zh": "cn",
            "ja": "ja",
        }
        # Extract language from path segments
        # Handle two path formats:
        # 1. corpora/en/dev/... or corpora/cn/test/... (language as separate segment)
        # 2. data/en_dev_trials_f_ssl/wav/... (language embedded in directory name)
        path_segments = re.split(r'[/\\]', wav_path)
        for segment in path_segments:
            # First check if segment exactly matches a language code
            for k, v in language_map.items():
                if segment == k or segment == v:
                    return k
            # If no exact match, check if segment starts with language code (e.g., "en_dev_trials_f_ssl")
            for k, v in language_map.items():
                # Check if segment starts with language code followed by underscore or end of string
                if segment.startswith(f'{k}_') or segment.startswith(f'{v}_'):
                    return k
        return None

    def _load_gold_transcripts(self, dataset_path):
        logger.info('Load gold transcripts instead of recognizing them')
        utt_start_token = self.model_hparams['utt_start_token']
        utt_end_token = self.model_hparams['utt_end_token']
        texts = Text(is_phones=self.is_phones)
        texts.load_text(dataset_path)
        if utt_start_token and utt_end_token:
            texts.sentences = [utt_start_token + sentence + utt_end_token for sentence in texts.sentences]
        return texts

    def _create_model_instance(self, hparams, device):
        recognizer = hparams.get('recognizer')
        if recognizer == 'whisper':
            return WhisperASR(**hparams, device=device)
        else:
            raise ValueError(f'Invalid recognizer option: {recognizer}')

    def _combine_texts(self, main_text_instance, additional_text_instances):
        for add_text_instance in additional_text_instances:
            main_text_instance.add_instances(sentences=add_text_instance.sentences,
                                             utterances=add_text_instance.utterances,
                                             speakers=add_text_instance.speakers,
                                             languages=add_text_instance.languages)

        return main_text_instance

    def _remove_temp_files(self, out_dir):
        temp_text_files = [filename for filename in out_dir.glob('text*') if filename.name != 'text']
        temp_utt2spk_files = [filename for filename in out_dir.glob('utt2spk*') if filename.name != 'utt2spk']

        for file in temp_text_files + temp_utt2spk_files:
            file.unlink()
