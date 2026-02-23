import torch
import torchaudio
import tqdm
import pandas as pd
import warnings

from pathlib import Path
from sklearn.metrics import recall_score, accuracy_score
from utils import read_kaldi_format, scan_checkpoint, setup_logger

logger = setup_logger(__name__)

# IEMOCAP 4-class labels (used by emotion2vec backend)
IEMOCAP_LABELS = ["ang", "hap", "neu", "sad"]
LAB2IND = {l: i for i, l in enumerate(IEMOCAP_LABELS)}


class FoldSERDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        data = []
        utt2spk = read_kaldi_format(data_path / "utt2spk")
        for utt_id, wav_file in read_kaldi_format(data_path / "wav.scp").items():
            wav, sr = torchaudio.load(str(wav_file))
            wav_len = wav.shape
            spk = utt2spk[utt_id]
            data.append((utt_id, spk, wav, wav_len, sr))

        # Sort the data based on audio length
        self.data = sorted(data, key=lambda x: x[3], reverse=True)

    def __getitem__(self, idx):
        wavname, spk, wav, wav_len, sr = self.data[idx]
        return wavname, spk, wav, wav_len, sr

    def __len__(self):
        return len(self.data)

def _eval_ser_speechbrain(eval_datasets, eval_data_dir, models_path, anon_data_suffix, params, device):
    from speechbrain.inference.interfaces import foreign_class
    results_dir = params['results_dir']
    test_sets = eval_datasets + [f'{dataset}{anon_data_suffix}' for dataset in eval_datasets]
    results = []
    classifiers = {}
    for test_set in tqdm.tqdm(test_sets):
        data_path = eval_data_dir / test_set
        dataset = FoldSERDataset(data_path)
        utt2emo = read_kaldi_format(data_path / "utt2emo")
        for spkfold, fold in read_kaldi_format(data_path / "spk2fold").items():
            if fold not in classifiers:
                model_dir = models_path / f"fold_{fold}"
                ckpt_dir = scan_checkpoint(model_dir, 'CKPT')
                if ckpt_dir is None:
                    raise FileNotFoundError(
                        f"No SER checkpoint found in {model_dir}. "
                        "Train the SER model first."
                    )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Use ckpt_dir (CKPT+1): its hyperparams.yaml has pretrainer.
                    # model_dir (fold_X) has training hyperparams without pretrainer;
                    # fetch prefers savedir/hyperparams.yaml when it exists.
                    classifiers[fold] = foreign_class(
                        source=ckpt_dir,
                        savedir=ckpt_dir,
                        run_opts={'device': device},
                        classname="CustomEncoderWav2vec2Classifier",
                        pymodule_file="custom_interface.py",
                    )
                classifiers[fold].hparams.label_encoder.ignore_len()
            hyp, ref, per_emo = [], [], {}
            for uttid, spkid, wav, wav_len, sr in tqdm.tqdm(dataset):
                if spkid != spkfold:
                    continue
                out_prob, score, index, text_lab = classifiers[fold].classify_batch(wav)
                lab2ind = classifiers[fold].hparams.label_encoder.lab2ind
                hyp.append(lab2ind[text_lab[0]])
                ref.append(lab2ind[utt2emo[uttid]])
                if utt2emo[uttid] not in per_emo:
                    per_emo[utt2emo[uttid]] = {"hyp": [], "ref": []}
                per_emo[utt2emo[uttid]]["hyp"].append([lab2ind[text_lab[0]]])
                per_emo[utt2emo[uttid]]["ref"].append([lab2ind[utt2emo[uttid]]])
            if not ref:
                continue
            uar = round(recall_score(y_true=ref, y_pred=hyp, average="macro") * 100, 3)
            score_per_emo = {f"ACC_{k}": round(accuracy_score(y_true=v["ref"], y_pred=v["hyp"]) * 100, 3) for k, v in per_emo.items()}
            test_set_info = test_set.split('_') if '_' in test_set else [test_set, "_"]
            results.append({'dataset': test_set_info[0], 'split': test_set_info[1], 'fold': fold,
                           'ser': 'anon' if anon_data_suffix in test_set else 'original', 'UAR': uar, **score_per_emo})
            print(f'{test_set} fold: {fold} - UAR: {uar}')
    return results


def _eval_ser_emotion2vec(eval_datasets, eval_data_dir, models_path, anon_data_suffix, params, device):
    from .emotion2vec_ser import Emotion2vecSERClassifier, LAB2IND
    model_id = params.get('model_id', 'emotion2vec/emotion2vec_plus_large')
    hub = params.get('hub', 'hf')
    classifier = Emotion2vecSERClassifier(model_id=model_id, hub=hub, device=device)
    test_sets = eval_datasets + [f'{dataset}{anon_data_suffix}' for dataset in eval_datasets]
    results = []
    for test_set in tqdm.tqdm(test_sets):
        data_path = eval_data_dir / test_set
        wav_scp = read_kaldi_format(data_path / "wav.scp")
        utt2emo = read_kaldi_format(data_path / "utt2emo")
        hyp, ref, per_emo = [], [], {}
        for uttid, wav_path in tqdm.tqdm(wav_scp.items()):
            ref_lab = utt2emo.get(uttid)
            if ref_lab not in LAB2IND:
                continue
            out = classifier.classify_file(wav_path)
            if out is None:
                continue
            _, score, pred_idx, text_lab = out
            pred = text_lab[0]
            hyp.append(LAB2IND[pred])
            ref.append(LAB2IND[ref_lab])
            if ref_lab not in per_emo:
                per_emo[ref_lab] = {"hyp": [], "ref": []}
            per_emo[ref_lab]["hyp"].append([LAB2IND[pred]])
            per_emo[ref_lab]["ref"].append([LAB2IND[ref_lab]])
        if not ref:
            continue
        uar = round(recall_score(y_true=ref, y_pred=hyp, average="macro") * 100, 3)
        score_per_emo = {f"ACC_{k}": round(accuracy_score(y_true=v["ref"], y_pred=v["hyp"]) * 100, 3) for k, v in per_emo.items()}
        test_set_info = test_set.split('_') if '_' in test_set else [test_set, "_"]
        results.append({'dataset': test_set_info[0], 'split': test_set_info[1], 'fold': 1,
                       'ser': 'anon' if anon_data_suffix in test_set else 'original', 'UAR': uar, **score_per_emo})
        print(f'{test_set} (emotion2vec+ SER) - UAR: {uar}')
    return results


@torch.no_grad()
def evaluate_ser(eval_datasets, eval_data_dir, models_path, anon_data_suffix, params, device):
    results_dir = params['results_dir']
    backend = params.get('backend', 'speechbrain').lower()
    logger.info(f"Emotion recognition on {backend} backend")
    if backend == 'emotion2vec':
        results = _eval_ser_emotion2vec(eval_datasets, eval_data_dir, models_path, anon_data_suffix, params, device)
    else:
        results = _eval_ser_speechbrain(eval_datasets, eval_data_dir, models_path, anon_data_suffix, params, device)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results)
    print(results_df)
    agg_cols = {'UAR': ['mean']}
    acc_cols = [c for c in results_df.columns if c.startswith('ACC_')]
    agg_cols.update({c: ['mean'] for c in acc_cols})
    result_mean = results_df.groupby(['dataset', 'split', 'ser']).agg(agg_cols)
    result_mean.reset_index(inplace=True)
    print(result_mean)

    results_df.to_csv(results_dir / f'results_folds{anon_data_suffix}.csv')
    result_mean.to_csv(results_dir / f'results{anon_data_suffix}.csv')
    return result_mean
