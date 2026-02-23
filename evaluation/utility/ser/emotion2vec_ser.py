"""
emotion2vec+-based Speech Emotion Recognition.
Model: emotion2vec/emotion2vec_plus_large (via FunASR)
Maps 9 emotions to IEMOCAP 4-class: ang, hap, neu, sad
Requires: pip install -U funasr
"""
import numpy as np
from pathlib import Path

# emotion2vec 9-class: 0 angry, 1 disgusted, 2 fearful, 3 happy, 4 neutral, 5 other, 6 sad, 7 surprised, 8 unknown
EMOTION2VEC_IDS = ["angry", "disgusted", "fearful", "happy", "neutral", "other", "sad", "surprised", "unknown"]
EMOTION2VEC_TO_IEMOCAP = {
    "angry": "ang",
    "disgusted": "ang",
    "fearful": "sad",
    "happy": "hap",
    "neutral": "neu",
    "other": "neu",
    "sad": "sad",
    "surprised": "neu",
    "unknown": "neu",
}

IEMOCAP_LABELS = ["ang", "hap", "neu", "sad"]
LAB2IND = {l: i for i, l in enumerate(IEMOCAP_LABELS)}


class Emotion2vecSERClassifier:
    """Wrap emotion2vec+ large (FunASR) for compatibility with evaluate_ser."""

    def __init__(self, model_id="emotion2vec/emotion2vec_plus_large", hub="hf", device="cuda"):
        from funasr import AutoModel
        self.model_id = model_id
        self.model = AutoModel(
            model=model_id,
            hub=hub,  # "hf"/"huggingface" for overseas; "ms"/"modelscope" for China
        )

    def classify_file(self, wav_path):
        """
        wav_path: path to 16kHz wav file.
        Returns: (probs, score, index, [label]) compatible with SpeechBrain interface.
        """
        rec_result = self.model.generate(
            str(wav_path),
            output_dir=None,
            granularity="utterance",
            extract_embedding=False,
        )
        # rec_result: list of dicts with 'labels' and 'scores'
        if not rec_result:
            return None
        res = rec_result[0] if isinstance(rec_result, list) else rec_result
        labels = res.get("labels", [])
        scores = res.get("scores", [])
        if not scores:
            return None
        scores_arr = np.array(scores) if not isinstance(scores, np.ndarray) else scores
        pred_idx = int(np.argmax(scores_arr))
        pred_label_raw = labels[pred_idx] if pred_idx < len(labels) else pred_idx
        # Normalize: may be int 0-8, "0", "angry", or path "xxx/angry"
        if isinstance(pred_label_raw, int) and 0 <= pred_label_raw < len(EMOTION2VEC_IDS):
            pred_label_raw = EMOTION2VEC_IDS[pred_label_raw]
        elif isinstance(pred_label_raw, str):
            if pred_label_raw.isdigit():
                idx = int(pred_label_raw)
                pred_label_raw = EMOTION2VEC_IDS[idx] if idx < len(EMOTION2VEC_IDS) else "unknown"
            elif "/" in pred_label_raw:
                pred_label_raw = pred_label_raw.split("/")[-1]
            if pred_label_raw == "<unk>":
                pred_label_raw = "unknown"
            pred_label_raw = pred_label_raw.lower().strip()
        else:
            pred_label_raw = "unknown"
        iemocap_label = EMOTION2VEC_TO_IEMOCAP.get(pred_label_raw, "neu")
        score = float(scores_arr[pred_idx])
        # Build simple probs-like tensor for compatibility (not used by eval)
        return None, score, LAB2IND[iemocap_label], [iemocap_label]
