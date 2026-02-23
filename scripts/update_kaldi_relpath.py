#!/usr/bin/env python3
"""
Update Kaldi wav.scp to use relative paths: corpora/train_{lang}/wav/{utt_id}.wav
Also refresh utt2spk, spk2utt based on current folder structure.
"""
import argparse
import re
from pathlib import Path

try:
    from utils import save_kaldi_format
except ImportError:
    def save_kaldi_format(data, filename):
        with open(filename, "w", encoding="utf-8") as f:
            for k, v in sorted(data.items(), key=lambda x: x[0]):
                val = " ".join(v) if isinstance(v, list) else v
                f.write(f"{k} {val}\n")


def utt2spk_from_stem(dir_name: str, stem: str) -> str:
    """Infer speaker from utt stem. dir_name e.g. train_chinese."""
    lang = dir_name.replace("train_", "")
    parts = stem.split("_")
    if lang == "english":
        return parts[0]
    if lang in ("french", "german", "spanish"):
        return parts[0]
    if lang == "chinese":
        m = re.search(r"S\d{4}", stem)
        return m.group(0) if m else parts[0]
    if lang == "japanese":
        if len(parts) >= 3:
            return "_".join(parts[:2])
        return parts[0]
    return parts[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data/train_sampled"))
    parser.add_argument("--path-prefix", default="corpora", help="e.g. corpora -> corpora/train_english/wav/utt.wav")
    args = parser.parse_args()

    for lang_dir in sorted(args.data_dir.iterdir()):
        if not lang_dir.is_dir() or not lang_dir.name.startswith("train_"):
            continue
        dir_name = lang_dir.name  # e.g. train_chinese
        wav_dir = lang_dir / "wav"
        if not wav_dir.exists():
            print(f"Skipping {dir_name}: no wav/")
            continue

        wav_files = list(wav_dir.glob("*.wav"))
        if not wav_files:
            continue

        wav_scp = {}
        utt2spk = {}
        spk2utt = {}
        rel_path = f"{args.path_prefix}/{dir_name}/wav"

        for f in wav_files:
            utt_id = f.stem
            wav_scp[utt_id] = f"{rel_path}/{f.name}"
            spk = utt2spk_from_stem(dir_name, utt_id)
            utt2spk[utt_id] = spk
            spk2utt.setdefault(spk, []).append(utt_id)

        for spk in spk2utt:
            spk2utt[spk] = sorted(spk2utt[spk])

        save_kaldi_format(wav_scp, lang_dir / "wav.scp")
        save_kaldi_format(utt2spk, lang_dir / "utt2spk")
        save_kaldi_format(spk2utt, lang_dir / "spk2utt")

        print(f"{dir_name}: {len(wav_scp)} utts, path prefix {rel_path}")


if __name__ == "__main__":
    main()
