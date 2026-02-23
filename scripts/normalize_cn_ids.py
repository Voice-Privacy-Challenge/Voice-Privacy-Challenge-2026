#!/usr/bin/env python3
"""
Normalize utterance IDs in data/cn_* directories: remove .wav suffix
to match wav.scp format (utt2spk, trials, spk2utt, text, utt2dur use .wav; wav.scp does not).
"""
from pathlib import Path


def strip_wav(s):
    """Strip .wav from end if present."""
    return s[:-4] if s.endswith('.wav') else s


def normalize_file(filepath, col_indices, inplace=True):
    """Normalize specified columns in a space-separated file. col_indices is 0-based."""
    if not filepath.exists():
        return
    lines = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > max(col_indices):
                for i in col_indices:
                    if i < len(parts):
                        parts[i] = strip_wav(parts[i])
            lines.append(' '.join(parts) + '\n')
    if inplace:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print(f"  Normalized {filepath}")


def normalize_dir(datadir):
    """Normalize all relevant files in a data directory."""
    datadir = Path(datadir)
    if not datadir.is_dir():
        return

    # utt2spk: first column (utt_id)
    normalize_file(datadir / 'utt2spk', [0])

    # spk2utt: all columns after first (utt_ids)
    if (datadir / 'spk2utt').exists():
        lines = []
        with open(datadir / 'spk2utt', 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    parts = [parts[0]] + [strip_wav(p) for p in parts[1:]]
                lines.append(' '.join(parts) + '\n')
        with open(datadir / 'spk2utt', 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print(f"  Normalized {datadir / 'spk2utt'}")

    # text: first column
    normalize_file(datadir / 'text', [0])

    # utt2dur: first column
    normalize_file(datadir / 'utt2dur', [0])

    # trials: second column (test_utt_id)
    normalize_file(datadir / 'trials', [1])


def main():
    data_root = Path('/app/tem/data/')
    for d in sorted(data_root.glob('cn_*')):
        if d.is_dir():
            print(f"Normalizing {d.name}...")
            normalize_dir(d)
    print("Done.")


if __name__ == '__main__':
    main()
