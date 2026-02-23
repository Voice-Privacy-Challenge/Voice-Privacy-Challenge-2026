#!/usr/bin/env python3
"""
Normalize ja_dev_enrolls_nac IDs to match ja_dev_enrolls format.
Strip speaker prefix (jvsXXX_) from utterance IDs: jvs003_BASIC5000_0440 -> BASIC5000_0440
"""
from pathlib import Path


def strip_speaker_prefix(s):
    """Strip jvsXXX_ prefix from utterance ID."""
    if '_' in s and s.split('_', 1)[0].startswith('jvs'):
        return s.split('_', 1)[1]
    return s


def normalize_file(filepath, col_indices):
    """Normalize specified columns (0-based) by stripping speaker prefix."""
    if not filepath.exists():
        return
    lines = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > max(col_indices):
                for i in col_indices:
                    if i < len(parts):
                        parts[i] = strip_speaker_prefix(parts[i])
            lines.append(' '.join(parts) + '\n')
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f"  Normalized {filepath}")


def normalize_dir(datadir):
    """Normalize all relevant files to match ja_dev_enrolls format."""
    datadir = Path(datadir)
    if not datadir.is_dir():
        return

    normalize_file(datadir / 'utt2spk', [0])

    if (datadir / 'spk2utt').exists():
        lines = []
        with open(datadir / 'spk2utt', 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    parts = [parts[0]] + [strip_speaker_prefix(p) for p in parts[1:]]
                lines.append(' '.join(parts) + '\n')
        with open(datadir / 'spk2utt', 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print(f"  Normalized {datadir / 'spk2utt'}")

    normalize_file(datadir / 'text', [0])
    normalize_file(datadir / 'utt2dur', [0])
    normalize_file(datadir / 'enrolls', [0])


def main():
    data_root = Path(__file__).resolve().parent.parent / 'data'
    # ja_dev_enrolls_nac is the one with speaker-prefixed IDs
    for d in ['ja_dev_enrolls_nac', 'ja_test_enrolls_nac']:
        datadir = data_root / d
        if datadir.exists():
            print(f"Normalizing {d}...")
            normalize_dir(datadir)
    print("Done.")


if __name__ == '__main__':
    main()
