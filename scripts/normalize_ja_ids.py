#!/usr/bin/env python3
"""
Normalize identifiers in data/ja_* directories for JVS Japanese corpus:
  - Unicode NFKC normalization (full-width ↔ half-width, compatibility chars)
  - Strip .wav suffix if present to match wav.scp format
Ensures speaker/utterance IDs match across utt2spk, trials, embeddings, etc.

After running, clear cached ASV speaker embeddings (or set force_compute_extraction)
so embeddings are re-extracted with normalized IDs.
"""
import unicodedata
from pathlib import Path


def normalize_id(s: str) -> str:
    """NFKC normalization + optional .wav strip."""
    s = unicodedata.normalize('NFKC', s)
    return s[:-4] if s.endswith('.wav') else s


def normalize_file(filepath: Path, col_indices: list[int], inplace: bool = True) -> None:
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
                        parts[i] = normalize_id(parts[i])
            lines.append(' '.join(parts) + '\n')
    if inplace:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print(f"  Normalized {filepath}")


def normalize_dir(datadir: Path) -> None:
    """Normalize all relevant files in a data directory."""
    if not datadir.is_dir():
        return

    # utt2spk: col 0 (utt_id), col 1 (spk_id) - normalize both
    normalize_file(datadir / 'utt2spk', [0, 1])

    # spk2utt: col 0 (spk_id), cols 1+ (utt_ids)
    if (datadir / 'spk2utt').exists():
        lines = []
        with open(datadir / 'spk2utt', 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    parts = [normalize_id(parts[0])] + [normalize_id(p) for p in parts[1:]]
                lines.append(' '.join(parts) + '\n')
        with open(datadir / 'spk2utt', 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print(f"  Normalized {datadir / 'spk2utt'}")

    # spk2gender: col 0 (spk_id)
    normalize_file(datadir / 'spk2gender', [0])

    # text: col 0 (utt_id)
    normalize_file(datadir / 'text', [0])

    # utt2dur: col 0 (utt_id)
    normalize_file(datadir / 'utt2dur', [0])

    # wav.scp: col 0 (utt_id) - keys must match utt2spk
    normalize_file(datadir / 'wav.scp', [0])

    # trials: col 0 (enroll spk_id), col 1 (test utt_id)
    normalize_file(datadir / 'trials', [0, 1])

    # enrolls: utterance IDs
    if (datadir / 'enrolls').exists():
        lines = []
        with open(datadir / 'enrolls', 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(normalize_id(line) + '\n')
                else:
                    lines.append('\n')
        with open(datadir / 'enrolls', 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print(f"  Normalized {datadir / 'enrolls'}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Normalize identifiers in ja_* data directories.')
    parser.add_argument('data_root', nargs='?', default='/app/tem/data',
                        help='Root directory containing ja_* subdirs (default: /app/tem/data)')
    args = parser.parse_args()
    data_root = Path(args.data_root)

    if not data_root.exists():
        print(f"Data root {data_root} does not exist.")
        return

    for d in sorted(data_root.glob('ja_*')):
        if d.is_dir():
            print(f"Normalizing {d.name}...")
            normalize_dir(d)
    print("Done.")


if __name__ == '__main__':
    main()
