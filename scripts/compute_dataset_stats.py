#!/usr/bin/env python3
"""
Compute speaker and utterance statistics for train sets across languages.
Output format suitable for LaTeX table.

Data paths:
  - English:  /app/data/voiceprivacy2026
  - French:   /app/data/mls_french/train
  - German:   /app/data/mls_german/train
  - Spanish:  /app/data/mls_spanish/train
  - Japanese: /app/data/JTubeSpeech-ASV/train
  - Chinese:  /app/data/aishell/data_aishell/wav/train
"""
import argparse
from pathlib import Path
from collections import defaultdict

# Table 1 buckets: 1-100, 101-500, 501-1000, 1001-5000, 5001+
BUCKETS_TABLE1 = [
    (1, 100, "1--100"),
    (101, 500, "101--500"),
    (501, 1000, "501--1000"),
    (1001, 5000, "1001--5000"),
    (5001, float("inf"), "5001+"),
]

# Table 2 (MLS-en) buckets: 1-5, 5-10, 10-20, 20-50, 50-100
BUCKETS_TABLE2 = [
    (1, 5, "1--5"),
    (6, 10, "5--10"),
    (11, 20, "10--20"),
    (21, 50, "20--50"),
    (51, 100, "50--100"),
]


def _count_english(data_dir: Path) -> dict[str, int]:
    """English: voiceprivacy2026/audio/{spk}/{book}/file. Audio dir structure."""
    root = data_dir / "audio"
    spk2count = defaultdict(int)
    for spk_dir in root.iterdir():
        if spk_dir.is_dir():
            n = len(list(spk_dir.rglob("*.wav"))) + len(list(spk_dir.rglob("*.flac")))
            if n > 0:
                spk2count[spk_dir.name] = n
    return dict(spk2count)


def _count_mls(data_dir: Path) -> dict[str, int]:
    """MLS (French/German/Spanish): mls_xx/train/audio/{spk}/{book}/file.wav"""
    root = data_dir / "train" / "audio"
    if not root.exists():
        return {}
    spk2count = {}
    for spk_dir in root.iterdir():
        if spk_dir.is_dir():
            n = len(list(spk_dir.rglob("*.wav")))
            if n > 0:
                spk2count[spk_dir.name] = n
    return spk2count


def _count_japanese(data_dir: Path) -> dict[str, int]:
    """Japanese: JTubeSpeech-ASV/train/{channel}/file.wav, speaker in filename PREFIX_SPK_ID"""
    root = data_dir / "train"
    spk2count = defaultdict(int)
    for f in root.rglob("*.wav"):
        parts = f.stem.split("_")
        spk = "_".join(parts[:3]) if len(parts) >= 3 else f.parent.name
        spk2count[spk] += 1
    return dict(spk2count)


def _count_chinese(data_dir: Path) -> dict[str, int]:
    """Chinese: aishell/data_aishell/wav/train/{spk}/file.wav"""
    root = data_dir / "data_aishell" / "wav" / "train"
    if not root.exists():
        root = data_dir / "train"
    spk2count = {}
    for spk_dir in root.iterdir():
        if spk_dir.is_dir():
            n = len(list(spk_dir.rglob("*.wav")))
            if n > 0:
                spk2count[spk_dir.name] = n
    return spk2count


def compute_lang_stats(spk2count: dict[str, int], buckets: list) -> dict:
    """Compute stats from speaker->count mapping."""
    counts = list(spk2count.values())
    if not counts:
        return None
    total_speakers = len(counts)
    total_utterances = sum(counts)
    counts_sorted = sorted(counts)
    n = len(counts_sorted)
    median = (
        counts_sorted[n // 2]
        if n % 2
        else (counts_sorted[n // 2 - 1] + counts_sorted[n // 2]) / 2
    )
    avg = total_utterances / total_speakers

    dist = []
    for lo, hi, label in buckets:
        c = sum(1 for x in counts if lo <= x <= hi)
        pct = 100 * c / total_speakers
        dist.append((label, c, pct))

    return {
        "total_speakers": total_speakers,
        "total_utterances": total_utterances,
        "avg_per_speaker": avg,
        "median": int(median) if median == int(median) else median,
        "min": min(counts),
        "max": max(counts),
        "distribution": dist,
    }


def compute_mls_en_list_stats(lst_path: Path, buckets: list) -> dict | None:
    """Compute stats from .lst file (format: spkid_sessionid_audioid)."""
    spk2count = defaultdict(int)
    with open(lst_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            spkid = line.split("_")[0]
            spk2count[spkid] += 1
    return compute_lang_stats(dict(spk2count), buckets)


def main():
    parser = argparse.ArgumentParser(description="Compute dataset statistics for LaTeX tables")
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument("--table", choices=["1", "2", "all"], default="all")
    args = parser.parse_args()

    data_dir = args.data_dir

    if args.table in ("1", "all"):
        # Table 1: Cross-language
        configs = [
            ("English", data_dir / "voiceprivacy2026", _count_english),
            ("French", data_dir / "mls_french", _count_mls),
            ("German", data_dir / "mls_german", _count_mls),
            ("Spanish", data_dir / "mls_spanish", _count_mls),
            ("Japanese", data_dir / "JTubeSpeech-ASV", _count_japanese),
            ("Chinese", data_dir / "aishell", _count_chinese),
        ]

        print("\n" + "=" * 70)
        print("Table 1: Speaker and Utterance Statistics Across Languages (train-set)")
        print("=" * 70)

        for name, path, count_fn in configs:
            if not path.exists():
                print(f"Skipping {name}: {path} not found")
                continue
            spk2count = count_fn(path)
            s = compute_lang_stats(spk2count, BUCKETS_TABLE1)
            if s is None:
                print(f"{name}: no data")
                continue
            print(f"\n--- {name} ---")
            print(f"  Speakers:   {s['total_speakers']:,}")
            print(f"  Utterances: {s['total_utterances']:,}")
            print(f"  Avg/Speaker: {s['avg_per_speaker']:.1f}")
            print(f"  Median: {s['median']}, Min: {s['min']}, Max: {s['max']:,}")
            print("  Distribution:")
            for label, c, pct in s["distribution"]:
                print(f"    {label}: {c} ({pct:.1f}%)")

    if args.table in ("2", "all"):
        # Table 2: MLS-en by list
        lst_dir = data_dir / "voiceprivacy2026"
        lst_files = sorted(lst_dir.glob("data_*.lst"))
        lst_files = [f for f in lst_files if f.suffix == ".lst"]

        print("\n" + "=" * 70)
        print("Table 2: MLS-en Speaker and utterance statistics by list")
        print("=" * 70)

        for lst_path in lst_files:
            s = compute_mls_en_list_stats(lst_path, BUCKETS_TABLE2)
            if s is None:
                continue
            name = lst_path.stem
            print(f"\n--- {name} ---")
            print(f"  Speakers:   {s['total_speakers']:,}")
            print(f"  Utterances: {s['total_utterances']:,}")
            print(f"  Avg/Speaker: {s['avg_per_speaker']:.1f}")
            print("  Distribution:")
            for label, c, pct in s["distribution"]:
                print(f"    {label}: {c} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
