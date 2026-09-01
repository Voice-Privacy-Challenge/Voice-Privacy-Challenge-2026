#!/usr/bin/env python3
"""Generate LibriSpeech mixed / f / m trials the same way as build_mls_data.py.

Labels come from utt2spk (never copied from old trial files):
    label = "target" if utt2spk[utt] == enroll_spk else "nontarget"

Trials are the full cartesian product of enroll speakers × trial utterances
(enrollment utterances are excluded from the trial pool). Mixed therefore
includes true cross-gender (m↔f) pairs.

Uses existing Kaldi maps under:
    data/libri_{partition}_enrolls/
    data/libri_{partition}_trials_mixed/
and optionally writes trials_f / trials_m (+ filtered maps).

Does not re-sample enrollments; the official enrolls file is kept as-is.
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path


def _read_map(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        key, val = line.split(maxsplit=1)
        out[key] = val
    return out


def _write_file(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _build_trials(
    targets: list[str],
    candidates: list[str],
    utt2spk: dict[str, str],
) -> tuple[list[str], Counter[str]]:
    """Same logic as build_mls_data._build_trials."""
    counter: Counter[str] = Counter()
    lines: list[str] = []
    for spk in targets:
        for utt in candidates:
            label = "target" if utt2spk.get(utt) == spk else "nontarget"
            counter[label] += 1
            lines.append(f"{spk} {utt} {label}")
    return lines, counter


def _spk2utt_from_utt2spk(utt2spk: dict[str, str]) -> dict[str, list[str]]:
    spk2utt: dict[str, list[str]] = {}
    for utt, spk in utt2spk.items():
        spk2utt.setdefault(spk, []).append(utt)
    return {spk: sorted(utts) for spk, utts in spk2utt.items()}


def _filter_and_write_trial_dir(
    out_dir: Path,
    trial_lines: list[str],
    utt2spk: dict[str, str],
    spk2gender: dict[str, str],
    text: dict[str, str],
    utt2dur: dict[str, str],
    wav: dict[str, str],
) -> None:
    """Write a trials_* dir with Kaldi maps for utterances appearing in trials."""
    utts = sorted({line.split()[1] for line in trial_lines})
    selected_spks = {utt2spk[u] for u in utts if u in utt2spk}
    spk2utt = _spk2utt_from_utt2spk({u: utt2spk[u] for u in utts if u in utt2spk})

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_file(out_dir / "trials", trial_lines)
    _write_file(out_dir / "utt2spk", [f"{u} {utt2spk[u]}" for u in utts if u in utt2spk])
    _write_file(out_dir / "spk2utt", [f"{s} {' '.join(us)}" for s, us in sorted(spk2utt.items())])
    _write_file(
        out_dir / "spk2gender",
        [f"{s} {spk2gender[s]}" for s in sorted(selected_spks) if s in spk2gender],
    )
    if text:
        _write_file(out_dir / "text", [f"{u} {text[u]}" for u in utts if u in text])
    if utt2dur:
        _write_file(out_dir / "utt2dur", [f"{u} {utt2dur[u]}" for u in utts if u in utt2dur])
    if wav:
        _write_file(out_dir / "wav.scp", [f"{u} {wav[u]}" for u in utts if u in wav])


def process_partition(data_dir: Path, partition: str, write_fm: bool, backup: bool) -> None:
    enroll_dir = data_dir / f"libri_{partition}_enrolls"
    mixed_dir = data_dir / f"libri_{partition}_trials_mixed"
    trials_f_dir = data_dir / f"libri_{partition}_trials_f"
    trials_m_dir = data_dir / f"libri_{partition}_trials_m"

    for required in (enroll_dir / "enrolls", enroll_dir / "spk2gender", mixed_dir / "utt2spk", mixed_dir / "spk2gender"):
        if not required.is_file():
            raise FileNotFoundError(required)

    enroll_utts = [line.strip() for line in (enroll_dir / "enrolls").read_text().splitlines() if line.strip()]
    enroll_utt_set = set(enroll_utts)

    # Prefer enroll utt2spk when available; else parse speaker from utt id.
    if (enroll_dir / "utt2spk").is_file():
        enroll_utt2spk = _read_map(enroll_dir / "utt2spk")
        enroll_spks = sorted({enroll_utt2spk[u] for u in enroll_utts if u in enroll_utt2spk})
    else:
        enroll_spks = sorted({u.split("-")[0] for u in enroll_utts})

    trial_utt2spk = _read_map(mixed_dir / "utt2spk")
    spk2gender = {k: v.lower() for k, v in _read_map(mixed_dir / "spk2gender").items()}
    spk2gender.update({k: v.lower() for k, v in _read_map(enroll_dir / "spk2gender").items()})

    text = _read_map(mixed_dir / "text") if (mixed_dir / "text").is_file() else {}
    utt2dur = _read_map(mixed_dir / "utt2dur") if (mixed_dir / "utt2dur").is_file() else {}
    wav = _read_map(mixed_dir / "wav.scp") if (mixed_dir / "wav.scp").is_file() else {}

    # Trial pool = all mixed utts except enrollment utterances (MLS convention).
    candidates = sorted(u for u in trial_utt2spk if u not in enroll_utt_set)
    if not candidates:
        raise SystemExit(f"{partition}: no trial candidates after excluding enrolls")

    missing_gender = [s for s in enroll_spks if s not in spk2gender]
    if missing_gender:
        raise SystemExit(f"{partition}: missing spk2gender for enroll speakers: {missing_gender[:5]}")

    female_targets = [s for s in enroll_spks if spk2gender[s] == "f"]
    male_targets = [s for s in enroll_spks if spk2gender[s] == "m"]
    mixed_targets = male_targets + female_targets

    female_candidates = [u for u in candidates if spk2gender.get(trial_utt2spk[u], "?") == "f"]
    male_candidates = [u for u in candidates if spk2gender.get(trial_utt2spk[u], "?") == "m"]

    female_trials, female_stats = _build_trials(female_targets, female_candidates, trial_utt2spk)
    male_trials, male_stats = _build_trials(male_targets, male_candidates, trial_utt2spk)
    mixed_trials, mixed_stats = _build_trials(mixed_targets, candidates, trial_utt2spk)

    # Sanity: no same-speaker nontarget; has cross-gender.
    bad = 0
    cross = 0
    for line in mixed_trials:
        spk, utt, lab = line.split()
        us = trial_utt2spk[utt]
        if spk == us and lab != "target":
            bad += 1
        if spk2gender.get(spk) != spk2gender.get(us):
            cross += 1
    if bad:
        raise SystemExit(f"{partition}: internal error, {bad} same-spk nontarget lines")

    print(f"=== libri_{partition} ===")
    print(f"  enroll speakers: {len(enroll_spks)} (f={len(female_targets)}, m={len(male_targets)})")
    print(f"  enroll utts: {len(enroll_utts)}; trial candidates: {len(candidates)}")
    print(
        f"  trials_f: {len(female_trials)} "
        f"(target={female_stats['target']}, nontarget={female_stats['nontarget']})"
    )
    print(
        f"  trials_m: {len(male_trials)} "
        f"(target={male_stats['target']}, nontarget={male_stats['nontarget']})"
    )
    print(
        f"  trials_mixed: {len(mixed_trials)} "
        f"(target={mixed_stats['target']}, nontarget={mixed_stats['nontarget']}, "
        f"cross-gender={cross})"
    )

    trials_path = mixed_dir / "trials"
    if backup and trials_path.is_file():
        bak = mixed_dir / "trials_before_mls_style"
        if not bak.is_file():
            shutil.copy2(trials_path, bak)
            print(f"  backup: {bak.name}")

    _write_file(trials_path, mixed_trials)
    # Keep a clean copy name consistent with earlier fix workflow.
    _write_file(mixed_dir / "trials_correct", mixed_trials)
    print(f"  wrote {trials_path}")

    if write_fm:
        _filter_and_write_trial_dir(
            trials_f_dir, female_trials, trial_utt2spk, spk2gender, text, utt2dur, wav
        )
        _filter_and_write_trial_dir(
            trials_m_dir, male_trials, trial_utt2spk, spk2gender, text, utt2dur, wav
        )
        print(f"  wrote {trials_f_dir}/trials and {trials_m_dir}/trials")


def propagate_to_suffixes(data_dir: Path, partition: str) -> None:
    """Copy base mixed trials into libri_{partition}_trials_mixed_* anon dirs."""
    src = data_dir / f"libri_{partition}_trials_mixed" / "trials"
    if not src.is_file():
        raise FileNotFoundError(src)
    text = src.read_text(encoding="utf-8")
    n = 0
    for d in sorted(data_dir.glob(f"libri_{partition}_trials_mixed_*")):
        if not d.is_dir():
            continue
        dst = d / "trials"
        if not dst.is_file() and not (d / "utt2spk").is_file():
            continue
        bak = d / "trials_before_mls_style"
        if dst.is_file() and not bak.is_file():
            shutil.copy2(dst, bak)
        dst.write_text(text, encoding="utf-8")
        (d / "trials_correct").write_text(text, encoding="utf-8")
        n += 1
    print(f"  propagated trials -> {n} libri_{partition}_trials_mixed_* dirs")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild LibriSpeech trials with build_mls_data.py labeling logic."
    )
    parser.add_argument(
        "--partition",
        type=str,
        choices=["test", "dev", "both"],
        default="both",
        help="Partition to rebuild (default: both)",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--write-fm",
        action="store_true",
        default=True,
        help="Also write libri_*_trials_f and libri_*_trials_m (default: True)",
    )
    parser.add_argument("--no-write-fm", action="store_false", dest="write_fm")
    parser.add_argument(
        "--propagate-suffixes",
        action="store_true",
        default=True,
        help="Copy new mixed trials into anon-suffixed trials_mixed_* dirs (default: True)",
    )
    parser.add_argument("--no-propagate-suffixes", action="store_false", dest="propagate_suffixes")
    parser.add_argument("--no-backup", action="store_true", help="Do not backup existing trials")
    args = parser.parse_args()

    parts = ["dev", "test"] if args.partition == "both" else [args.partition]
    for part in parts:
        process_partition(args.data_dir, part, write_fm=args.write_fm, backup=not args.no_backup)
        if args.propagate_suffixes:
            propagate_to_suffixes(args.data_dir, part)


if __name__ == "__main__":
    main()
