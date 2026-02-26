#!/usr/bin/env python3
"""Create Kaldi-ready MLS German data, enroll/trial lists and subset directories."""
from __future__ import annotations

import argparse
import math
import random
from collections import Counter
from pathlib import Path

try:
    import soundfile as sf
except ModuleNotFoundError as exc:
    raise SystemExit("Install soundfile (pip install soundfile) to run this script.") from exc


def _load_transcripts(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            content = line.strip()
            if not content:
                continue
            utt, text = content.split("\t", 1)
            data[utt] = text
    return data


def _load_segments(path: Path) -> dict[str, tuple[float, float]]:
    data: dict[str, tuple[float, float]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            utt, _, start, end = parts
            data[utt] = (float(start), float(end))
    return data


def _write_file(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _get_relative_path_from_corpora(wav_path: Path | str) -> str:
    """Convert absolute path to relative path starting from 'corpora'."""
    wav_path_str = str(wav_path)
    # Find 'corpora' in the path (handle both '/corpora/' and 'corpora/')
    corpora_idx = wav_path_str.find('/corpora/')
    if corpora_idx != -1:
        # Found '/corpora/', return from 'corpora' onwards
        return wav_path_str[corpora_idx + 1:]
    
    corpora_idx = wav_path_str.find('corpora/')
    if corpora_idx != -1:
        # Found 'corpora/' (at beginning or elsewhere), return from 'corpora' onwards
        return wav_path_str[corpora_idx:]
    
    # If 'corpora' not found, return as-is (might already be relative)
    return wav_path_str


def _detect_available_partitions(path: Path) -> set[str]:
    """Detect which partition names actually exist in metainfo.txt"""
    available = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.lower().startswith("speaker"):
                continue
            cols = [col.strip() for col in stripped.split("|")]
            if len(cols) < 3:
                continue
            partition = cols[2].lower()
            available.add(partition)
    return available


def _map_partitions(requested: set[str], available: set[str]) -> set[str]:
    """Map requested partition names to actual names in metainfo.txt"""
    mapped = set()
    for req in requested:
        req_lower = req.lower()
        if req_lower in available:
            mapped.add(req_lower)
        elif req_lower == "dev" and "valid" in available:
            # Map "dev" to "valid" if "dev" doesn't exist but "valid" does
            mapped.add("valid")
        elif req_lower == "valid" and "dev" in available:
            # Map "valid" to "dev" if "valid" doesn't exist but "dev" does
            mapped.add("dev")
        else:
            mapped.add(req_lower)  # Keep original, will be filtered out if not found
    return mapped


def _load_speaker_genders(path: Path, partitions: set[str]) -> dict[str, str]:
    genders: dict[str, str] = {}
    available_partitions = _detect_available_partitions(path)
    partitions = _map_partitions(partitions, available_partitions)
    partitions = {p.lower() for p in partitions}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.lower().startswith("speaker"):
                continue
            cols = [col.strip() for col in stripped.split("|")]
            if len(cols) < 3:
                continue
            speaker, gender, partition = cols[0], cols[1], cols[2]
            if partition.lower() not in partitions:
                continue
            genders[speaker] = gender
    return genders


def _build_kaldi_maps(
    transcripts: dict[str, str],
    subset_dir: Path,
) -> tuple[dict[str, str], dict[str, float], dict[str, str], dict[str, str], dict[str, list[str]]]:
    """Build Kaldi maps and return dictionaries directly."""
    text: dict[str, str] = {}
    wav: dict[str, str] = {}
    utt2spk: dict[str, str] = {}
    utt2dur: dict[str, float] = {}
    spk2utt: dict[str, list[str]] = {}
    audio_root = subset_dir / "audio"

    for utt in sorted(transcripts):
        parts = utt.split("_")
        if len(parts) < 3:
            raise ValueError(f"Unexpected utterance id: {utt}")
        speaker, book = parts[0], parts[1]
        wav_path = audio_root / speaker / book / f"{utt}.wav"
        if not wav_path.exists():
            raise FileNotFoundError(f"{wav_path} missing for {utt}")
        with sf.SoundFile(wav_path) as fh:
            duration = len(fh) / fh.samplerate

        text[utt] = transcripts[utt]
        wav[utt] = _get_relative_path_from_corpora(wav_path.resolve())
        utt2spk[utt] = speaker
        utt2dur[utt] = duration
        spk2utt.setdefault(speaker, []).append(utt)

    return text, utt2dur, wav, utt2spk, spk2utt


def _write_maps(
    target_dir: Path,
    text: dict[str, str],
    utt2dur: dict[str, float],
    wav: dict[str, str],
    utt2spk: dict[str, str],
    spk2utt: dict[str, list[str]],
    spk2gender: dict[str, str],
) -> None:
    """Write Kaldi map files from dictionaries."""
    _write_file(target_dir / "text", [f"{utt} {text[utt]}" for utt in sorted(text)])
    _write_file(target_dir / "wav.scp", [f"{utt} {wav[utt]}" for utt in sorted(wav)])
    _write_file(target_dir / "utt2spk", [f"{utt} {utt2spk[utt]}" for utt in sorted(utt2spk)])
    _write_file(target_dir / "utt2dur", [f"{utt} {utt2dur[utt]:.3f}" for utt in sorted(utt2dur)])
    _write_file(target_dir / "spk2utt", [f"{spk} {' '.join(sorted(utts))}" for spk, utts in sorted(spk2utt.items())])
    _write_file(target_dir / "spk2gender", [f"{spk} {spk2gender[spk]}" for spk in sorted(spk2gender)])


def _sample_enrollment(
    targets: list[str],
    spk2utt: dict[str, list[str]],
    ratio: float,
    seed: int,
) -> dict[str, list[str]]:
    enroll: dict[str, list[str]] = {}
    rng = random.Random(seed)
    for spk in targets:
        utts = spk2utt.get(spk, [])
        count = max(1, min(len(utts), math.ceil(len(utts) * ratio)))
        if count >= len(utts):
            enrollments = list(utts)
        else:
            enrollments = rng.sample(utts, count)
        enroll[spk] = sorted(enrollments)
    return enroll


def _get_enrollment_lines(enroll_map: dict[str, list[str]]) -> list[str]:
    """Get flat list of enrollment utterance IDs."""
    return [utt for spk in sorted(enroll_map) for utt in enroll_map[spk]]


def _build_trials(
    targets: list[str],
    candidates: list[str],
    utt2spk: dict[str, str],
) -> tuple[list[str], Counter[str]]:
    counter: Counter[str] = Counter()
    lines: list[str] = []
    for spk in targets:
        for utt in candidates:
            label = "target" if utt2spk.get(utt) == spk else "nontarget"
            counter[label] += 1
            lines.append(f"{spk} {utt} {label}")
    return lines, counter


def _filter_maps(
    utts: set[str],
    utt2spk: dict[str, str],
    spk2gender: dict[str, str],
    spk2utt: dict[str, list[str]],
    text: dict[str, str],
    utt2dur: dict[str, float],
    wav: dict[str, str],
    out_dir: Path,
    trials_lines: list[str] | None,
    enroll_lines: list[str] | None,
) -> None:
    """Filter and write Kaldi maps for a subset of utterances."""
    filtered_utts = sorted(utts & set(text.keys()))
    selected_spks = {utt2spk[utt] for utt in filtered_utts}
    
    filtered_text = {utt: text[utt] for utt in filtered_utts}
    filtered_utt2spk = {utt: utt2spk[utt] for utt in filtered_utts}
    filtered_utt2dur = {utt: utt2dur[utt] for utt in filtered_utts}
    filtered_wav = {utt: wav[utt] for utt in filtered_utts}
    filtered_spk2utt = {
        spk: sorted([utt for utt in utts_list if utt in filtered_utts])
        for spk, utts_list in spk2utt.items()
        if spk in selected_spks
    }
    filtered_spk2utt = {spk: utts for spk, utts in filtered_spk2utt.items() if utts}
    filtered_spk2gender = {spk: spk2gender[spk] for spk in selected_spks if spk in spk2gender}

    _write_maps(out_dir, filtered_text, filtered_utt2dur, filtered_wav, filtered_utt2spk, filtered_spk2utt, filtered_spk2gender)
    
    if trials_lines is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_dir.joinpath("trials").write_text("\n".join(trials_lines) + "\n", encoding="utf-8")
    if enroll_lines is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_dir.joinpath("enrolls").write_text("\n".join(enroll_lines) + "\n", encoding="utf-8")


def _process_single_subset(
    subset_name: str,
    language_dir: Path,
    data_dir: Path,
    spk2gender_map: dict[str, str],
    lan: str,
    enroll_per_spk_ratio: float,
    enroll_seed: int,
    out_enroll_dir: Path | None,
    out_trials_f_dir: Path | None,
    out_trials_m_dir: Path | None,
    out_trials_mixed_dir: Path | None,
    skip_subsets: bool,
) -> None:
    """Process a single subset independently."""
    subset_dir = language_dir / subset_name
    transcripts = _load_transcripts(subset_dir / "transcripts.txt")
    segments = _load_segments(subset_dir / "segments.txt")
    missing = sorted(set(transcripts) - set(segments))
    if missing:
        raise SystemExit(f"Missing segments for utterances in {subset_name}: " + ", ".join(missing[:5]))

    text, utt2dur, wav, utt2spk, spk2utt = _build_kaldi_maps(transcripts, subset_dir)

    # Build spk2utt with sorted unique utterances
    merged_spk2utt = {spk: sorted(set(utts)) for spk, utts in spk2utt.items()}
    
    # Filter spk2gender to only include speakers in this subset
    filtered_spk2gender = {spk: spk2gender_map[spk] for spk in merged_spk2utt if spk in spk2gender_map}

    # Write Kaldi maps for this subset
    subset_data_dir = data_dir.parent / f"{lan}_{subset_name}"
    _write_maps(subset_data_dir, text, utt2dur, wav, utt2spk, merged_spk2utt, filtered_spk2gender)

    # Build enrollment and trials from this subset's data
    male_targets = [spk for spk, gender in spk2gender_map.items() if gender.upper() == "M" and spk in merged_spk2utt]
    female_targets = [spk for spk, gender in spk2gender_map.items() if gender.upper() == "F" and spk in merged_spk2utt]
    mixed_targets = male_targets + female_targets
    
    if not (0 < enroll_per_spk_ratio <= 1):
        raise SystemExit("--enroll-per-spk-ratio must be 0<ratio<=1 or percentage")

    enroll_map = _sample_enrollment(mixed_targets, merged_spk2utt, enroll_per_spk_ratio, enroll_seed)
    all_enroll_utts = {utt for utts in enroll_map.values() for utt in utts}
    enroll_lines = _get_enrollment_lines(enroll_map)
    
    # Determine output directories
    enroll_dir = out_enroll_dir or (data_dir.parent / f"{lan}_{subset_name}_enrolls")
    trial_f_dir = out_trials_f_dir or (data_dir.parent / f"{lan}_{subset_name}_trials_f")
    trial_m_dir = out_trials_m_dir or (data_dir.parent / f"{lan}_{subset_name}_trials_m")
    trial_mixed_dir = out_trials_mixed_dir or (data_dir.parent / f"{lan}_{subset_name}_trials_mixed")
    
    # Build trial candidates (exclude enrollment utterances)
    female_candidates = [utt for spk in female_targets for utt in merged_spk2utt.get(spk, []) if utt not in all_enroll_utts]
    male_candidates = [utt for spk in male_targets for utt in merged_spk2utt.get(spk, []) if utt not in all_enroll_utts]
    mixed_candidates = female_candidates + male_candidates

    female_trials, female_stats = _build_trials(female_targets, female_candidates, utt2spk)
    male_trials, male_stats = _build_trials(male_targets, male_candidates, utt2spk)
    mixed_trials, mixed_stats = _build_trials(mixed_targets, mixed_candidates, utt2spk)

    if not skip_subsets:
        _filter_maps(set(enroll_lines), utt2spk, spk2gender_map, merged_spk2utt, text, utt2dur, wav, enroll_dir, None, enroll_lines)
        _filter_maps(set(u.split()[1] for u in female_trials), utt2spk, spk2gender_map, merged_spk2utt, text, utt2dur, wav, trial_f_dir, female_trials, None)
        _filter_maps(set(u.split()[1] for u in male_trials), utt2spk, spk2gender_map, merged_spk2utt, text, utt2dur, wav, trial_m_dir, male_trials, None)
        _filter_maps(set(u.split()[1] for u in mixed_trials), utt2spk, spk2gender_map, merged_spk2utt, text, utt2dur, wav, trial_mixed_dir, mixed_trials, None)

    # Calculate per-speaker statistics
    all_targets = sorted(male_targets + female_targets)
    spk_stats = {}
    for spk in all_targets:
        total_utts = len(merged_spk2utt.get(spk, []))
        enroll_utts = len(enroll_map.get(spk, []))
        trial_utts = total_utts - enroll_utts
        
        # Count target and nontarget trials for this speaker
        target_count_sep = sum(1 for line in female_trials + male_trials 
                          if line.split()[0] == spk and line.split()[2] == "target")
        nontarget_count_sep = sum(1 for line in female_trials + male_trials 
                             if line.split()[0] == spk and line.split()[2] == "nontarget")
        target_count_mixed = sum(1 for line in mixed_trials 
                          if line.split()[0] == spk and line.split()[2] == "target")
        nontarget_count_mixed = sum(1 for line in mixed_trials 
                             if line.split()[0] == spk and line.split()[2] == "nontarget")
        
        spk_stats[spk] = {
            "gender": spk2gender_map.get(spk, "Unknown"),
            "total_utts": total_utts,
            "enroll_utts": enroll_utts,
            "trial_utts": trial_utts,
            "target_trials_sep": target_count_sep,
            "nontarget_trials_sep": nontarget_count_sep,
            "target_trials_mixed": target_count_mixed,
            "nontarget_trials_mixed": nontarget_count_mixed,
        }

    # Print summary statistics
    print(f"\n{'='*80}")
    print(f"Statistics for {lan} {subset_name}")
    print(f"{'='*80}\n")
    
    print("| Subset | Trials | Female | Male | Mixed | Separated Total | Speakers |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    print(
        f"| MLS German {subset_name} | Same-speaker | "
        f"{female_stats['target']} | {male_stats['target']} | {mixed_stats['target']} | "
        f"{female_stats['target'] + male_stats['target']} | "
        f"Female {len(female_targets)} / Male {len(male_targets)} / Mixed {len(mixed_targets)} |"
    )
    print(
        f"| MLS German {subset_name} | Different-speaker | "
        f"{female_stats['nontarget']} | {male_stats['nontarget']} | {mixed_stats['nontarget']} | "
        f"{female_stats['nontarget'] + male_stats['nontarget']} | "
        f"Female {len(female_targets)} / Male {len(male_targets)} / Mixed {len(mixed_targets)} |"
    )
    
    # Print overall statistics
    total_speakers = len(all_targets)
    total_utterances = sum(len(utts) for utts in merged_spk2utt.values())
    total_enroll_utts = len(all_enroll_utts)
    total_trial_utts = total_utterances - total_enroll_utts
    total_target_trials_sep = female_stats['target'] + male_stats['target']
    total_nontarget_trials_sep = female_stats['nontarget'] + male_stats['nontarget']
    
    print(f"\nOverall Statistics:")
    print(f"  Total speakers: {total_speakers} (Female: {len(female_targets)}, Male: {len(male_targets)})")
    print(f"  Total utterances: {total_utterances}")
    print(f"  Enrollment utterances: {total_enroll_utts} ({total_enroll_utts/total_utterances*100:.1f}%)")
    print(f"  Trial utterances: {total_trial_utts} ({total_trial_utts/total_utterances*100:.1f}%)")
    print(f"  Total separated trials: {total_target_trials_sep + total_nontarget_trials_sep}")
    print(f"    Separated Target trials: {total_target_trials_sep}")
    print(f"    Separated Nontarget trials: {total_nontarget_trials_sep}")
    print(f"  Total mixed trials: {mixed_stats['target'] + mixed_stats['nontarget']}")
    print(f"    Mixed Target trials: {mixed_stats['target']}")
    print(f"    Mixed Nontarget trials: {mixed_stats['nontarget']}")
    print(f"  Enrollment ratio: {enroll_per_spk_ratio*100:.1f}%")
    
    # Print per-speaker statistics
    print(f"\nPer-Speaker Statistics:")
    print(f"{'Speaker':<15} {'Gender':<8} {'Total':<8} {'Enroll':<8} {'Trial':<8} {'Sep_Target':<12} {'Mix_Target':<12} {'Sep_Nontrg':<12} {'Mix_Nontrg':<12}")
    print("-" * 110)
    for spk in sorted(all_targets):
        stats = spk_stats[spk]
        print(f"{spk:<15} {stats['gender']:<8} {stats['total_utts']:<8} "
              f"{stats['enroll_utts']:<8} {stats['trial_utts']:<8} "
              f"{stats['target_trials_sep']:<12} {stats['target_trials_mixed']:<12} "
              f"{stats['nontarget_trials_sep']:<12} {stats['nontarget_trials_mixed']:<12}")
    
    # Print gender-based statistics
    print(f"\nGender-based Statistics:")
    for gender, gender_label in [("F", "Female"), ("M", "Male")]:
        gender_speakers = [spk for spk in all_targets if spk2gender_map.get(spk, "").upper() == gender]
        if not gender_speakers:
            continue
        gender_total_utts = sum(spk_stats[spk]['total_utts'] for spk in gender_speakers)
        gender_enroll_utts = sum(spk_stats[spk]['enroll_utts'] for spk in gender_speakers)
        gender_trial_utts = sum(spk_stats[spk]['trial_utts'] for spk in gender_speakers)
        gender_target_trials = sum(spk_stats[spk]['target_trials_sep'] for spk in gender_speakers)
        gender_nontarget_trials_sep = sum(spk_stats[spk]['nontarget_trials_sep'] for spk in gender_speakers)
        gender_nontarget_trials_mixed = sum(spk_stats[spk]['nontarget_trials_mixed'] for spk in gender_speakers)
        
        print(f"  {gender_label}:")
        print(f"    Speakers: {len(gender_speakers)}")
        print(f"    Total utterances: {gender_total_utts}")
        print(f"    Enrollment utterances: {gender_enroll_utts}")
        print(f"    Trial utterances: {gender_trial_utts}")
        print(f"    Target trials: {gender_target_trials}")
        print(f"    Separated Nontarget trials: {gender_nontarget_trials_sep}")
        print(f"    Mixed Nontarget trials: {gender_nontarget_trials_mixed}")
    
    # Print utterances for each speaker
    print(f"\n{'='*80}")
    print(f"UTTERANCES FOR EACH SPEAKER ({lan} {subset_name})")
    print(f"{'='*80}")
    for spk in sorted(all_targets):
        gender = spk2gender_map.get(spk, "Unknown")
        all_utts = merged_spk2utt.get(spk, [])
        enroll_utts = set(enroll_map.get(spk, []))
        trial_utts = [utt for utt in all_utts if utt not in enroll_utts]
        
        print(f"\n{spk} ({gender}) - {len(all_utts)} total utterances:")
        if enroll_utts:
            print(f"  Enrollment ({len(enroll_utts)} utterances):")
        if trial_utts:
            print(f"  Trial ({len(trial_utts)} utterances):")

    
    print(f"\n{'='*80}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--language-dir",
        type=Path,
        default=Path("Voice-Privacy-Challenge-2024/corpora/de"),
        help="Path under corpora/ for the language (e.g. de).",
    )
    parser.add_argument(
        "--subset-name",
        type=str,
        default=None,
        help="Subset folder name under the language directory. If not specified, processes dev and test subsets separately (excluding train).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("Voice-Privacy-Challenge-2024/data/mls_german"),
    )
    parser.add_argument(
        "--out-enroll-dir",
        type=Path,
        default=None,
        help="Directory for filtered enrollment subset.",
    )
    parser.add_argument(
        "--out-trials-f-dir",
        type=Path,
        default=None,
        help="Directory for filtered female trials subset.",
    )
    parser.add_argument(
        "--out-trials-m-dir",
        type=Path,
        default=None,
        help="Directory for filtered male trials subset.",
    )
    parser.add_argument(
        "--out-trials-mixed-dir",
        type=Path,
        default=None,
        help="Directory for filtered mixed trials subset.",
    )
    parser.add_argument(
        "--enroll-per-spk-ratio",
        type=str,
        default="15%",
        help="Share of each speaker's utterances to use for enrollment (15% by default).",
    )
    parser.add_argument(
        "--enroll-seed",
        type=int,
        default=123,
        help="Random seed used to pick enrollment utterances.",
    )
    parser.add_argument(
        "--partitions",
        type=str,
        default=None,
        help="Comma-separated partitions from metainfo.txt to include. If not specified, will auto-detect based on subsets being processed.",
    )
    parser.add_argument(
        "--skip-subsets",
        action="store_true",
        help="Skip materializing enroll/trial subset directories.",
    )
    args = parser.parse_args()

    # Determine which subsets to process
    if args.subset_name:
        subsets_to_process = [args.subset_name]
    else:
        # Auto-detect available subsets (only dev and test, excluding train)
        available_subsets = ["dev", "test"]
        subsets_to_process = [
            s for s in available_subsets
            if (args.language_dir / s / "transcripts.txt").exists()
        ]
        if not subsets_to_process:
            raise SystemExit(f"No valid subsets (dev/test) found in {args.language_dir}")

    # Auto-determine partitions if not specified
    if args.partitions is None:
        # Use the same partitions as the subsets being processed
        partitions = set(subsets_to_process)
    else:
        partitions = {p.strip() for p in args.partitions.split(",")}
    
    # Load speaker genders with automatic partition name mapping
    spk2gender_map = _load_speaker_genders(args.language_dir / "metainfo.txt", partitions)
    lan = args.language_dir.name

    # Parse enrollment ratio
    ratio_raw = args.enroll_per_spk_ratio.strip()
    percent = ratio_raw.endswith("%")
    ratio_value = float(ratio_raw[:-1]) / 100.0 if percent else float(ratio_raw)
    if not (0 < ratio_value <= 1):
        raise SystemExit("--enroll-per-spk-ratio must be 0<ratio<=1 or percentage")

    # Process each subset separately
    for subset_name in subsets_to_process:
        _process_single_subset(
            subset_name,
            args.language_dir,
            args.data_dir,
            spk2gender_map,
            lan,
            ratio_value,
            args.enroll_seed,
            args.out_enroll_dir,
            args.out_trials_f_dir,
            args.out_trials_m_dir,
            args.out_trials_mixed_dir,
            args.skip_subsets,
        )


if __name__ == "__main__":
    main()

