#!/usr/bin/env python3
"""Build Kaldi-style data + enrollment/trial subsets for the JVS Japanese corpus."""

from __future__ import annotations

import argparse
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

try:
    import soundfile as sf
except ModuleNotFoundError as exc:
    raise SystemExit("Install soundfile (pip install soundfile) to run this script.") from exc


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


def _load_transcripts(path: Path, speaker_id: str) -> dict[str, str]:
    """Load transcripts from JVS format (UTT_ID: text).
    
    Args:
        path: Path to transcripts_utf8.txt
        speaker_id: Speaker ID (e.g., jvs001) to prefix utterance IDs
    """
    transcripts: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            if ":" in stripped:
                utt_id_raw, text = stripped.split(":", 1)
                utt_id_raw = utt_id_raw.strip()
                # Create unique utterance ID: speaker_id_utt_id
                utt_id = f"{speaker_id}_{utt_id_raw}"
                transcripts[utt_id] = text.strip()
    return transcripts


def _map_audio_files(source_dir: Path, subset_name: str) -> dict[str, tuple[str, Path]]:
    """Map audio files from JVS structure.
    
    Structure: jvs_ver1/jvs001/{subset_name}/wav16kHz16bit/VOICEACTRESS100_001.wav
    Speaker ID is extracted from parent directory (jvs001).
    Utterance ID format: speaker_id_utt_id (e.g., jvs001_VOICEACTRESS100_001)
    
    Args:
        source_dir: Root directory containing jvs001/, jvs002/, etc.
        subset_name: Subset name (e.g., "parallel100" or "nonpara30")
    """
    lookup: dict[str, tuple[str, Path]] = {}
    # Find all wav16kHz16bit directories under jvs*/{subset_name}/
    pattern = f"jvs*/{subset_name}/wav16kHz16bit"
    for wav_dir in source_dir.glob(pattern):
        speaker_id = wav_dir.parent.parent.name  # e.g., jvs001
        for wav_path in wav_dir.glob("*.wav"):
            if wav_path.is_file():
                utt_id_raw = wav_path.stem  # e.g., VOICEACTRESS100_001
                # Create unique utterance ID: speaker_id_utt_id
                utt_id = f"{speaker_id}_{utt_id_raw}"
                lookup[utt_id] = (speaker_id, wav_path)
    return lookup


def _load_speaker_genders(path: Path) -> dict[str, str]:
    """Load speaker gender information from gender_f0range.txt."""
    genders: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.lower().startswith("speaker"):
                continue
            cols = stripped.split()
            if len(cols) < 2:
                continue
            speaker = cols[0]
            gender_raw = cols[1].upper()
            if gender_raw == "M":
                genders[speaker] = "M"
            elif gender_raw == "F":
                genders[speaker] = "F"
    return genders


def _build_kaldi_maps(
    transcripts: dict[str, str],
    audio_index: dict[str, tuple[str, Path]],
) -> tuple[
    list[str],
    list[str],
    list[str],
    list[str],
    list[str],
    dict[str, list[str]],
]:
    text_lines: list[str] = []
    wav_map: dict[str, str] = {}
    utt2spk: dict[str, str] = {}
    utt2dur: dict[str, float] = {}
    spk2utt: dict[str, list[str]] = {}

    missing_audio = []
    for utt in sorted(transcripts):
        entry = audio_index.get(utt)
        if entry is None:
            missing_audio.append(utt)
            continue  # Skip utterances without audio files
        speaker, wav_path = entry
        try:
            with sf.SoundFile(wav_path) as fh:
                duration = len(fh) / fh.samplerate
        except Exception as e:
            print(f"Warning: Could not read {wav_path}: {e}")
            missing_audio.append(utt)
            continue
        text_lines.append(f"{utt} {transcripts[utt]}")
        wav_map[utt] = _get_relative_path_from_corpora(wav_path.resolve())
        utt2spk[utt] = speaker
        utt2dur[utt] = duration
        spk2utt.setdefault(speaker, []).append(utt)
    
    if missing_audio:
        print(f"\nWarning: {len(missing_audio)} utterances in transcripts have no corresponding audio files")
        print(f"  Sample missing utterances: {missing_audio[:10]}")
        if len(missing_audio) > 10:
            print(f"  ... and {len(missing_audio) - 10} more")

    wav_lines = [f"{utt} {wav_map[utt]}" for utt in sorted(wav_map)]
    utt2spk_lines = [f"{utt} {utt2spk[utt]}" for utt in sorted(utt2spk)]
    spk2utt_lines = [f"{spk} {' '.join(sorted(utts))}" for spk, utts in sorted(spk2utt.items())]
    utt2dur_lines = [f"{utt} {utt2dur[utt]:.3f}" for utt in sorted(utt2dur)]
    return wav_lines, utt2spk_lines, utt2dur_lines, spk2utt_lines, text_lines, spk2utt


def _write_maps(
    target_dir: Path,
    text_lines: list[str],
    wav_lines: list[str],
    utt2spk_lines: list[str],
    utt2dur_lines: list[str],
    spk2utt_lines: list[str],
    spk2gender_lines: list[str],
) -> None:
    _write_file(target_dir / "text", text_lines)
    _write_file(target_dir / "wav.scp", wav_lines)
    _write_file(target_dir / "utt2spk", utt2spk_lines)
    _write_file(target_dir / "utt2dur", utt2dur_lines)
    _write_file(target_dir / "spk2utt", spk2utt_lines)
    _write_file(target_dir / "spk2gender", spk2gender_lines)


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


def _write_enrollments(path: Path, enroll_map: dict[str, list[str]]) -> list[str]:
    lines: list[str] = []
    for spk in sorted(enroll_map):
        lines.extend(enroll_map[spk])
    _write_file(path, lines)
    return lines


def _build_trials(
    targets: list[str],
    candidates: list[str],
    utt2spk: dict[str, str],
    max_trial_utts_per_speaker: int | None = None,
    seed: int = 123,
    upsample_if_insufficient: bool = False,
) -> tuple[list[str], Counter[str]]:
    """Build trials with optional balancing."""
    counter: Counter[str] = Counter()
    lines: list[str] = []
    
    # Group candidates by speaker for balancing
    spk_candidates: dict[str, list[str]] = {}
    for utt in candidates:
        spk = utt2spk.get(utt)
        if spk:
            spk_candidates.setdefault(spk, []).append(utt)
    
    # Balance candidates if requested
    if max_trial_utts_per_speaker is not None:
        rng = random.Random(seed)
        balanced_candidates = []
        # Sort speakers to ensure deterministic order
        for spk in sorted(spk_candidates.keys()):
            utts = spk_candidates[spk]
            if len(utts) > max_trial_utts_per_speaker:
                balanced_candidates.extend(rng.sample(utts, max_trial_utts_per_speaker))
            elif upsample_if_insufficient and len(utts) < max_trial_utts_per_speaker:
                num_needed = max_trial_utts_per_speaker - len(utts)
                additional = rng.choices(utts, k=num_needed)
                balanced_candidates.extend(utts + additional)
            else:
                balanced_candidates.extend(utts)
        candidates = balanced_candidates
    
    # Sort targets and candidates to ensure deterministic order
    sorted_targets = sorted(targets)
    sorted_candidates = sorted(candidates)
    for spk in sorted_targets:
        for utt in sorted_candidates:
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
    selected_spks = {utt2spk[utt] for utt in utts if utt in utt2spk}
    filtered_spk2utt = {spk: [utt for utt in utts_list if utt in utts] for spk, utts_list in spk2utt.items()}
    filtered_spk2utt = {spk: utts for spk, utts in filtered_spk2utt.items() if utts}
    filtered_text = {utt: text[utt] for utt in sorted(utts) if utt in text}
    filtered_utt2spk = {utt: utt2spk[utt] for utt in sorted(utts) if utt in utt2spk}
    filtered_utt2dur = {utt: f"{utt2dur[utt]:.3f}" for utt in sorted(utts) if utt in utt2dur}
    filtered_wav = {utt: wav[utt] for utt in sorted(utts) if utt in wav}
    filtered_spk2gender = {spk: spk2gender[spk] for spk in sorted(selected_spks) if spk in spk2gender}

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_file(out_dir / "spk2gender", [f"{spk} {filtered_spk2gender[spk]}" for spk in filtered_spk2gender])
    _write_file(out_dir / "spk2utt", [f"{spk} {' '.join(filtered_spk2utt[spk])}" for spk in filtered_spk2utt])
    _write_file(out_dir / "text", [f"{utt} {filtered_text[utt]}" for utt in filtered_text])
    _write_file(out_dir / "utt2spk", [f"{utt} {filtered_utt2spk[utt]}" for utt in filtered_utt2spk])
    _write_file(out_dir / "utt2dur", [f"{utt} {filtered_utt2dur[utt]}" for utt in filtered_utt2dur])
    _write_file(out_dir / "wav.scp", [f"{utt} {filtered_wav[utt]}" for utt in filtered_wav])
    if trials_lines is not None:
        (out_dir / "trials").write_text("\n".join(trials_lines) + "\n", encoding="utf-8")
    if enroll_lines is not None:
        (out_dir / "enrolls").write_text("\n".join(enroll_lines) + "\n", encoding="utf-8")


def _process_single_set(
    female_targets: list[str],
    male_targets: list[str],
    spk2utt: dict[str, list[str]],
    spk2gender_map: dict[str, str],
    text_lines: list[str],
    wav_lines: list[str],
    utt2spk_lines: list[str],
    utt2dur_lines: list[str],
    spk2utt_lines: list[str],
    set_name: str,
    output_prefix: str,
    data_dir: Path,
    enroll_per_spk_ratio: float,
    enroll_seed: int,
    max_trial_utts_per_speaker: int | None,
    trial_balance_seed: int,
    upsample_insufficient_speakers: bool,
    out_enroll_dir: Path | None,
    out_trials_f_dir: Path | None,
    out_trials_m_dir: Path | None,
    out_trials_mixed_dir: Path | None,
    skip_subsets: bool,
) -> None:
    """Process a single set (dev or test) and generate enroll/trials."""
    print(f"\n{'='*80}")
    print(f"PROCESSING {set_name.upper()} SET")
    print(f"{'='*80}")
    
    # Print selected speakers summary
    print(f"\n{'='*80}")
    print(f"{set_name.upper()} SET - SELECTED SPEAKERS")
    print(f"{'='*80}")
    print(f"Total speakers: {len(female_targets) + len(male_targets)} (Female: {len(female_targets)}, Male: {len(male_targets)})")
    
    # Print Female speakers separately
    print(f"\n{'='*80}")
    print(f"{set_name.upper()} SET - FEMALE SPEAKERS ({len(female_targets)})")
    print(f"{'='*80}")
    print(f"{'Rank':<6} {'Speaker ID':<15} {'Utterances':<12}")
    print("-" * 35)
    female_targets_sorted = sorted(female_targets, key=lambda spk: len(spk2utt.get(spk, [])), reverse=True)
    for rank, spk in enumerate(female_targets_sorted, 1):
        count = len(spk2utt.get(spk, []))
        print(f"{rank:<6} {spk:<15} {count:<12}")
    
    # Print Male speakers separately
    print(f"\n{'='*80}")
    print(f"{set_name.upper()} SET - MALE SPEAKERS ({len(male_targets)})")
    print(f"{'='*80}")
    print(f"{'Rank':<6} {'Speaker ID':<15} {'Utterances':<12}")
    print("-" * 35)
    male_targets_sorted = sorted(male_targets, key=lambda spk: len(spk2utt.get(spk, [])), reverse=True)
    for rank, spk in enumerate(male_targets_sorted, 1):
        count = len(spk2utt.get(spk, []))
        print(f"{rank:<6} {spk:<15} {count:<12}")
    
    enroll_map = _sample_enrollment(male_targets + female_targets, spk2utt, enroll_per_spk_ratio, enroll_seed)
    all_enroll_utts = {utt for utts in enroll_map.values() for utt in utts}
    enroll_file = data_dir / f"{output_prefix}_enrolls"
    trial_f_file = data_dir / f"{output_prefix}_trials_f"
    trial_m_file = data_dir / f"{output_prefix}_trials_m"
    enroll_lines = _write_enrollments(enroll_file, enroll_map)

    # Build candidate utterances (exclude enrollment)
    male_candidates_raw = [
        utt
        for spk in male_targets
        for utt in spk2utt.get(spk, [])
        if utt not in all_enroll_utts
    ]
    female_candidates_raw = [
        utt
        for spk in female_targets
        for utt in spk2utt.get(spk, [])
        if utt not in all_enroll_utts
    ]
    
    # Calculate trial utterance statistics per speaker for balancing
    def _get_trial_utt_counts(targets_list, candidates_list, spk2utt_dict, enroll_set):
        trial_counts = {}
        for spk in targets_list:
            trial_utts = [utt for utt in spk2utt_dict.get(spk, []) if utt not in enroll_set]
            trial_counts[spk] = len(trial_utts)
        return trial_counts
    
    female_trial_counts = _get_trial_utt_counts(female_targets, female_candidates_raw, spk2utt, all_enroll_utts)
    male_trial_counts = _get_trial_utt_counts(male_targets, male_candidates_raw, spk2utt, all_enroll_utts)
    
    # Auto-determine balance threshold if not specified - use median
    max_trial_utts = max_trial_utts_per_speaker
    if max_trial_utts is None:
        # Use median as default balancing threshold
        all_counts = list(female_trial_counts.values()) + list(male_trial_counts.values())
        if all_counts:
            max_trial_utts = int(statistics.median(all_counts))
            print(f"\nAuto-balancing: Using median trial utterances per speaker ({max_trial_utts})")
        else:
            print(f"\nWarning: No trial utterances found, cannot determine median")
    
    # Print imbalance statistics
    if max_trial_utts is not None:
        print(f"\nTrial Utterance Balance Statistics:")
        print(f"  Balance threshold: {max_trial_utts} utterances per speaker")
        if upsample_insufficient_speakers:
            print(f"  Upsampling: Speakers with fewer utterances will be upsampled (with replacement)")
        else:
            print(f"  No upsampling: Speakers with fewer utterances will keep all their utterances")
        for gender_label, counts_dict in [("Female", female_trial_counts), ("Male", male_trial_counts)]:
            if counts_dict:
                counts = list(counts_dict.values())
                max_count = max(counts)
                min_count = min(counts)
                median_count = int(statistics.median(counts)) if counts else 0
                speakers_above_threshold = sum(1 for c in counts if c > max_trial_utts)
                speakers_below_threshold = sum(1 for c in counts if c < max_trial_utts)
                print(f"  {gender_label}: min={min_count}, median={median_count}, max={max_count}, "
                      f"speakers above threshold={speakers_above_threshold}/{len(counts)}, "
                      f"speakers below threshold={speakers_below_threshold}/{len(counts)}")
    
    utt2spk_map = {utt: spk for spk, utts in spk2utt.items() for utt in utts}
    female_trials, female_stats = _build_trials(
        female_targets, female_candidates_raw, utt2spk_map,
        max_trial_utts_per_speaker=max_trial_utts,
        seed=trial_balance_seed,
        upsample_if_insufficient=upsample_insufficient_speakers,
    )
    male_trials, male_stats = _build_trials(
        male_targets, male_candidates_raw, utt2spk_map,
        max_trial_utts_per_speaker=max_trial_utts,
        seed=trial_balance_seed,
        upsample_if_insufficient=upsample_insufficient_speakers,
    )

    mixed_targets = female_targets + male_targets
    mixed_candidates_raw = female_candidates_raw + male_candidates_raw
    mixed_trials, mixed_stats = _build_trials(
        mixed_targets, mixed_candidates_raw, utt2spk_map,
        max_trial_utts_per_speaker=max_trial_utts,
        seed=trial_balance_seed,
        upsample_if_insufficient=upsample_insufficient_speakers,
    )

    trial_mixed_file = data_dir / f"{output_prefix}_trials_mixed"

    _write_file(trial_f_file, female_trials)
    _write_file(trial_m_file, male_trials)
    _write_file(trial_mixed_file, mixed_trials)

    # Calculate per-speaker statistics
    spk_stats = {}
    all_targets = sorted(female_targets + male_targets)
    
    for spk in all_targets:
        total_utts = len(spk2utt.get(spk, []))
        enroll_utts = len(enroll_map.get(spk, []))
        trial_utts = total_utts - enroll_utts
        spk_stats[spk] = {
            "gender": spk2gender_map.get(spk, "?"),
            "total_utts": total_utts,
            "enroll_utts": enroll_utts,
            "trial_utts": trial_utts,
            "target_trials_sep": 0,
            "nontarget_trials_sep": 0,
            "target_trials_mixed": 0,
            "nontarget_trials_mixed": 0,
        }
    
    # Parse female trials
    for line in female_trials:
        parts = line.split()
        if len(parts) >= 3:
            spk, utt, label = parts[0], parts[1], parts[2]
            if label == "target":
                spk_stats[spk]["target_trials_sep"] += 1
            elif label == "nontarget":
                spk_stats[spk]["nontarget_trials_sep"] += 1
    
    # Parse male trials
    for line in male_trials:
        parts = line.split()
        if len(parts) >= 3:
            spk, utt, label = parts[0], parts[1], parts[2]
            if label == "target":
                spk_stats[spk]["target_trials_sep"] += 1
            elif label == "nontarget":
                spk_stats[spk]["nontarget_trials_sep"] += 1

    # Parse mixed trials
    for line in mixed_trials:
        parts = line.split()
        if len(parts) >= 3:
            spk, utt, label = parts[0], parts[1], parts[2]
            if label == "target":
                spk_stats[spk]["target_trials_mixed"] += 1
            elif label == "nontarget":
                spk_stats[spk]["nontarget_trials_mixed"] += 1

    # Print per-speaker statistics (similar to MLS output format)
    print(f"\n{'='*80}")
    print(f"PER-SPEAKER STATISTICS ({set_name.upper()})")
    print(f"{'='*80}")
    print(f"{'Speaker':<15} {'Gender':<8} {'Total':<8} {'Enroll':<8} {'Trial':<8} {'Sep_Target':<12} {'Mix_Target':<12} {'Sep_Nontrg':<12} {'Mix_Nontrg':<12}")
    print("-" * 110)
    for spk in sorted(all_targets):
        stats = spk_stats[spk]
        print(f"{spk:<15} {stats['gender']:<8} {stats['total_utts']:<8} "
              f"{stats['enroll_utts']:<8} {stats['trial_utts']:<8} "
              f"{stats['target_trials_sep']:<12} {stats['target_trials_mixed']:<12} "
              f"{stats['nontarget_trials_sep']:<12} {stats['nontarget_trials_mixed']:<12}")

    total_female = len(female_targets)
    total_male = len(male_targets)
    total_mixed = len(mixed_targets)
    print(f"\n| Subset | Trials | Female | Male | Mixed | Separated Total | Speakers |")
    print(f"| --- | --- | --- | --- | --- | --- | --- |")
    print(
        f"| JVS {output_prefix} | Same-speaker | {female_stats['target']} | {male_stats['target']} | {mixed_stats['target']} | "
        f"{female_stats['target'] + male_stats['target']} | Female {total_female} / Male {total_male} / Mixed {total_mixed} |"
    )
    print(
        f"| JVS {output_prefix} | Different-speaker | {female_stats['nontarget']} | {male_stats['nontarget']} | {mixed_stats['nontarget']} | "
        f"{female_stats['nontarget'] + male_stats['nontarget']} | Female {total_female} / Male {total_male} / Mixed {total_mixed} |"
    )

    if skip_subsets:
        return

    text_map = {line.split()[0]: " ".join(line.split()[1:]) for line in text_lines}
    utt2dur_map = {line.split()[0]: float(line.split()[1]) for line in utt2dur_lines}
    wav_map = {line.split()[0]: line.split()[1] for line in wav_lines}
    utt2spk_map = {line.split()[0]: line.split()[1] for line in utt2spk_lines}
    default_out_base = data_dir.parent
    enroll_dir = out_enroll_dir or (default_out_base / f"{output_prefix}_enrolls")
    trials_f_dir = out_trials_f_dir or (default_out_base / f"{output_prefix}_trials_f")
    trials_m_dir = out_trials_m_dir or (default_out_base / f"{output_prefix}_trials_m")
    trials_mixed_dir = out_trials_mixed_dir or (default_out_base / f"{output_prefix}_trials_mixed")
    _filter_maps(
        set(enroll_lines),
        utt2spk_map,
        {spk: spk2gender_map[spk] for spk in spk2gender_map if spk in spk2utt},
        spk2utt,
        text_map,
        utt2dur_map,
        wav_map,
        enroll_dir,
        None,
        enroll_lines,
    )
    _filter_maps(
        {u.split()[1] for u in female_trials},
        utt2spk_map,
        {spk: spk2gender_map[spk] for spk in spk2gender_map if spk in spk2utt},
        spk2utt,
        text_map,
        utt2dur_map,
        wav_map,
        trials_f_dir,
        female_trials,
        None,
    )
    _filter_maps(
        {u.split()[1] for u in male_trials},
        utt2spk_map,
        {spk: spk2gender_map[spk] for spk in spk2gender_map if spk in spk2utt},
        spk2utt,
        text_map,
        utt2dur_map,
        wav_map,
        trials_m_dir,
        male_trials,
        None,
    )
    _filter_maps(
        {u.split()[1] for u in mixed_trials},
        utt2spk_map,
        {spk: spk2gender_map[spk] for spk in spk2gender_map if spk in spk2utt},
        spk2utt,
        text_map,
        utt2dur_map,
        wav_map,
        trials_mixed_dir,
        mixed_trials,
        None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build JVS Japanese Kaldi data and trials.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("../corpora/ja/"),
        help="Path to the JVS corpus root directory (contains jvs001/, jvs002/, etc.).",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="parallel100",
        choices=["parallel100", "nonpara30", "both"],
        help="Subset to use: parallel100 (100 common utterances), nonpara30 (30 unique utterances per speaker), or both (process and merge both).",
    )
    parser.add_argument(
        "--gender-file",
        type=Path,
        default=Path("../corpora/ja/gender_f0range.txt"),
        help="Path to gender_f0range.txt file.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("../data/ja"),
        help="Root output directory for the Kaldi maps (text, wav.scp, ...).",
    )
    parser.add_argument(
        "--enroll-per-spk-ratio",
        type=str,
        default="15%",
        help="Share of each speaker's utterances reserved for enrollment (e.g. 15%).",
    )
    parser.add_argument(
        "--enroll-seed",
        type=int,
        default=123,
        help="Random seed used when sampling enrollment utterances.",
    )
    parser.add_argument(
        "--out-enroll-dir",
        type=Path,
        default=None,
        help="Directory that will hold the filtered enrollment Kaldi maps.",
    )
    parser.add_argument(
        "--out-trials-f-dir",
        type=Path,
        default=None,
        help="Directory for filtered female trial maps.",
    )
    parser.add_argument(
        "--out-trials-m-dir",
        type=Path,
        default=None,
        help="Directory for filtered male trial maps.",
    )
    parser.add_argument(
        "--out-trials-mixed-dir",
        type=Path,
        default=None,
        help="Directory for filtered mixed trial maps.",
    )
    parser.add_argument(
        "--skip-subsets",
        action="store_true",
        help="Skip materializing filtered enrollment/trial directories.",
    )
    parser.add_argument(
        "--max-trial-utts-per-speaker",
        type=int,
        default=None,
        help="Maximum number of trial utterances per speaker for balancing. "
             "If set, speakers with more utterances will be downsampled to this number.",
    )
    parser.add_argument(
        "--trial-balance-seed",
        type=int,
        default=123,
        help="Random seed for balancing trial utterances (used with --max-trial-utts-per-speaker).",
    )
    parser.add_argument(
        "--upsample-insufficient-speakers",
        action="store_true",
        help="If set, upsample (with replacement) speakers with fewer trial utterances "
             "to reach the balance threshold. Otherwise, keep all their utterances as-is.",
    )
    parser.add_argument(
        "--split-dev-test",
        action="store_true",
        default=True,
        help="Split speakers into dev and test sets: 15F+15M for dev, 15F+15M for test. (default: True)",
    )
    parser.add_argument(
        "--no-split-dev-test",
        dest="split_dev_test",
        action="store_false",
        help="Disable splitting speakers into dev and test sets (use all selected speakers as one set).",
    )
    parser.add_argument(
        "--dev-female-count",
        type=int,
        default=24,
        help="Number of female speakers for dev set (default: 15).",
    )
    parser.add_argument(
        "--dev-male-count",
        type=int,
        default=24,
        help="Number of male speakers for dev set (default: 15).",
    )
    parser.add_argument(
        "--test-female-count",
        type=int,
        default=24,
        help="Number of female speakers for test set (default: 15).",
    )
    parser.add_argument(
        "--test-male-count",
        type=int,
        default=24,
        help="Number of male speakers for test set (default: 15).",
    )
    parser.add_argument(
        "--no-random-split",
        action="store_true",
        help="Use sequential split instead of random split (default: random split when --split-dev-test is used).",
    )
    parser.add_argument(
        "--total-female-count",
        type=int,
        default=48,
        help="Total number of female speakers to select before splitting (default: 30).",
    )
    parser.add_argument(
        "--total-male-count",
        type=int,
        default=48,
        help="Total number of male speakers to select before splitting (default: 30).",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=123,
        help="Random seed for splitting speakers into dev/test (default: 123).",
    )
    parser.add_argument(
        "--min-utterances",
        type=int,
        default=2,
        help="Minimum number of utterances per speaker to include (default: 2).",
    )
    parser.add_argument(
        "--balance-gender",
        action="store_true",
        help="Balance gender by selecting female speakers with most utterances to match male speaker count.",
    )
    args = parser.parse_args()

    # Always process both subsets: nonpara30 and parallel100
    # Then generate three datasets: nonpara30, parallel100, and both
    subsets_to_process = ["nonpara30", "parallel100"]
    source_dir = args.source_dir
    
    # Step 1: Collect data from both subsets and find common speakers
    print(f"\n{'='*80}")
    print(f"STEP 1: COLLECTING DATA FROM BOTH SUBSETS")
    print(f"{'='*80}")
    
    subset_data: dict[str, dict] = {}  # Store data per subset
    subset_speakers: dict[str, set[str]] = {}
    
    for subset_name in subsets_to_process:
        subset_dirs = list(source_dir.glob(f"jvs*/{subset_name}"))
        
        if not subset_dirs:
            print(f"Warning: No {subset_name} directories found in {source_dir}")
            continue
        
        print(f"\n{'='*80}")
        print(f"PROCESSING {subset_name.upper()}")
        print(f"{'='*80}")
        print(f"Found {len(subset_dirs)} {subset_name} directories")
        
        # Load transcripts from this subset
        subset_transcripts = {}
        subset_speaker_set = set()
        for subset_dir in subset_dirs:
            transcripts_file = subset_dir / "transcripts_utf8.txt"
            if transcripts_file.exists():
                speaker_id = subset_dir.parent.name  # e.g., jvs001
                speaker_transcripts = _load_transcripts(transcripts_file, speaker_id)
                subset_transcripts.update(speaker_transcripts)
                subset_speaker_set.add(speaker_id)
                print(f"  Loaded {len(speaker_transcripts)} transcripts from {speaker_id}")
            else:
                print(f"  Warning: {transcripts_file} not found, skipping")
        
        # Map audio files from this subset
        subset_audio_index = _map_audio_files(source_dir, subset_name)
        audio_speakers = {speaker for speaker, _ in subset_audio_index.values()}
        subset_speaker_set.update(audio_speakers)
        subset_speakers[subset_name] = subset_speaker_set
        
        subset_data[subset_name] = {
            'transcripts': subset_transcripts,
            'audio_index': subset_audio_index,
        }
        
        print(f"  Total transcripts from {subset_name}: {len(subset_transcripts)}")
        print(f"  Total audio files from {subset_name}: {len(subset_audio_index)}")
        print(f"  Speakers in {subset_name}: {len(subset_speaker_set)}")
    
    # Find common speakers
    if len(subset_speakers) != 2:
        raise SystemExit(f"Need exactly 2 subsets, found {len(subset_speakers)}")
    
    common_speakers = subset_speakers[subsets_to_process[0]] & subset_speakers[subsets_to_process[1]]
    print(f"\n{'='*80}")
    print(f"COMMON SPEAKERS")
    print(f"{'='*80}")
    print(f"Speakers in {subsets_to_process[0]}: {len(subset_speakers[subsets_to_process[0]])}")
    print(f"Speakers in {subsets_to_process[1]}: {len(subset_speakers[subsets_to_process[1]])}")
    print(f"Common speakers (present in BOTH subsets): {len(common_speakers)}")
    
    if not common_speakers:
        raise SystemExit(f"No common speakers found between {subsets_to_process[0]} and {subsets_to_process[1]}!")
    
    # Filter each subset to only include common speakers
    for subset_name in subsets_to_process:
        subset_transcripts = subset_data[subset_name]['transcripts']
        subset_audio_index = subset_data[subset_name]['audio_index']
        
        filtered_transcripts = {
            utt_id: text for utt_id, text in subset_transcripts.items()
            if any(utt_id.startswith(f"{spk}_") for spk in common_speakers)
        }
        filtered_audio_index = {
            utt_id: (speaker, wav_path) for utt_id, (speaker, wav_path) in subset_audio_index.items()
            if speaker in common_speakers
        }
        
        subset_data[subset_name]['transcripts'] = filtered_transcripts
        subset_data[subset_name]['audio_index'] = filtered_audio_index
        
        print(f"{subset_name} (filtered to common speakers): {len(filtered_transcripts)} transcripts, {len(filtered_audio_index)} audio files")
    
    # Merge data from both subsets for speaker selection
    all_transcripts = {}
    all_audio_index = {}
    for subset_name in subsets_to_process:
        all_transcripts.update(subset_data[subset_name]['transcripts'])
        all_audio_index.update(subset_data[subset_name]['audio_index'])
    
    if not all_transcripts:
        raise SystemExit("No transcripts found!")
    
    print(f"\n{'='*80}")
    print(f"MERGED DATA (for speaker selection)")
    print(f"{'='*80}")
    print(f"Total transcripts: {len(all_transcripts)}")
    print(f"Total audio files: {len(all_audio_index)}")
    
    # Use merged data for speaker selection
    audio_index = all_audio_index
    
    # Build Kaldi maps
    wav_lines, utt2spk_lines, utt2dur_lines, spk2utt_lines, text_lines, spk2utt = _build_kaldi_maps(
        all_transcripts, audio_index
    )

    # Load speaker genders
    spk2gender_map = _load_speaker_genders(args.gender_file)
    print(f"\nLoaded gender information for {len(spk2gender_map)} speakers from {args.gender_file}")
    print(f"Speakers in spk2utt: {len(spk2utt)}")
    print(f"Speakers with gender info: {len([spk for spk in spk2utt if spk in spk2gender_map])}")
    
    # Debug: show some speaker IDs
    if spk2utt:
        sample_spks = list(spk2utt.keys())[:5]
        print(f"Sample speaker IDs in spk2utt: {sample_spks}")
    if spk2gender_map:
        sample_gender_spks = list(spk2gender_map.keys())[:5]
        print(f"Sample speaker IDs in gender map: {sample_gender_spks}")
    
    spk2gender_lines = [f"{spk} {spk2gender_map[spk]}" for spk in sorted(spk2gender_map) if spk in spk2utt]

    # Write main Kaldi maps
    args.data_dir.mkdir(parents=True, exist_ok=True)
    _write_maps(
        args.data_dir,
        text_lines,
        wav_lines,
        utt2spk_lines,
        utt2dur_lines,
        spk2utt_lines,
        spk2gender_lines,
    )

    # Get all speakers by gender, filter by minimum utterances, and sort by utterance count (descending)
    # Debug: check gender matching
    male_count = sum(1 for spk in spk2utt if spk2gender_map.get(spk) == "M")
    female_count = sum(1 for spk in spk2utt if spk2gender_map.get(spk) == "F")
    unknown_count = sum(1 for spk in spk2utt if spk not in spk2gender_map or spk2gender_map.get(spk) not in ["M", "F"])
    print(f"\nGender distribution in spk2utt:")
    print(f"  Male: {male_count}")
    print(f"  Female: {female_count}")
    print(f"  Unknown/Missing: {unknown_count}")
    
    male_speakers_all = [
        spk for spk in spk2utt 
        if spk2gender_map.get(spk) == "M" and len(spk2utt.get(spk, [])) >= args.min_utterances
    ]
    female_speakers_all = [
        spk for spk in spk2utt 
        if spk2gender_map.get(spk) == "F" and len(spk2utt.get(spk, [])) >= args.min_utterances
    ]
    
    print(f"\nAfter filtering (min_utterances={args.min_utterances}):")
    print(f"  Male speakers: {len(male_speakers_all)}")
    print(f"  Female speakers: {len(female_speakers_all)}")
    
    # Count excluded speakers
    excluded_male = [spk for spk in spk2utt if spk2gender_map.get(spk) == "M" and len(spk2utt.get(spk, [])) < args.min_utterances]
    excluded_female = [spk for spk in spk2utt if spk2gender_map.get(spk) == "F" and len(spk2utt.get(spk, [])) < args.min_utterances]
    
    if excluded_male or excluded_female:
        print(f"\n{'='*80}")
        print(f"EXCLUDING SPEAKERS WITH < {args.min_utterances} UTTERANCES")
        print(f"{'='*80}")
        if excluded_female:
            print(f"\nExcluded Female Speakers ({len(excluded_female)}):")
            for spk in excluded_female:
                count = len(spk2utt.get(spk, []))
                print(f"  {spk}: {count} utterances")
        if excluded_male:
            print(f"\nExcluded Male Speakers ({len(excluded_male)}):")
            for spk in excluded_male:
                count = len(spk2utt.get(spk, []))
                print(f"  {spk}: {count} utterances")
        print(f"\nTotal excluded: {len(excluded_female) + len(excluded_male)} speakers")
        print(f"Remaining: {len(female_speakers_all)} female + {len(male_speakers_all)} male = {len(female_speakers_all) + len(male_speakers_all)} speakers")
    
    # Sort by utterance count (descending)
    male_speakers_sorted = sorted(male_speakers_all, key=lambda spk: len(spk2utt.get(spk, [])), reverse=True)
    female_speakers_sorted = sorted(female_speakers_all, key=lambda spk: len(spk2utt.get(spk, [])), reverse=True)
    
    # Split into dev/test if requested
    if args.split_dev_test:
        # Use default values if not specified
        total_female = args.total_female_count if args.total_female_count > 0 else 30
        total_male = args.total_male_count if args.total_male_count > 0 else 30
        
        # First, select top N female and top N male speakers by utterance count
        selected_female = female_speakers_sorted[:total_female]
        selected_male = male_speakers_sorted[:total_male]
        
        # Check if we have enough speakers
        if len(selected_female) < total_female:
            raise SystemExit(f"Not enough female speakers: need {total_female}, have {len(selected_female)}")
        if len(selected_male) < total_male:
            raise SystemExit(f"Not enough male speakers: need {total_male}, have {len(selected_male)}")
        
        # Selected speakers will be displayed in the split sections below (DEV/TEST)
        
        # Split selected speakers into dev/test
        # Default to random split when using split_dev_test, unless --no-random-split is specified
        use_random_split = not args.no_random_split
        
        if use_random_split:
            # Random split
            rng = random.Random(args.split_seed)
            selected_female_shuffled = selected_female.copy()
            selected_male_shuffled = selected_male.copy()
            rng.shuffle(selected_female_shuffled)
            rng.shuffle(selected_male_shuffled)
            
            dev_female = selected_female_shuffled[:args.dev_female_count]
            test_female = selected_female_shuffled[args.dev_female_count:args.dev_female_count + args.test_female_count]
            dev_male = selected_male_shuffled[:args.dev_male_count]
            test_male = selected_male_shuffled[args.dev_male_count:args.dev_male_count + args.test_male_count]
            
            print(f"\n{'='*80}")
            print(f"RANDOM SPLIT INTO DEV AND TEST (seed={args.split_seed})")
            print(f"{'='*80}")
        else:
            # Sequential split (original behavior)
            dev_female = selected_female[:args.dev_female_count]
            test_female = selected_female[args.dev_female_count:args.dev_female_count + args.test_female_count]
            dev_male = selected_male[:args.dev_male_count]
            test_male = selected_male[args.dev_male_count:args.dev_male_count + args.test_male_count]
            
            print(f"\n{'='*80}")
            print(f"SEQUENTIAL SPLIT INTO DEV AND TEST")
            print(f"{'='*80}")
        
        # Check if we have enough speakers after split
        if len(dev_female) < args.dev_female_count:
            raise SystemExit(f"Not enough female speakers for dev set: need {args.dev_female_count}, have {len(dev_female)}")
        if len(dev_male) < args.dev_male_count:
            raise SystemExit(f"Not enough male speakers for dev set: need {args.dev_male_count}, have {len(dev_male)}")
        if len(test_female) < args.test_female_count:
            raise SystemExit(f"Not enough female speakers for test set: need {args.test_female_count}, have {len(test_female)}")
        if len(test_male) < args.test_male_count:
            raise SystemExit(f"Not enough male speakers for test set: need {args.test_male_count}, have {len(test_male)}")
        
        # Display DEV Female
        print(f"\n{'='*80}")
        print(f"DEV SET - FEMALE SPEAKERS ({len(dev_female)})")
        print(f"{'='*80}")
        print(f"{'Rank':<6} {'Speaker ID':<15} {'Utterances':<12} {'Global Rank':<12}")
        print("-" * 50)
        for rank, spk in enumerate(dev_female, 1):
            count = len(spk2utt.get(spk, []))
            global_rank = female_speakers_sorted.index(spk) + 1
            print(f"{rank:<6} {spk:<15} {count:<12} {global_rank:<12}")
        
        # Display DEV Male
        print(f"\n{'='*80}")
        print(f"DEV SET - MALE SPEAKERS ({len(dev_male)})")
        print(f"{'='*80}")
        print(f"{'Rank':<6} {'Speaker ID':<15} {'Utterances':<12} {'Global Rank':<12}")
        print("-" * 50)
        for rank, spk in enumerate(dev_male, 1):
            count = len(spk2utt.get(spk, []))
            global_rank = male_speakers_sorted.index(spk) + 1
            print(f"{rank:<6} {spk:<15} {count:<12} {global_rank:<12}")
        
        # Display TEST Female
        print(f"\n{'='*80}")
        print(f"TEST SET - FEMALE SPEAKERS ({len(test_female)})")
        print(f"{'='*80}")
        print(f"{'Rank':<6} {'Speaker ID':<15} {'Utterances':<12} {'Global Rank':<12}")
        print("-" * 50)
        for rank, spk in enumerate(test_female, 1):
            count = len(spk2utt.get(spk, []))
            global_rank = female_speakers_sorted.index(spk) + 1
            print(f"{rank:<6} {spk:<15} {count:<12} {global_rank:<12}")
        
        # Display TEST Male
        print(f"\n{'='*80}")
        print(f"TEST SET - MALE SPEAKERS ({len(test_male)})")
        print(f"{'='*80}")
        print(f"{'Rank':<6} {'Speaker ID':<15} {'Utterances':<12} {'Global Rank':<12}")
        print("-" * 50)
        for rank, spk in enumerate(test_male, 1):
            count = len(spk2utt.get(spk, []))
            global_rank = male_speakers_sorted.index(spk) + 1
            print(f"{rank:<6} {spk:<15} {count:<12} {global_rank:<12}")
        
        # Print summary statistics
        dev_f_counts = [len(spk2utt.get(spk, [])) for spk in dev_female]
        dev_m_counts = [len(spk2utt.get(spk, [])) for spk in dev_male]
        test_f_counts = [len(spk2utt.get(spk, [])) for spk in test_female]
        test_m_counts = [len(spk2utt.get(spk, [])) for spk in test_male]
        
        # Print summary statistics separated by set and gender
        print(f"\n{'='*80}")
        print(f"SUMMARY STATISTICS")
        print(f"{'='*80}")
        
        print(f"\nDEV SET:")
        print(f"  Female: min={min(dev_f_counts)}, max={max(dev_f_counts)}, "
              f"mean={sum(dev_f_counts)/len(dev_f_counts):.1f}, "
              f"median={sorted(dev_f_counts)[len(dev_f_counts)//2]}")
        print(f"  Male: min={min(dev_m_counts)}, max={max(dev_m_counts)}, "
              f"mean={sum(dev_m_counts)/len(dev_m_counts):.1f}, "
              f"median={sorted(dev_m_counts)[len(dev_m_counts)//2]}")
        
        print(f"\nTEST SET:")
        print(f"  Female: min={min(test_f_counts)}, max={max(test_f_counts)}, "
              f"mean={sum(test_f_counts)/len(test_f_counts):.1f}, "
              f"median={sorted(test_f_counts)[len(test_f_counts)//2]}")
        print(f"  Male: min={min(test_m_counts)}, max={max(test_m_counts)}, "
              f"mean={sum(test_m_counts)/len(test_m_counts):.1f}, "
              f"median={sorted(test_m_counts)[len(test_m_counts)//2]}")
        
        # Process both dev and test sets
        ratio_raw = args.enroll_per_spk_ratio.strip()
        percent = ratio_raw.endswith("%")
        ratio_value = float(ratio_raw[:-1]) / 100.0 if percent else float(ratio_raw)
        if not (0 < ratio_value <= 1):
            raise SystemExit("--enroll-per-spk-ratio must be 0<ratio<=1 or a percentage")
        
        print(f"\n{'='*80}")
        print(f"STEP 2: PROCESSING COMBINED DATASET")
        print(f"{'='*80}")
        print(f"✓ Same speakers selected: {len(dev_female + dev_male + test_female + test_male)} speakers")
        print(f"✓ Combining nonpara30 and parallel100 subsets")
        
        # Combine transcripts and audio from both subsets
        combined_transcripts = {}
        combined_audio_index = {}
        for subset_name in subsets_to_process:
            combined_transcripts.update(subset_data[subset_name]['transcripts'])
            combined_audio_index.update(subset_data[subset_name]['audio_index'])
        
        # Build combined Kaldi maps
        combined_wav_lines, combined_utt2spk_lines, combined_utt2dur_lines, combined_spk2utt_lines, combined_text_lines, combined_spk2utt = _build_kaldi_maps(
            combined_transcripts, combined_audio_index
        )
        
        # Process combined DEV set
        _process_single_set(
            dev_female,
            dev_male,
            combined_spk2utt,
            spk2gender_map,
            combined_text_lines,
            combined_wav_lines,
            combined_utt2spk_lines,
            combined_utt2dur_lines,
            combined_spk2utt_lines,
            "dev",
            "ja_dev",
            args.data_dir,
            ratio_value,
            args.enroll_seed,
            args.max_trial_utts_per_speaker,
            args.trial_balance_seed,
            args.upsample_insufficient_speakers,
            args.out_enroll_dir,
            args.out_trials_f_dir,
            args.out_trials_m_dir,
            args.out_trials_mixed_dir,
            args.skip_subsets,
        )
        
        # Process combined TEST set
        _process_single_set(
            test_female,
            test_male,
            combined_spk2utt,
            spk2gender_map,
            combined_text_lines,
            combined_wav_lines,
            combined_utt2spk_lines,
            combined_utt2dur_lines,
            combined_spk2utt_lines,
            "test",
            "ja_test",
            args.data_dir,
            ratio_value,
            args.enroll_seed,
            args.max_trial_utts_per_speaker,
            args.trial_balance_seed,
            args.upsample_insufficient_speakers,
            args.out_enroll_dir,
            args.out_trials_f_dir,
            args.out_trials_m_dir,
            args.out_trials_mixed_dir,
            args.skip_subsets,
        )
        return
    else:
        # Use all speakers or limit to total counts if specified
        # Note: default values are 30, so check if explicitly set to 0
        if args.total_female_count > 0 and args.total_male_count > 0:
            # Limit to top N speakers by utterance count
            male_targets = male_speakers_sorted[:args.total_male_count]
            female_targets_all = female_speakers_sorted[:args.total_female_count]
            print(f"\n{'='*80}")
            print(f"SELECTED TOP {args.total_female_count} FEMALE + TOP {args.total_male_count} MALE SPEAKERS")
            print(f"{'='*80}")
            print(f"Selected: {len(female_targets_all)} female + {len(male_targets)} male = {len(female_targets_all) + len(male_targets)} speakers")
        else:
            # Use all speakers (only if explicitly set to 0)
            male_targets = male_speakers_sorted
            female_targets_all = female_speakers_sorted
            print(f"\n{'='*80}")
            print(f"USING ALL AVAILABLE SPEAKERS (total_female_count={args.total_female_count}, total_male_count={args.total_male_count})")
            print(f"{'='*80}")
            print(f"Total: {len(female_targets_all)} female + {len(male_targets)} male = {len(female_targets_all) + len(male_targets)} speakers")
    
    # Balance gender if requested (only if not splitting)
    if args.balance_gender and not args.split_dev_test:
        num_male = len(male_targets)
        if len(female_targets_all) > num_male:
            # Select top N female speakers from already limited list
            female_targets = female_targets_all[:num_male]
        else:
            female_targets = female_targets_all
    elif not args.split_dev_test:
        # Use already limited targets from above (or all if not limited)
        female_targets = female_targets_all
    
    # Process single set (non-split mode) - generate all three datasets
    ratio_raw = args.enroll_per_spk_ratio.strip()
    percent = ratio_raw.endswith("%")
    ratio_value = float(ratio_raw[:-1]) / 100.0 if percent else float(ratio_raw)
    if not (0 < ratio_value <= 1):
        raise SystemExit("--enroll-per-spk-ratio must be 0<ratio<=1 or a percentage")
    
    print(f"\n{'='*80}")
    print(f"PROCESSING COMBINED DATASET")
    print(f"{'='*80}")
    print(f"✓ Combining nonpara30 and parallel100 subsets")
    
    # Combine transcripts and audio from both subsets
    combined_transcripts = {}
    combined_audio_index = {}
    for subset_name in subsets_to_process:
        combined_transcripts.update(subset_data[subset_name]['transcripts'])
        combined_audio_index.update(subset_data[subset_name]['audio_index'])
    
    # Build combined Kaldi maps
    combined_wav_lines, combined_utt2spk_lines, combined_utt2dur_lines, combined_spk2utt_lines, combined_text_lines, combined_spk2utt = _build_kaldi_maps(
        combined_transcripts, combined_audio_index
    )
    
    _process_single_set(
        female_targets,
        male_targets,
        combined_spk2utt,
        spk2gender_map,
        combined_text_lines,
        combined_wav_lines,
        combined_utt2spk_lines,
        combined_utt2dur_lines,
        combined_spk2utt_lines,
        "jvs",
        "ja",
        args.data_dir,
        ratio_value,
        args.enroll_seed,
        args.max_trial_utts_per_speaker,
        args.trial_balance_seed,
        args.upsample_insufficient_speakers,
        args.out_enroll_dir,
        args.out_trials_f_dir,
        args.out_trials_m_dir,
        args.out_trials_mixed_dir,
        args.skip_subsets,
    )


if __name__ == "__main__":
    main()
