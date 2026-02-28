import argparse
import os
import random

def _read_enroll_file(enroll_file):
    """Read the enroll file and return a list of (speaker, utterance) pairs."""
    with open(enroll_file, "r") as f:
        enroll_lines = f.readlines()
    enroll_pairs = []
    for line in enroll_lines:
        utterance = line.strip()
        speaker = utterance.split("-")[0]
        enroll_pairs.append((speaker, utterance))
    return enroll_pairs

def _read_trial_file(trial_file):
    """Read the trial file and return a list of (speaker, utterance, label) tuples."""
    with open(trial_file, "r") as f:
        trial_lines = f.readlines()
    trial_tuples = []
    for line in trial_lines:
        speaker, utterance, label = line.strip().split()
        trial_tuples.append((speaker, utterance, label))
    return trial_tuples


def _load_speaker_genders(spk2gender_file):
    """Read the spk2gender file and return a dict mapping speaker id to gender."""
    genders = {}
    with open(spk2gender_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                spk_id, gender = line.split()
                genders[spk_id] = gender
    return genders


def merge_kaldi_files(file_name, dir_f, dir_m, out_dir):
    """Merges a Kaldi file from f and m dirs, sorts them by the first column, and writes to out_dir."""
    lines = []
    
    file_f = os.path.join(dir_f, file_name)
    if os.path.exists(file_f):
        with open(file_f, "r") as f:
            lines_f = f.readlines()
            lines.extend(lines_f)
    else:
        lines_f = []
        
    file_m = os.path.join(dir_m, file_name)
    if os.path.exists(file_m):
        with open(file_m, "r") as f:
            lines_m = f.readlines()
            lines.extend(lines_m)
    else:
        lines_m = []

    # Sort lines by the first column
    lines.sort(key=lambda x: x.split()[0])
    
    out_file = os.path.join(out_dir, file_name)
    with open(out_file, "w") as f:
        f.writelines(lines)
        
    print(f"{file_name}: {len(lines_f)} (f), {len(lines_m)} (m) -> {len(lines)} (mixed)")


def main():
    parser = argparse.ArgumentParser(description="Generate mixed trials for LibriSpeech evaluation.")
    parser.add_argument("--partition", type=str, required=True, choices=["test", "dev"], help="Partition name (test or dev)")
    args = parser.parse_args()

    # Paths built dynamically based on partition choice
    enrolls_file = f"data/libri_{args.partition}_enrolls/enrolls"
    spk2gender_file = f"data/libri_{args.partition}_enrolls/spk2gender"
    trials_f_file = f"data/libri_{args.partition}_trials_f/trials"
    trials_m_file = f"data/libri_{args.partition}_trials_m/trials"

    trials_f_dir = f"data/libri_{args.partition}_trials_f"
    trials_m_dir = f"data/libri_{args.partition}_trials_m"
    trials_mixed_dir = f"data/libri_{args.partition}_trials_mixed"

    if not os.path.exists(trials_mixed_dir):
        os.makedirs(trials_mixed_dir)
        print(f"Created directory: {trials_mixed_dir}")

    enroll_pairs = _read_enroll_file(enrolls_file)
    male_trials = _read_trial_file(trials_f_file)
    female_trials = _read_trial_file(trials_m_file)

    unique_female_trials = set([t[1] for t in female_trials])
    unique_male_trials = set([t[1] for t in male_trials])

    print(f"Number of enroll pairs: {len(enroll_pairs)}")
    print(f"Number of female trials: {len(female_trials)}")
    print(f"Number of male trials: {len(male_trials)}")


    female_counts = {u: 0 for u in unique_female_trials}
    for _, u, label in female_trials:
        if label == "nontarget":
            female_counts[u] += 1
    avg_female_trials_count = round(sum(female_counts.values()) / len(female_counts))

    male_counts = {u: 0 for u in unique_male_trials}
    for _, u, label in male_trials:
        if label == "nontarget":
            male_counts[u] += 1
    avg_male_trials_count = round(sum(male_counts.values()) / len(male_counts))

    print(f"Average female nontarget trials per speaker: {avg_female_trials_count}")
    print(f"Average male nontarget trials per speaker: {avg_male_trials_count}")


    # i'm out of time and i have a train to catch and i still have to fucking pack
    # so i'll just do a stupidly simple approach
    # for each unique female trial, i'll pair them with a bunch of random males (nontarget, ofc)
    # same for males, other way around
    # for both i'll keep the same number of average values per nontarget utterance in their original partition
    # well maybe there could have been more "stupidly simple" approaches
    # but hey why don't you write the code yourself while i pack then

    gender_mapping = _load_speaker_genders(spk2gender_file)
    male_enrolls = [pair[0] for pair in enroll_pairs if gender_mapping[pair[0]] == "m"]
    female_enrolls = [pair[0] for pair in enroll_pairs if gender_mapping[pair[0]] == "f"]

    male_enroll_vs_female_trials = []
    for trial in unique_female_trials:
        random_males = random.sample(male_enrolls, avg_female_trials_count)
        for male in random_males:
            male_enroll_vs_female_trials.append((male, trial, "nontarget"))

    female_enroll_vs_male_trials = []
    for trial in unique_male_trials:
        random_females = random.sample(female_enrolls, avg_male_trials_count)
        for female in random_females:
            female_enroll_vs_male_trials.append((female, trial, "nontarget"))

    mixed_trials = male_enroll_vs_female_trials + female_enroll_vs_male_trials
    print(f'Obtained {len(mixed_trials)} mixed trials.')

    all_trials = mixed_trials + male_trials + female_trials
    print(f'New trial count: {len(all_trials)} (female, male, mixed).')

    # done. now we just need to merge the other kaldi files for trials_f and trials_m
    # it's all boilerplate so i'll let gemini do it
    # it's gonna do a much better job than me anyway

    print("\nMerging Kaldi files...")
    kaldi_files = ["spk2gender", "spk2utt", "wav.scp", "text", "utt2dur", "utt2spk"]
    for kf in kaldi_files:
        merge_kaldi_files(kf, trials_f_dir, trials_m_dir, trials_mixed_dir)

    print("\nWriting new trials file...")
    mixed_trials_file = os.path.join(trials_mixed_dir, "trials")
    with open(mixed_trials_file, "w") as f:
        for spk, utt, label in all_trials:
            f.write(f"{spk} {utt} {label}\n")


if __name__ == "__main__":
    main()
