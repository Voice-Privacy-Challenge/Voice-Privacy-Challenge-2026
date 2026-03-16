#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0 # change based on your GPU configuration
set -e

cd "$(dirname "$0")"
source env.sh

### Variables

#select track
track=track1 #track1, track2

# Select the anonymization pipeline
if [ -n "$1" ]; then
  anon_config=$1
else
  # anon_config=configs/${track}/anon_mcadams.yaml # B2 anonymization costs 2 hours
  # anon_config=configs/${track}/anon_sttts.yaml # B3 anonymization using precomputed features, costs 13 hours
  # anon_config=configs/${track}/anon_nac.yaml # B4
  anon_config=configs/${track}/anon_asrbn.yaml # B5 anonymization costs 1 hour

fi
echo "Using config: $anon_config"


force_compute='--force_compute False'

# JSON to modify run_evaluation(s) configs, see below
eval_overwrite="{"

### Anonymization + Evaluation:

# find the $anon_suffix (data/dataset_$anon_suffix) = to where the anonymization produces the data files
anon_suffix=$(python3 -c "from hyperpyyaml import load_hyperpyyaml; f = open('${anon_config}'); print(load_hyperpyyaml(f, None).get('anon_suffix', ''))")
if [[ $anon_suffix ]]; then
  eval_overwrite="$eval_overwrite \"anon_data_suffix\": \"$anon_suffix\"}"
else
  eval_overwrite="$eval_overwrite}"
fi

# Generate anonymized audio for Libri (dev/test), IEMOCAP (dev/test), and Libri train-clean-360
# LibriSpeech dev 
# data/libri_dev_enrolls${anon_data_suffix}/wav/*.wav
# data/libri_dev_trials_mixed${anon_data_suffix}/wav/*.wav

# LibriSpeech test
# data/libri_test_enrolls${anon_data_suffix}/wav/*.wav
# data/libri_test_trials_mixed${anon_data_suffix}/wav/*.wav

# IEMOCAP
# data/IEMOCAP_dev${anon_data_suffix}/wav/*.wav
# data/IEMOCAP_test${anon_data_suffix}/wav/*.wav

# LibriSpeech train-clean-360 (for semi-attacker training)
# data/train-clean-360${anon_data_suffix}/wav/*.wav
python run_anonymization.py --config ${anon_config} ${force_compute}

# Perform libri dev+test & IEMOCAP dev+test pre evaluation using pretrained ASR/ASV/SER models
python run_evaluation.py --config $(dirname ${anon_config})/eval_pre.yaml --overwrite "${eval_overwrite}" ${force_compute}

# # Train post ASV using anonymized libri-360 and perform libri dev+test post evaluation
python run_evaluation.py --config $(dirname ${anon_config})/eval_post.yaml --overwrite "${eval_overwrite}" ${force_compute}

# # Merge results
config_dir=$(dirname ${anon_config})
results_summary_path_orig=$(eval_overwrite="${eval_overwrite}" config_dir="${config_dir}" python3 -c "
import os, json
from hyperpyyaml import load_hyperpyyaml
overwrite = json.loads(os.environ.get('eval_overwrite', '{}'))
f = open(os.environ['config_dir'] + '/eval_pre.yaml')
print(load_hyperpyyaml(f, overwrite).get('results_summary_path', ''))
")
results_summary_path_anon=$(eval_overwrite="${eval_overwrite}" config_dir="${config_dir}" python3 -c "
import os, json
from hyperpyyaml import load_hyperpyyaml
overwrite = json.loads(os.environ.get('eval_overwrite', '{}'))
f = open(os.environ['config_dir'] + '/eval_post.yaml')
print(load_hyperpyyaml(f, overwrite).get('results_summary_path', ''))
")
[[ "$results_summary_path_anon" == *"_test_tool"* ]] && exit 0

results_exp=exp/results_summary/$track
mkdir -p ${results_exp}
{ cat "${results_summary_path_orig}"; echo; cat "${results_summary_path_anon}"; } > "${results_exp}/result_for_rank${anon_suffix}"

# Zip for submission: result_for_rank + CSVs + asv_anon files only (no model dirs)
zip -r ${results_exp}/result_for_submission${anon_suffix}.zip \
  "${results_exp}/result_for_rank${anon_suffix}" \
  exp/asr/*${anon_suffix} \
  exp/asr/results*${anon_suffix}.csv \
  exp/ser/results*${anon_suffix}.csv \
  exp/ser/results_folds*${anon_suffix}.csv \
  exp/asv_ssl/results*${anon_suffix}.csv 2>/dev/null || true
[ -d "exp/asv_anon${anon_suffix}" ] && find exp/asv_anon${anon_suffix} -maxdepth 1 -type f -exec zip -q ${results_exp}/result_for_submission${anon_suffix}.zip {} \; 2>/dev/null || true

