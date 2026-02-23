#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
set -e

cd "$(dirname "$0")"
source env.sh

### Variables
#select track
track=track2 #track1, track2

# Select the anonymization pipeline
if [ -n "$1" ]; then
  anon_config=$1
else
  # anon_config=configs/$track/anon_sttts.yaml
  anon_config=configs/$track/anon_ssl.yaml
fi
echo "Using config: $anon_config"

force_compute=
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
echo $anon_suffix
# Generate anonymized audio (multilang dev+test set & emotion_track2)
echo "Running anonymization..."
python run_anonymization.py --config ${anon_config} ${force_compute}

# Perform multilang dev+test & emotion_track2 pre evaluation using pretrained ASR/ASV/SER models
python run_evaluation.py --config $(dirname ${anon_config})/eval_pre.yaml --overwrite "${eval_overwrite}" ${force_compute}

# Merge results
results_summary_path_orig=$(python3 -c "from hyperpyyaml import load_hyperpyyaml; f = open('$(dirname ${anon_config})/eval_pre.yaml'); print(load_hyperpyyaml(f, ${eval_overwrite}).get('results_summary_path', ''))")


results_exp=exp/results_summary/$track
mkdir -p ${results_exp}
# Only copy eval_pre results 
cp "${results_summary_path_orig}" "${results_exp}/result_for_rank${anon_suffix}"
# Copy CSV results (ASR=openai/whisper-large-v3, SER=emotion2vec, ASV=asv_ssl)
[ -f "exp/openai/whisper-large-v3/results${anon_suffix}.csv" ] && cp "exp/openai/whisper-large-v3/results${anon_suffix}.csv" "${results_exp}/asr_results${anon_suffix}.csv"
[ -f "exp/ser_emotion2vec/results${anon_suffix}.csv" ] && cp "exp/ser_emotion2vec/results${anon_suffix}.csv" "${results_exp}/ser_results${anon_suffix}.csv"
[ -f "exp/asv_ssl/results${anon_suffix}.csv" ] && cp "exp/asv_ssl/results${anon_suffix}.csv" "${results_exp}/asv_results${anon_suffix}.csv"
# Zip for submission (result_for_rank is primary; CSVs and exp outputs for inspection)
# Include exp/asv_ssl but exclude track1 (libri_*)
zip ${results_exp}/result_for_submission${anon_suffix}.zip -r \
  -x "*libri*" \
  "${results_exp}/result_for_rank${anon_suffix}" \
  exp/openai exp/ser_emotion2vec exp/asv_ssl \
  exp/results_summary/*${anon_suffix}* \
  ${results_exp}/*.csv \
  > /dev/null 2>&1 || true
