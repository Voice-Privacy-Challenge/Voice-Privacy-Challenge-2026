#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
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
  #anon_config=configs/$track/anon_post_sttts.yaml
  anon_config=configs/$track/anon_post_ssl.yaml
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
# Generate anonymized audio (multilang training set)
echo "Running anonymization..."
python run_anonymization.py --config ${anon_config} ${force_compute}

# perfom ASV post training evaluation
python run_evaluation.py --config $(dirname ${anon_config})/eval_post_cn.yaml --overwrite "${eval_overwrite}" ${force_compute}
python run_evaluation.py --config $(dirname ${anon_config})/eval_post_ja.yaml --overwrite "${eval_overwrite}" ${force_compute}
#python run_evaluation.py --config $(dirname ${anon_config})/eval_post_en.yaml --overwrite "${eval_overwrite}" ${force_compute}
#python run_evaluation.py --config $(dirname ${anon_config})/eval_post_de.yaml --overwrite "${eval_overwrite}" ${force_compute}
#python run_evaluation.py --config $(dirname ${anon_config})/eval_post_es.yaml --overwrite "${eval_overwrite}" ${force_compute}
#python run_evaluation.py --config $(dirname ${anon_config})/eval_post_fr.yaml --overwrite "${eval_overwrite}" ${force_compute}


# Record post results only (ignore eval_pre)
results_exp=exp/results_summary/$track
mkdir -p ${results_exp}
: > "${results_exp}/result_for_rank${anon_suffix}"
for f in exp/results_summary/${track}/eval_anon_*${anon_suffix}/results_anon.txt; do
  [ -f "$f" ] && { cat "$f"; echo; } >> "${results_exp}/result_for_rank${anon_suffix}"
done

# Zip: track2-only; include exp/asv_ssl but exclude track1 (libri_*)
zip ${results_exp}/result_for_submission${anon_suffix}.zip -r \
  -x "*libri*" \
  "${results_exp}/result_for_rank${anon_suffix}" \
  exp/openai exp/ser_emotion2vec \
  exp/asv_ssl \
  exp/asv_anon_track2*${anon_suffix} \
  exp/results_summary/${track} exp/results_summary/eval_orig${anon_suffix} \
  > /dev/null 2>&1 || true
