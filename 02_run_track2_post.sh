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
  anon_config=configs/$track/anon_post_BM1.yaml
  #anon_config=configs/$track/anon_post_BM2.yaml
  #anon_config=configs/$track/anon_post_BM3.yaml

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
python run_anonymization.py --config ${anon_config} ${force_compute}

