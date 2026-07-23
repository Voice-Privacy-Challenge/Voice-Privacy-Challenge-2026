#!/bin/bash

# Fresh install with "rm .micromamba/micromamba .done-*"

set -e

nj=$(nproc)
home=$PWD
venv_dir=$PWD/venv
source ./env.sh

ESPAK_VERSION=1.51.1
compute_and_write_hash "anonymization/pipelines/ssl/requirements.txt"    # SHA256: 163861d89b6b64d1d64e5e4a7bb080388f866505536bf9d6d1a412e0d747036f
trigger_new_install "exp/ssl_models .done-ssl-requirements"



mark=.done-ssl-requirements
if [ ! -f $mark ]; then
  echo " == Installing SSL python libraries =="
  pip3 install -r anonymization/pipelines/ssl/requirements.txt  || exit 1
  touch $mark
fi


# Download SSL pre-models only if perform SSL anonymization
if [ ! -d exp/ssl_models ]; then
    echo "Download pretrained models of SSL-based speaker anonymization system..."
    mkdir -p exp/
    cd exp/
    wget  -O ssl_models.zip https://duke.app.box.com/shared/static/qe1idwbhzaueyoziimjm6wo56olhtnf7
    unzip ssl_models.zip
    wget  -O ssl_models/hubert_base_ls960.pt https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt
    cd ../
    #wget  -O exp/ssl_models/pretrained_models_anon_xv.tar.gz https://zenodo.org/record/6529898/files/pretrained_models_anon_xv.tar.gz
    #tar -xzvf exp/ssl_models/pretrained_models_anon_xv.tar.gz -C exp/ssl_models
    #rm exp/ssl_models/*.tar.gz
fi



