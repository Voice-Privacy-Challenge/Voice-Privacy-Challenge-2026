#!/bin/bash

set -e

source env.sh

# librispeech_corpus=PATH_TO_Librispeech
# iemocap_corpus=PATH_TO_IEMOCAP

if [ ! -d "data/libri_dev" ]; then
    echo "Downloading ..."
    wget -O track1_evaldata.zip https://duke.app.box.com/shared/static/ehfkeudzj6nnm4cq3kbcjtaqt3mofudb
    unzip track1_evaldata.zip
fi



check=corpora/LibriSpeech/train-clean-360
if [ ! -d $check ]; then
    if [ ! -z $librispeech_corpus ]; then
        if [ -d $librispeech_corpus/train-clean-360 ]; then
            [ -d corpora/LibriSpeech ] && rm corpora/LibriSpeech
            echo "Linking '$librispeech_corpus' to 'corpora'"
            mkdir -p corpora
            ln -s $librispeech_corpus corpora
        else
          echo "librispeech_corpus is defined to '$librispeech_corpus', but '$librispeech_corpus/train-clean-360' does not exists."
          echo "Either remove the librispeech_corpus variable from the $0 script to download the dataset or modify it to the correct target."
          exit 1
        fi
    fi
fi
#Download LibriSpeech-360
if [ ! -d $check ]; then
    echo "Download train-clean-360..."
    mkdir -p corpora
    cd corpora
    if [ ! -f train-clean-360.tar.gz ] ; then
        echo "Download train-clean-360..."
        wget --no-check-certificate https://www.openslr.org/resources/12/train-clean-360.tar.gz
    fi
    echo "Unpacking train-clean-360"
    tar -xzf train-clean-360.tar.gz
    cd ../
fi


model=asr
if [ ! -d "exp/$model" ]; then
    if [ ! -f .${model}.zip ]; then
        echo "Download pretrained $model models pre-trained..."
        wget -O ${model}.zip https://duke.app.box.com/shared/static/2pfagrs17mtcw2os2roc66svg9fs56j2
        mv ${model}.zip .${model}.zip
    fi
    echo "Unpacking pretrained evaluation models"
    unzip .${model}.zip
fi

model=ser
if [ ! -d "exp/$model" ]; then
    if [ ! -f .${model}.zip ]; then
        echo "Download pretrained $model models pre-trained..."
        wget -O ${model}.zip https://duke.app.box.com/shared/static/0j6t1pyjm8zkjnifee7v8q3o2zvk7mkg
        mv ${model}.zip .${model}.zip
    fi
    echo "Unpacking pretrained evaluation models"
    unzip .${model}.zip
fi

if [ ! -d "data/IEMOCAP/wav/Session1" ]; then
    if [ ! -z $iemocap_corpus ]; then
        if [ -d $iemocap_corpus/Session1 ]; then
            echo "Linking '$iemocap_corpus' to 'data/IEMOCAP/wav'"
            ln -s $iemocap_corpus data/IEMOCAP/wav
        else
          echo "iemocap_corpus is defined to '$iemocap_corpus', but '$iemocap_corpus/Session1' does not exists."
          echo "Please fix your path to iemocap_corpus in the $0 script."
          exit 1
        fi
    fi
fi

model=asv_ssl
if [ ! -d "exp/$model" ]; then
    mkdir -p exp
    cd exp
    if [ ! -f .${model}.zip ]; then
        echo "Download pretrained $model models pre-trained..."
        wget -O ${model}.zip https://duke.app.box.com/shared/static/wo07rwi5xcgam1v6qlzfnzb536rqug14
        mv ${model}.zip .${model}.zip
    fi
    echo "Unpacking pretrained evaluation models"
    unzip .${model}.zip
    cd ../
fi


# IEMOCAP_full_release
if [ ! -d "data/IEMOCAP/wav/Session1" ]; then
    mkdir -p ./data/IEMOCAP/
    cat << EOF
==============================================================================
    Plase download or link the IEMOCAP corpus to './data/IEMOCAP/wav'
      - Download IEMOCAP from its web-page (license agreement is required)
          - https://sail.usc.edu/iemocap/
      - Link
          - ln -s YOUR_PATH data/IEMOCAP/wav/
==============================================================================
EOF
exit 1
fi
