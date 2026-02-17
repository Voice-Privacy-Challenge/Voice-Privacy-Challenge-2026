#!/bin/bash

# Fresh install with "rm .micromamba/micromamba .done-*"

set -e

nj=$(nproc)
home=$PWD
env_sh=env_sttts_multi.sh

venv_dir=$PWD/venv_sttts_multi
source ./env.sh
touch $env_sh
source $env_sh
export MAMBA_ROOT_PREFIX=".micromamba"  # Local install of micromamba (where the libs/bin will be cached)
mamba_bin="$MAMBA_ROOT_PREFIX/micromamba"

# Core packages (essential, no problematic link scripts)
MAMBA_CORE_PACKAGES="sshpass OpenSSH sox libflac tar libacl inotify-tools git-lfs wget curl make cmake ncurses ninja python=3.11 automake libtool boost gxx=12.3.0 gcc=12.3.0 python-sounddevice pkg-config zip"
# Optional packages that may have link script issues (will be installed separately or skipped)
MAMBA_OPTIONAL_PACKAGES="ocl-icd-system nvtop ffmpeg"

ESPAK_VERSION=1.51.1
CUDA_VERSION=12.8
TORCH_VERSION=2.8.0

compute_and_write_hash "anonymization/pipelines/sttts_multi/requirements.txt"
trigger_new_install "exp/sttts_multi_models" #"exp/sttts_multi_models .done-*-sttts-multi .done-sttts-multi-requirements" # .done-espeak"

\rm $env_sh 2> /dev/null || true
touch $env_sh

# Download GAN pre-models only if perform GAN anonymization
if [ ! -d exp/sttts_multi_models ]; then
    echo "Download pretrained models of GAN-based speaker anonymization system..."
    mkdir -p exp/sttts_multi_models
    wget -q -O exp/sttts_multi_models/embedding_gan.pt https://github.com/DigitalPhonetics/IMS-Toucan/releases/download/v3.0/embedding_gan.pt
    wget -q -O exp/sttts_multi_models/ToucanTTS_Meta.pt https://github.com/DigitalPhonetics/IMS-Toucan/releases/download/v3.1/ToucanTTS_Meta.pt
    wget -q -O exp/sttts_multi_models/Vocoder.pt https://github.com/DigitalPhonetics/IMS-Toucan/releases/download/v3.1.1/Vocoder.pt
    wget -q -O exp/sttts_multi_models/aligner.pt https://github.com/DigitalPhonetics/IMS-Toucan/releases/download/v3.1/aligner.pt
fi

# Deactivate previous venv
echo "micromamba deactivate" >> $env_sh
source ./$env_sh

# Make new venv for this anonymization pipeline
mark=.done-venv-sttts-multi
if [ ! -f $mark ]; then
  echo " == Making virtual environment =="
  "$mamba_bin" create -y --prefix "$venv_dir"

  # Step 1: Install core packages (excluding those that pull in gdk-pixbuf)
  echo "Step 1: Installing core packages..."
  "$mamba_bin" install -y --prefix "$venv_dir" -c conda-forge --override-channels $MAMBA_CORE_PACKAGES || exit 1

  # Verify Python is installed and working
  if [ ! -f "$venv_dir/bin/python" ]; then
    echo "Error: Python installation failed"
    exit 1
  fi
  "$venv_dir/bin/python" --version || exit 1

  # Step 2: Try to install optional packages (may fail due to link script errors, but that's OK)
  echo "Step 2: Installing optional packages (may have link script warnings)..."
  set +e
  # Try to install optional packages, but don't fail if link script errors occur
  "$mamba_bin" install -y --prefix "$venv_dir" -c conda-forge --override-channels $MAMBA_OPTIONAL_PACKAGES 2>&1 | tee /tmp/mamba_optional.log | grep -v "libmamba failed to execute pre/post link script" || true
  optional_status=${PIPESTATUS[0]}
  set -e


  # Check if the failure was due to link script errors or something else
  if [ $optional_status -ne 0 ]; then
    if grep -q "libmamba failed to execute pre/post link script" /tmp/mamba_optional.log 2>/dev/null; then
      failed_packages=$(grep "libmamba failed to execute pre/post link script" /tmp/mamba_optional.log | sed 's/.*for //' | sort -u | tr '\n' ' ')
      echo "Warning: Optional packages installation failed due to link script errors for: $failed_packages"
      echo "Core packages are installed. Optional packages can be installed via system package manager if needed."
    else
      echo "Warning: Optional packages installation had issues, but core packages are installed. Continuing..."
    fi
  fi

  touch $mark
fi

echo "micromamba activate $venv_dir" >> $env_sh
echo "export LD_LIBRARY_PATH=$venv_dir/lib/:$LD_LIBRARY_PATH" >> $env_sh
source ./$env_sh

mark=.done-cuda-sttts-multi
if [ ! -f $mark ]; then
  echo " == Installing cuda =="
  # For CUDA 12.x, use the appropriate channel format
  micromamba install -y --prefix "$venv_dir" -c nvidia -c conda-forge cuda-toolkit==${CUDA_VERSION} || exit 1
  "$venv_dir/bin/nvcc" --version || exit 1
  touch $mark
fi

CUDAROOT=$venv_dir
echo "export CUDAROOT=$CUDAROOT" >> $env_sh
source ./$env_sh


cuda_version_without_dot=$(echo $CUDA_VERSION | xargs | sed 's/\.//')
mark=.done-pytorch-sttts-multi
if [ ! -f $mark ]; then
  echo " == Installing pytorch $TORCH_VERSION for cuda $CUDA_VERSION =="
  # Determine PyTorch CUDA index based on version
  # PyTorch 2.8.0 supports cu126, cu128, cu129 (not cu121)
  # For CUDA 12.8, use cu128 index
  if [[ "$TORCH_VERSION" == "2.8."* ]] && [[ "$CUDA_VERSION" == "12.8" ]]; then
    pytorch_cuda="128"  # PyTorch 2.8.x uses cu128 for CUDA 12.8
  elif [[ "$TORCH_VERSION" == "2.9."* ]] && [[ "$CUDA_VERSION" == "12.8" ]]; then
    pytorch_cuda="128"  # PyTorch 2.9.x uses cu128
  else
    pytorch_cuda="$cuda_version_without_dot"
  fi

  version="==$TORCH_VERSION+cu$pytorch_cuda"
  # Match torchvision version to PyTorch version
  # PyTorch 2.8.0 uses torchvision 0.23.0+cu128 (0.19.0 not available in cu128)
  if [[ "$TORCH_VERSION" == "2.8."* ]]; then
      torchvision_version="==0.23.0+cu$pytorch_cuda"
      torchcodec_version="==0.7"
  else
    torchvision_version="==0.24.1+cu$pytorch_cuda"
  fi
  torchaudio_version="==$TORCH_VERSION+cu$pytorch_cuda"
  echo -e "\npip3 install torch$version torchvision$torchvision_version torchaudio$torchaudio_version torchcodec${torchcodec_version} --force-reinstall --index-url https://download.pytorch.org/whl/cu$pytorch_cuda\n"
  pip3 install torch$version torchvision$torchvision_version torchaudio$torchaudio_version torchcodec${torchcodec_version} --force-reinstall --index-url https://download.pytorch.org/whl/cu$pytorch_cuda \
    || { echo "Failed to find pytorch $TORCH_VERSION for cuda '$CUDA_VERSION', use specify other torch/cuda version (with variables in install.sh script)"  ; exit 1; }
  python3 -c "import torch; print('Torch version:', torch.__version__)" || exit 1
  echo -e "torch$version\ntorchvision$torchvision_version\ntorchaudio$torchaudio_version" > .pip_constraints.txt
  touch $mark
fi


mark=.done-sttts-multi-requirements
if [ ! -f $mark ]; then
  echo " == Installing STTTS python libraries =="
  pip3 install -r anonymization/pipelines/sttts_multi/requirements.txt  || exit 1
  touch $mark
fi


mark=.done-espeak
if [ ! -f $mark ]; then
  echo " == Installing G2P espeak-ng =="
  wget https://github.com/espeak-ng/espeak-ng/archive/$ESPAK_VERSION/espeak-ng-$ESPAK_VERSION.tar.gz
  \rm espeak-ng-$ESPAK_VERSION -rf || true
  tar -xvzf ./espeak-ng-$ESPAK_VERSION.tar.gz
  \rm ./espeak-ng-$ESPAK_VERSION.tar.gz
  cd espeak-ng-$ESPAK_VERSION
  ./autogen.sh || true # First time fails?
  ./autogen.sh

  sed -i "s|.*define PATH_ESPEAK_DATA.*|\#define PATH_ESPEAK_DATA \"${venv_dir}/share/espeak-ng-data\"|" src/libespeak-ng/speech.h
  sed -i "58d" src/libespeak-ng/speech.h
  sed -i "59d" src/libespeak-ng/speech.h

  ./configure
  make -j $nj src/espeak-ng src/speak-ng
  make -j $nj

  make DESTDIR="$venv_dir" install

  yes | cp -rf ${venv_dir}/usr/local/* ${venv_dir} || true

  # espeak-ng --voices
  pip3 install phonemizer
  python3 -c "import phonemizer; phonemizer.phonemize('Good morning', language='en-gb')"
  python3 -c "import phonemizer; phonemizer.phonemize('Guten Morgen', language='de')"
  python3 -c "import phonemizer; phonemizer.phonemize('Bonjour', language='fr-fr')"

  cd $home
  touch $mark
fi
