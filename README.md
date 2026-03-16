# Recipe for VoicePrivacy Challenge 2026

Please visit the [challenge website](https://www.voiceprivacychallenge.org/) for more information about the Challenge.

## Install

1. `git clone https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2026.git`
2. `./00_install.sh`
3. `source env.sh`


## Tracks

<details>
<summary><b>Track 1</b></summary>

1. Download data and models:
```bash 01_download_data_model_track1.sh```

2. Run Track 1 (semi-informed EER, WER, UAR): ```02_run_track1.sh```

> [!IMPORTANT]  
> The [IEMOCAP](https://sail.usc.edu/iemocap/iemocap_release.htm) corpus must be downloaded on your own by submitting a request at https://sail.usc.edu/iemocap/iemocap_release.htm. The waiting time may take up to 7-9 days.

## Anonymization and Evaluation

1. Run anonymization and evaluation: `./02_run_track1.sh configs/track1/anon_mcadams.yaml`.  
    For each anonymization baseline, there is a corresponding config file:
    -  #### [Anonymization using the McAdams coefficient](https://arxiv.org/abs/2011.01130): **B2**
         [`configs/track1/anon_mcadams.yaml`](configs/track1/anon_mcadams.yaml)  A fast CPU-only signal-processing-based system  (default).

    -  #### [Anonymization using phonetic transcriptions and GAN (STTTS)](https://ieeexplore.ieee.org/document/10096607): **B3**
         [`configs/track1/anon_sttts.yaml`](configs/track1/anon_sttts.yaml)  A system based on an unmodified phone sequence, modified prosody, modified speaker embedding representations and speech synthesis.

    -  #### [Anonymization using neural audio codec (NAC) language modeling](https://arxiv.org/abs/2309.14129): **B4**

        [`configs/track1/anon_nac.yaml`](configs/track1/anon_nac.yaml) 

    -  #### [Anonymization using ASR-BN with vector quantization (VQ)](https://arxiv.org/abs/2308.04455): **B5** 

        [`configs/track1/anon_asrbn.yaml`](configs/track1/anon_asrbn.yaml) A fast system based on vector-quantized acoustic bottleneck, pitch, and one-hot speaker representations, and a HiFi-GAN speech synthesis model

To run anonymization and evaluation separately, please refer to [the anonymization and evaluation sections in the VPC2024 GitHub README.](https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2024/blob/main/README.md#:~:text=7%2D9%20days.-,Anonymization%20and%20Evaluation,-There%20are%20two).

## Results
#### Note, that WER results are computed on the trials part
The result file with all the metrics and all datasets for submission will be generated in:
* Summary results: `./exp/results_summary/track1/result_for_rank$anon_data_suffix`
* Additional information for submission: `./exp/results_summary/track1/result_for_submission${anon_data_suffix}.zip`

Please see the [RESULTS folder](./results/track1) for the provided anonymization baselines:

* [Results B2](./results/track1/result_for_rank_mcadams)
* [Results B3](./results/track1/result_for_rank_sttts)
* [Results B4](./results/track1/result_for_rank_nac)
* [Results B5](./results/track1/result_for_rank_asrbn_hifigan_bn_tdnnf_wav2vec2_vq_48_v1)
  
</details>

<details>
<summary><b>Track 2</b></summary>
    
1. Download data and models: ```bash 01_download_data_model_track2.sh```

2. Run Track 2 (semi-informed EER, WER, UAR): ```bash 02_run_track2.sh ```


## Anonymization and Evaluation
There are two options:
1. Run anonymization and evaluation: `./02_run_track2.sh configs/track2/anon_BM1.yaml`.  
    For each anonymization baseline, there is a corresponding config file:
    -  #### [Anonymization using self-supervised learning](https://arxiv.org/abs/2203.14834): **BM1**
         [`configs/track2/anon_BM1.yaml`](configs/track2/anon_BM1.yaml)  A system based on content, prosody, modified speaker embedding representations and speech synthesis  (default).

    -  #### [Anonymization using phonetic transcriptions and GAN](https://arxiv.org/abs/2407.02937): **BM2 and BM3**
         [`configs/track2/anon_BM2.yaml`](configs/track2/anon_BM2.yaml)  A system based on an unmodified phone sequence, modified prosody,             GAN-generated artificial speaker embeddings, and speech synthesis with IMS Toucan + HiFi-GAN.
       
         [`configs/track2/anon_BM3.yaml`](configs/track2/anon_BM3.yaml)  Compared to BM2, removes the prosody extractor and does not feed F0 into the synthesis model, thus keeping the original prosody.
       

      
2. Run anonymization and evaluation separately in two steps:

#### Step 1: Anonymization
```sh
python run_anonymization.py --config configs/track2/anon_BM1.yaml  

```
The anonymized audios will be saved in `$data_dir=data` into 30 folders corresponding to datasets.
The names of the created dataset folders for anonymized audio files are appended with the suffix, i.e. `$anon_data_suffix=_BM1`

```log

data/en_dev_enrolls${anon_data_suffix}/wav/*wav
data/en_dev_trials_mixed${anon_data_suffix}/wav/*wav
data/en_test_enrolls${anon_data_suffix}/wav/*wav
data/en_test_trials_mixed${anon_data_suffix}/wav/*wav

data/es_dev_enrolls${anon_data_suffix}/wav/*wav
data/es_dev_trials_mixed${anon_data_suffix}/wav/*wav
data/es_test_enrolls${anon_data_suffix}/wav/*wav
data/es_test_trials_mixed${anon_data_suffix}/wav/*wav

data/fr_dev_enrolls${anon_data_suffix}/wav/*wav
data/fr_dev_trials_mixed${anon_data_suffix}/wav/*wav
data/fr_test_enrolls${anon_data_suffix}/wav/*wav
data/fr_test_trials_mixed${anon_data_suffix}/wav/*wav

data/de_dev_enrolls${anon_data_suffix}/wav/*wav
data/de_dev_trials_mixed${anon_data_suffix}/wav/*wav
data/de_test_enrolls${anon_data_suffix}/wav/*wav
data/de_test_trials_mixed${anon_data_suffix}/wav/*wav

data/emodata_track2_dev${anon_data_suffix}/wav/*wav
data/emodata_track2_test${anon_data_suffix}/wav/*wav

data/train_english${anon_data_suffix}/wav/*wav
data/train_spanish${anon_data_suffix}/wav/*wav
data/train_french${anon_data_suffix}/wav/*wav
data/train_german${anon_data_suffix}/wav/*wav

data/cn_dev_enrolls${anon_data_suffix}/wav/*wav
data/cn_dev_trials_mixed${anon_data_suffix}/wav/*wav
data/cn_test_enrolls${anon_data_suffix}/wav/*wav
data/cn_test_trials_mixed${anon_data_suffix}/wav/*wav

data/ja_dev_enrolls${anon_data_suffix}/wav/*wav
data/ja_dev_trials_mixed${anon_data_suffix}/wav/*wav
data/ja_test_enrolls${anon_data_suffix}/wav/*wav
data/ja_test_trials_mixed${anon_data_suffix}/wav/*wav

```

For the next evaluation step, you should replicate the corresponding directory structure when developing your anonymization system.  

#### Step 2: Evaluation
2. perform evaluations
   
```sh
python run_evaluation.py --config configs/track2/eval_pre.yaml --overwrite "{\"anon_data_suffix\": \"$anon_data_suffix\"}" --force_compute True

python run_evaluation.py --config configs/track2/eval_post_en.yaml --overwrite "{\"anon_data_suffix\": \"$anon_data_suffix\"}" --force_compute True
python run_evaluation.py --config configs/track2/eval_post_de.yaml --overwrite "{\"anon_data_suffix\": \"$anon_data_suffix\"}" --force_compute True
python run_evaluation.py --config configs/track2/eval_post_es.yaml --overwrite "{\"anon_data_suffix\": \"$anon_data_suffix\"}" --force_compute True
python run_evaluation.py --config configs/track2/eval_post_fr.yaml --overwrite "{\"anon_data_suffix\": \"$anon_data_suffix\"}" --force_compute True
```


> All of the above steps are automated in [02_run_track2.sh](./02_run_track2.sh).


## Results
#### Note, that WER results are computed on the trials part
The result file with all the metrics and all datasets for submission will be generated in:
* Summary results: `./exp/results_summary/track2/result_for_rank$anon_data_suffix`

Please see the [RESULTS folder](./results/track2) for the provided anonymization baselines:

* [Results BM1](./results/track2/result_for_rank_BM1)
* [Results BM2](./results/track2/result_for_rank_BM2)
* [Results BM3](./results/track2/result_for_rank_BM3)



</details>


<details>
<summary><b>Runtime Summary</b></summary>

#### Track 1

| Script | Description | B2 | B3 | B4 | B5 |
|--------|-------------|----|----|----|----|
| `run_anonymization.py` | Generate anonymized audio (LibriSpeech dev+test, IEMOCAP dev+test, LibriSpeech-train-clean-360) | ~2h | ~13h | ~72h | ~1h |
| `run_evaluation.py` (eval_pre.yaml) | ASR/ASV/SER on LibriSpeech dev+test & IEMOCAP dev+test using pretrained models | | | | |
| `run_evaluation.py` (eval_post.yaml) | Train semi-informed ASV using anonymized LibriSpeech-train-clean-360h, then evaluate on LibriSpeech dev+test |～10h | | | |

#### Track 2

| Script | Description | BM1 | BM2 | BM3 |
|--------|-------------|-----|-----|-----|
| `run_anonymization.py` | Generate anonymized audio (multilingual dev+test, emodata\_track2) | ~9h | >1 day | >1 day |
| `run_evaluation.py` (eval_pre.yaml) | ASR (Whisper large-v3), ASV (asv\_ssl), SER (emotion2vec) on multilingual dev+test & emodata\_track2 | ori-asr-3.5h, anon-asr-3.5h | asv-0.5h | |
| `run_anonymization.py` (anon\_post\_*.yaml) | Generate anonymized audio on multilingual training set (for semi-informed attacker) | ~11h|~22h | |
| `run_evaluation.py` (eval\_post\_en.yaml) | Train semi-informed ASV using anonymized MLS-en data, then evaluate on MLS-en-dev+test | 50min/epoch * 4 epochs| | |
| `run_evaluation.py` (eval\_post\_de.yaml) | Train semi-informed ASV using anonymized MLS-de data, then evaluate on MLS-de-dev+test | 33min/epoch * 10epochs| | |
| `run_evaluation.py` (eval\_post\_fr.yaml) | Train semi-informed ASV using anonymized MLS-fr data, then evaluate on MLS-fr-dev+test | 15min/epoch * 10epochs| | |
| `run_evaluation.py` (eval\_post\_es.yaml) | Train semi-informed ASV using anonymized MLS-es data, then evaluate on MLS-es-dev+test | 12min/epoch * 10epochs| | |


</details>



## Data submission (To be updated)


## General information

#### Evaluation plan
For more details about the baseline and data, please see The [VoicePrivacy 2026 Challenge Evaluation Plan](https://www.voiceprivacychallenge.org/vp2026/docs/VPC_2026_march15.pdf)

#### Training data (To be updated)

#### Registration
Participants are requested to register for the evaluation. Registration should be performed once only for each participating entity using the following form: **[Registration](https://forms.office.com/r/74BKeXBWTZ)**.


## Organizers
- Xiaoxiao Miao - Duke Kunshan University, China
- Natalia Tomashenko - Université de Lorraine, CNRS, Inria, LORIA, F-54000 Nancy, France
- Ridwan Arefeen - Singapore Institute of Technology, Singapore
- Sarina Meyer - University of Stuttgart, Germany
- Michele Panariello - EURECOM, France
- Xin Wang - National Institute of Informatics, Japan
- Emmanuel Vincent - Université de Lorraine, CNRS, Inria, LORIA, F-54000 Nancy, France
- Junichi Yamagishi - National Institute of Informatics, Japan
- Nicholas Evans - EURECOM, France
- Massimiliano Todisco - EURECOM, France

Contact: organisers@lists.voiceprivacychallenge.org

## License

Copyright (C) 2026

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.


