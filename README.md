# Recipe for VoicePrivacy Challenge 2026

Please visit the [challenge website](https://www.voiceprivacychallenge.org/) for more information about the Challenge.

## Install

1. `git clone https://github.com/xiaoxiaomiao323/vpc2026-dev.git`
2. `./00_install.sh`
3. `source env.sh`


## Tracks

<details>
<summary><b>Track 1</b></summary>

1. Download data and models:
```bash 01_download_data_model_track1.sh```

2. Run Track 1 (semi-informed EER, ASR, UAR): ```02_run_track1.sh```

> [!IMPORTANT]  
> The [IEMOCAP](https://sail.usc.edu/iemocap/iemocap_release.htm) corpus must be downloaded on your own by submitting a request at https://sail.usc.edu/iemocap/iemocap_release.htm. The waiting time may take up to 7-9 days.


## Anonymization and Evaluation
There are two options:
1. Run anonymization and evaluation: `./02_run_track1.sh configs/track1/anon_mcadams.yaml`.  
    For each anonymization baseline, there is a corresponding config file:
    -  #### [Anonymization using the McAdams coefficient](https://arxiv.org/abs/2011.01130): **B2**
         [`configs/track1/anon_mcadams.yaml`](configs/track1/anon_mcadams.yaml)  A fast CPU-only signal processing-based system  (default).

    -  #### [Anonymization using phonetic transcriptions and GAN (STTTS)](https://ieeexplore.ieee.org/document/10096607): **B3**
         [`configs/track1/anon_sttts.yaml`](configs/track1/anon_sttts.yaml)  A system based on unmodified phone sequence, modified prosody, modified speaker embedding representations and speech synthesis.

    -  #### [Anonymization using **n**eural audio codec (NAC) language modeling](https://arxiv.org/abs/2309.14129): **B4**

        [`configs/track1/anon_nac.yaml`](configs/track1/anon_nac.yaml) 

    -  #### [Anonymization using ASR-BN with vector quantization (VQ)](https://arxiv.org/abs/2308.04455): **B5** 

        [`configs/track1/anon_asrbn.yaml`](configs/track1/anon_asrbn.yaml) A fast system based on vector quantized acoustic bottleneck, pitch, and one-hot speaker representations and  a HiFi-GAN speech synthesis model.
    
      
2. Run anonymization and evaluation separately in two steps:

#### Step 1: Anonymization
```sh
python run_anonymization.py --config configs/track1/anon_mcadams.yaml  #Computational time varies from 30 minutes to 10 hours, depending on the number of cores, for other methods it may be longer and depending on the available hardware. 

```
The anonymized audios will be saved in `$data_dir=data` into 7 folders corresponding to datasets.
The names of the created dataset folders for anonymized audio files are appended with the suffix, i.e. `$anon_data_suffix=_mcadams`

```log
data/libri_dev_enrolls${anon_data_suffix}/wav/*wav
data/libri_dev_trials_mixed${anon_data_suffix}/wav/*wav

data/libri_test_enrolls${anon_data_suffix}/wav/*wav
data/libri_test_trials_mixed${anon_data_suffix}/wav/*wav

data/IEMOCAP_dev${anon_data_suffix}/wav/*wav
data/IEMOCAP_test${anon_data_suffix}/wav/*wav

data/train-clean-360${anon_data_suffix}/wav/*wav
```
For the next evaluation step, you should replicate the corresponding directory structure when developing your anonymization system.  

#### Step 2: Evaluation
Evaluation metrics include:
- Privacy: Equal error rate (EER) for ignorant, lazy-informed, and semi-informed attackers (only results from the semi-informed attacker will be used in the challenge ranking) 
- Utility:
  - Word Error Rate (WER) by an automatic speech recognition (ASR) model (trained on LibriSpeech)
  - Unweighted Average Recall (UAR) by a speech emotion recognition (SER) model (trained on IEMOCAP).


To run evaluation for arbitrary anonymized data:
1. prepare 7 anonymized folders each containing the anonymized wav files:
```log
data/libri_dev_enrolls${anon_data_suffix}/wav/*wav
data/libri_dev_trials_mixed${anon_data_suffix}/wav/*wav

data/libri_test_enrolls${anon_data_suffix}/wav/*wav
data/libri_test_trials_mixed${anon_data_suffix}/wav/*wav

data/IEMOCAP_dev${anon_data_suffix}/wav/*wav
data/IEMOCAP_test${anon_data_suffix}/wav/*wav

data/train-clean-360${anon_data_suffix}/wav/*wav
```
2. perform evaluations
   
```sh
python run_evaluation.py --config configs/track1/eval_pre.yaml --overwrite "{\"anon_data_suffix\": \"$anon_data_suffix\"}" --force_compute True
python run_evaluation.py --config configs/track1/eval_post.yaml --overwrite "{\"anon_data_suffix\": \"$anon_data_suffix\"}" --force_compute True
```

3. get the final results for ranking
```sh
TODO
```

> All of the above steps are automated in [02_run_track1.sh](./02_run_track1.sh).

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

2. Run Track 2 (lazy-informed EER, ASR, UAR): ```bash 02_run_track2.sh ```

3. Run Track 2 post to generate anonymized data:```bash 02_run_track2_post.sh```

## Anonymization and Evaluation
There are two options:
1. Run anonymization and evaluation: `./02_run_track2.sh configs/track2/anon_BM1.yaml`.  
    For each anonymization baseline, there is a corresponding config file:
    -  #### [Anonymization using self-supervised learning](https://arxiv.org/abs/2203.14834): **BM1**
         [`configs/track2/anon_BM1.yaml`](configs/track2/anon_BM1.yaml)  A system based on content, prosody, modified speaker embedding representations and speech synthesis  (default).

    -  #### [Anonymization using phonetic transcriptions and GAN)](https://arxiv.org/abs/2407.02937): **BM2 and BM3**
         [`configs/track2/anon_BM2.yaml`](configs/track1/anon_sttts.yaml)  A system based on unmodified phone sequence, modified prosody, modified speaker embedding representations and speech synthesis.

      
2. Run anonymization and evaluation separately in two steps:

#### Step 1: Anonymization
```sh
python run_anonymization.py --config configs/track2/anon_BM1.yaml  # anonymization time around 9 hours for BM1.

```
The anonymized audios will be saved in `$data_dir=data` into 18(4*4+2) folders corresponding to datasets.
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

```

For the next evaluation step, you should replicate the corresponding directory structure when developing your anonymization system.  

#### Step 2: Evaluation
Evaluation metrics include:
- Privacy: Equal error rate (EER) for ignorant, lazy-informed (only results from the lazy-informed attacker will be submitted) 
- Utility:
  - Word Error Rate (WER) by an automatic speech recognition (ASR) model (trained on LibriSpeech)
  - Unweighted Average Recall (UAR) by a speech emotion recognition (SER) model (trained on IEMOCAP).

> All of the above steps are automated in [02_run_track2.sh](./02_run_track2.sh).

#### Step 3: anonymized data generation for ranking

```sh
02_run_track2_post.sh  # you can only run once after you determine your anonymization systems 

```
The anonymized audios will be saved in `$data_dir=data` into 12 folders corresponding to datasets.
The names of the created dataset folders for anonymized audio files are appended with the suffix, i.e. `$anon_data_suffix=_mcadams`

```log
data/

data/ja_dev_enrolls${anon_data_suffix}/wav/*wav
data/ja_dev_trials_mixed${anon_data_suffix}/wav/*wav
data/ja_test_enrolls${anon_data_suffix}/wav/*wav
data/ja_test_trials_mixed${anon_data_suffix}/wav/*wav

data/cn_dev_enrolls${anon_data_suffix}/wav/*wav
data/cn_dev_trials_mixed${anon_data_suffix}/wav/*wav
data/cn_test_enrolls${anon_data_suffix}/wav/*wav
data/cn_test_trials_mixed${anon_data_suffix}/wav/*wav

data/train_english${anon_data_suffix}/wav/*wav
data/train_german${anon_data_suffix}/wav/*wav
data/train_french${anon_data_suffix}/wav/*wav
data/train_spanish${anon_data_suffix}/wav/*wav

```

## Results
#### Note, that WER results are computed on the trials part
The result file with all the metrics and all datasets for submission will be generated in:
* Summary results: `./exp/results_summary/track2/result_for_rank$anon_data_suffix`

Please see the [RESULTS folder](./results/track2) for the provided anonymization baselines:

* [Results BM1](./results/track2/result_for_rank_BM1)
* [Results BM2](./results/track2/result_for_rank_BM2)
* [Results BM3](./results/track2/result_for_rank_BM3)



</details>







## Data submission
The anonymization and evaluation scripts should have generated the files and the directories with the explained format of `$anon_data_suffix` suffix.  
For data submission, the following command submit everything given a `$anon_data_suffix` argument:
```
VPC_DROPBOX_KEY=XXX VPC_DROPBOX_SECRET=YYY VPC_DROPBOX_REFRESHTOKEN=ZZZ VPC_TEAM=TEAM_NAME ./03_upload_submission.sh $anon_data_suffix
```
`VPC_DROPBOX_KEY`, `VPC_DROPBOX_SECRET`, `VPC_DROPBOX_REFRESHTOKEN`, and `VPC_TEAM=TEAM_NAME` are sent individually to each team upon receiving their system description.  






## General information

#### Evaluation plan
For more details about the baseline and data, please see [The VoicePrivacy 2024 Challenge Evaluation Plan](https://www.voiceprivacychallenge.org/docs/VoicePrivacy_2024_Eval_Plan_v2.0.pdf) - Updated on 1st April 2024

#### Training data
[Final list of models and data](https://www.voiceprivacychallenge.org/docs/VoicePrivacy_2024_Challenge_Final_list_of_models_and_data_for_training_anonymization_systems_-_26.03.2024.pdf) for training anonymization systems.

#### Registration
Participants are requested to register for the evaluation. Registration should be performed once only for each participating entity using the following form: **[Registration](https://forms.office.com/r/T2ZHD1p3UD)**.

## Some potential questions you may have and how to solve them:
> 1. $ASV_{eval}^{anon}$ training is slow

Training of the $ASV_{eval}^{anon}$ model may vary from about 2 up to 10 hours depending on the available hardware.
If you have an SSD or a high-performance drive, $ASV_{eval}^{anon}$ takes ~2h, but if the drive is old and slow, in a worse case,  $ASV_{eval}^{anon}$ training takes ~10h. Increasing $num_workers in config/eval_post.yaml may help to speed up the processing.

> 2. OOM problem when decoding by $ASR_{eval}$

Reduce the $eval_bachsize in config/eval_pre.yaml

> 3. The $ASR_{eval}$ is a [pretrained wav2vec+ctc trained on LibriSpeech-960h](https://huggingface.co/speechbrain/asr-wav2vec2-librispeech)

> 4. Error on `utils.prepare_results_in_kaldi_format`

means something bad happened when running the anonymization pipeline.  
Remove all `data/*$anon_data_suffix` directories and re-run anonymization and evaluation steps (if `$anon_data_suffix=suff`, also remove the directories that share a matching suffix: `$anon_data_suffix=something_suff`). Check that your anonymization pipeline produces a wav file for each dataset entry, every original wav should have its anonymized counterpart.  



## Organizers (in alphabetical order)

- Pierre Champion - Inria, France
- Nicholas Evans - EURECOM, France
- Sarina Meyer - University of Stuttgart, Germany
- Xiaoxiao Miao - Singapore Institute of Technology, Singapore
- Michele Panariello - EURECOM, France
- Massimiliano Todisco - EURECOM, France
- Natalia Tomashenko - Inria, France
- Emmanuel Vincent - Inria, France
- Xin Wang - NII, Japan
- Junichi Yamagishi - NII, Japan

Contact: organisers@lists.voiceprivacychallenge.org

## License

Copyright (C) 2024

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

## Reference

```
@article{tomashenko2024voiceprivacy,
      title={The {VoicePrivacy} 2024 Challenge Evaluation Plan}, 
      author={Natalia Tomashenko and Xiaoxiao Miao and Pierre Champion and Sarina Meyer and Xin Wang and Emmanuel Vincent and Michele Panariello and Nicholas Evans and Junichi Yamagishi and Massimiliano Todisco},
      year={2024},
      eprint={2404.02677},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```

## Acknowledgments

Some parts of the code and structure are based on [VoicePAT](https://github.com/DigitalPhonetics/VoicePAT) (Paper: https://ieeexplore.ieee.org/document/10365329)
