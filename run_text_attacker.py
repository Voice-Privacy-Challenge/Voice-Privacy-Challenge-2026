# Evaluation of the text attacker (speaker verification from transcripts only), see evaluation/privacy/text_attacker/README.md
# We need to set CUDA_VISIBLE_DEVICES before we import Pytorch, so we will read all arguments directly on startup
import os
import json
import time
import itertools
from argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser()
parser.add_argument('--config', default='configs/track1/eval_text.yaml')
parser.add_argument('--overwrite', type=str, default='{}')
parser.add_argument('--force_compute', type=str, default='False')
parser.add_argument('--gpu_ids', default='0')
args = parser.parse_args()

if 'CUDA_VISIBLE_DEVICES' not in os.environ:  # do not overwrite previously set devices
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

import torch

from utils import parse_yaml, setup_logger
from evaluation.privacy.text_attacker import evaluate_text_attacker

logger = setup_logger(__name__)


if __name__ == '__main__':
    params = parse_yaml(Path(args.config), overrides=json.loads(args.overwrite))
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    eval_data_dir = params['data_dir']
    anon_suffix = params['anon_data_suffix']

    logger.info('================================')
    logger.info('Perform text attacker evaluation')
    logger.info('================================')
    start_time = time.time()
    eval_data_name = params['privacy']['text_attacker']['dataset_name']
    eval_pairs = []
    for name in eval_data_name:
        for d in params['datasets']:
            if d['name'] != name:
                continue
            eval_pairs.extend([(f'{d["data"]}{enroll}', f'{d["data"]}{trial}')
                               for enroll, trial in itertools.product(d['enrolls'], d['trials'])])
    evaluate_text_attacker(eval_datasets=eval_pairs, eval_data_dir=eval_data_dir,
                           params=params['privacy']['text_attacker'], device=device, anon_data_suffix=anon_suffix,
                           force_compute=args.force_compute.lower() == 'true')
    logger.info("--- Text attacker evaluation time: %f min ---" % (float(time.time() - start_time) / 60))
