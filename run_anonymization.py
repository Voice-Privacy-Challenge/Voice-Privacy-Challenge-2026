# We need to set CUDA_VISIBLE_DEVICES before we import Pytorch, so we will read all arguments directly on startup
from argparse import ArgumentParser
import os
parser = ArgumentParser()

parser.add_argument('--config', default='anon_config.yaml')
parser.add_argument('--gpu_ids', default='0')
parser.add_argument('--force_compute', default=False, type=bool)
args = parser.parse_args()

if 'CUDA_VISIBLE_DEVICES' not in os.environ:  # do not overwrite previously set devices
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
else: # CUDA_VISIBLE_DEVICES more important than the gpu_ids arg
    args.gpu_ids = ",".join([ str(i) for i, _ in enumerate(os.environ['CUDA_VISIBLE_DEVICES'].split(","))])

from pathlib import Path
import subprocess
import sys
import torch

# Apply torchaudio compatibility patch before importing speechbrain
import torchaudio
if not hasattr(torchaudio, 'list_audio_backends'):
    def list_audio_backends():
        return ['soundfile', 'sox', 'ffmpeg']
    torchaudio.list_audio_backends = list_audio_backends

from utils import parse_yaml, get_datasets, check_dependencies, setup_logger

logger = setup_logger(__name__)

def shell_run(cmd):
    if subprocess.run(['bash', cmd]).returncode != 0:
        logger.error(f'Failed to bash execute: {cmd}')
        sys.exit(1)

if __name__ == '__main__':

    config = parse_yaml(Path(args.config))
    datasets = get_datasets(config)

    gpus = args.gpu_ids.split(',')

    devices = []
    if torch.cuda.is_available():
        for gpu in gpus:
            devices.append(torch.device(f'cuda:{gpu}'))
    else:
        devices.append(torch.device('cpu'))

    if config['pipeline'] == "mcadams":
        from anonymization.pipelines.mcadams import McAdamsPipeline as pipeline
    elif config['pipeline'] == "sttts":
        shell_run('anonymization/pipelines/sttts/install.sh')
        check_dependencies('anonymization/pipelines/sttts/requirements.txt')
        if "download_precomputed_intermediate_repr" in config and config["download_precomputed_intermediate_repr"]:
            shell_run('anonymization/pipelines/sttts/download_precomputed_intermediate_repr.sh')
        from anonymization.pipelines.sttts import STTTSPipeline as pipeline
    elif config['pipeline'] == "sttts_multi":
        shell_run('anonymization/pipelines/sttts_multi/install.sh')
        check_dependencies('anonymization/pipelines/sttts_multi/requirements.txt')
        if "download_precomputed_intermediate_repr" in config and config["download_precomputed_intermediate_repr"]:
            shell_run('anonymization/pipelines/sttts_multi/download_precomputed_intermediate_repr.sh')
        from anonymization.pipelines.sttts_multi import STTTSMultiPipeline as pipeline
    elif config['pipeline'] == "nac":
        shell_run('anonymization/pipelines/nac/install.sh')
        sys.path.append(os.path.join(os.path.dirname(__file__), 'anonymization/modules/nac/coqui_tts/'))
        if devices[0] == torch.device('cpu'):
            from anonymization.pipelines.nac.nac_pipeline import NACPipeline as pipeline
        else:
            from anonymization.pipelines.nac.nac_pipeline_accelerate import NACPipeline as pipeline
    elif config['pipeline'] == "asrbn":
        shell_run('anonymization/pipelines/asrbn/install.sh')
        check_dependencies('anonymization/pipelines/asrbn/requirements.txt')
        from anonymization.pipelines.asrbn import ASRBNPipeline as pipeline
    elif config['pipeline'] == "ssl":
        shell_run('anonymization/pipelines/ssl/install.sh')
        check_dependencies('anonymization/pipelines/ssl/requirements.txt')
        from anonymization.pipelines.ssl.ssl_pipeline import SSLPipeline as pipeline
    elif config['pipeline'] == "template":
        from anonymization.pipelines.template import TemplatePipeline as pipeline
    else:
        raise ValueError(f"Pipeline {config['pipeline']} not defined/imported")

    logger.info(f'Running pipeline: {config["pipeline"]}')
    p = pipeline(config=config, force_compute=args.force_compute, devices=devices)
    p.run_anonymization_pipeline(datasets)
