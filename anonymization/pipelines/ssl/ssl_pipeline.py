#!/usr/bin/env python3.0
# -*- coding: utf-8 -*-

from pathlib import Path
import torch
from ...modules.ssl.anonymizer import SelectionBasedAnonymizationPipeline

from .. import Pipeline, get_anon_level_from_config
from utils import setup_logger

logger = setup_logger(__name__)

class SSLPipeline(Pipeline):
    def __init__(self, config: dict, force_compute: bool = False, devices: list = [0]):
        """
        Instantiates a SSLPipeline.

        This pipeline consists of:
                  ->    F0 (yaapt: no transformation) 
            input ->    Soft HuBERT                  
                  ->    Speaker embedding extraction + k-anonymization 
                  
                 --->  Speech synthesis (hifigan)                                  ^
                                             
        Args:
            config (dict): a configuration dictionary, e.g., see anon_ssl.yaml
            force_compute (bool): if True, forces re-computation of
                all steps. otherwise uses saved results.
            devices (list): a list of torch-interpretable devices
        """
        self.config = config
        self.gpu_devices = devices
        self.force_compute = force_compute
        self.modules_config = config['modules']

        self.pipeline = SelectionBasedAnonymizationPipeline(
            device='cuda' if torch.cuda.is_available() else 'cpu',
            default_world_size=self.modules_config.get('world_size', 50),
            default_region_size=self.modules_config.get('region_size', 10),
            default_flag_proximity=self.modules_config.get('flag_proximity', 'random'),
            default_flag_cross_gender=self.modules_config.get('flag_cross_gender', False),
            default_gender_pool=self.modules_config.get('gender_pool', False),
        )

    

    def run_anonymization_pipeline(self, datasets):
        # anonymize each dataset
        for i, (dataset_name, dataset_path) in enumerate(datasets.items()):
            anon_level = get_anon_level_from_config(self.modules_config, dataset_name)
            print(f'{i + 1}/{len(datasets)}: SSL processing of "{dataset_name}" at anon_level "{anon_level}"...')
            self.pipeline.run(dataset_path=dataset_path,
                         anon_level=anon_level,
                         results_dir=self.config['results_dir'],
                         settings=self.modules_config,
                         force_compute=self.force_compute,
                         )
