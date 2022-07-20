# -*- coding: utf-8 -*-
# file: train_tc.py
# time: 21/06/2022
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import os
import random
import warnings

import findfile

# Transfer Experiments and Multitask Experiments

from pyabsa import TCTrainer, TADConfigManager, TCDatasetList, BERTTADModelList, TADTrainer, TCConfigManager, BERTTCModelList, GloVeTCModelList
from pyabsa.functional.dataset.dataset_manager import AdvTCDatasetList, DatasetItem

warnings.filterwarnings('ignore')
seeds = [random.randint(1, 10000) for _ in range(1)]


def get_config():
    config = TCConfigManager.get_tc_config_glove()
    config.model = GloVeTCModelList.LSTM
    config.cache_dataset = False
    config.seed = seeds
    return config

# dataset = DatasetItem('SST2')
# text_classifier = TADTrainer(config=get_config(),
#                              dataset=dataset,
#                              checkpoint_save_mode=1,
#                              auto_device=True
#                              ).load_trained_model()
# dataset = DatasetItem('AGNews10K')
# text_classifier = TADTrainer(config=get_config(),
#                              dataset=dataset,
#                              checkpoint_save_mode=1,
#                              auto_device=True
#                              ).load_trained_model()
dataset = DatasetItem('Amazon')
text_classifier = TADTrainer(config=get_config(),
                             dataset=dataset,
                             checkpoint_save_mode=1,
                             auto_device=True
                             ).load_trained_model()
