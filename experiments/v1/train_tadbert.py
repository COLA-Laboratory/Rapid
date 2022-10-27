# -*- coding: utf-8 -*-
# file: train_text_classification_bert.py
# time: 2021/8/5
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import os
import random
import warnings

import findfile

# Transfer Experiments and Multitask Experiments

from anonymous_demo import TCTrainer, TADConfigManager, TCDatasetList, ATDBERTTCModelList, TADTrainer
from anonymous_demo.functional.dataset.dataset_manager import AdvTCDatasetList, DatasetItem

warnings.filterwarnings('ignore')
seeds = [random.randint(1, 10000) for _ in range(1)]


def get_config():
    config = TADConfigManager.get_tad_config_english()
    config.model = ATDBERTTCModelList.TADBERT
    config.num_epoch = 10
    config.patience = 3
    config.evaluate_begin = 0
    config.max_seq_len = 80
    config.log_step = -1
    config.dropout = 0.5
    config.learning_rate = 1e-5
    config.cache_dataset = False
    config.seed = seeds
    config.l2reg = 1e-5
    config.cross_validate_fold = -1
    return config


dataset = DatasetItem('SST2')
# dataset = AdvTCDatasetList.TextFoolerAdv_SST2
text_classifier = TADTrainer(config=get_config(),
                             dataset=dataset,
                             checkpoint_save_mode=1,
                             auto_device=True
                             ).load_trained_model()

inference_sets = dataset
results = text_classifier.batch_infer(target_file=inference_sets,
                                      print_result=True,
                                      save_result=True,
                                      ignore_error=False,
                                      )

dataset = DatasetItem('IMDB10K')
text_classifier = TADTrainer(config=get_config(),
                             dataset=dataset,
                             checkpoint_save_mode=1,
                             auto_device=True
                             ).load_trained_model()

dataset = DatasetItem('AGnews10K')
text_classifier = TADTrainer(config=get_config(),
                             dataset=dataset,
                             checkpoint_save_mode=1,
                             auto_device=True
                             ).load_trained_model()
#
#
# # ------------------------------------------------------------------------------
# dataset = AdvTCDatasetList.TextFoolerAdv_SST2
# text_classifier = AOTCTrainer(config=get_config(),
#                               dataset=dataset,
#                               checkpoint_save_mode=0,
#                               auto_device=True
#                               ).load_trained_model()
#
# dataset = AdvTCDatasetList.BAEAdv_SST2
# text_classifier = AOTCTrainer(config=get_config(),
#                               dataset=dataset,
#                               checkpoint_save_mode=0,
#                               auto_device=True
#                               ).load_trained_model()
#
# dataset = AdvTCDatasetList.PWWSAdv_SST2
# text_classifier = AOTCTrainer(config=get_config(),
#                               dataset=dataset,
#                               checkpoint_save_mode=0,
#                               auto_device=True
#                               ).load_trained_model()
# # ------------------------------------------------------------------------------
#
#
# # ------------------------------------------------------------------------------
# dataset = AdvTCDatasetList.TextFoolerAdv_IMDB10K
# text_classifier = AOTCTrainer(config=get_config(),
#                               dataset=dataset,
#                               checkpoint_save_mode=0,
#                               auto_device=True
#                               ).load_trained_model()
#
# dataset = AdvTCDatasetList.BAEAdv_IMDB10K
# text_classifier = AOTCTrainer(config=get_config(),
#                               dataset=dataset,
#                               checkpoint_save_mode=0,
#                               auto_device=True
#                               ).load_trained_model()
#
# dataset = AdvTCDatasetList.PWWSAdv_IMDB10K
# text_classifier = AOTCTrainer(config=get_config(),
#                               dataset=dataset,
#                               checkpoint_save_mode=0,
#                               auto_device=True
#                               ).load_trained_model()
# # ------------------------------------------------------------------------------


# # ------------------------------------------------------------------------------
# dataset = AdvTCDatasetList.TextFoolerAdv_AGNews10K
# text_classifier = AOTCTrainer(config=get_config(),
#                               dataset=dataset,
#                               checkpoint_save_mode=1,
#                               auto_device=True
#                               ).load_trained_model()
#
# dataset = AdvTCDatasetList.BAEAdv_AGNews10K
# text_classifier = AOTCTrainer(config=get_config(),
#                               dataset=dataset,
#                               checkpoint_save_mode=1,
#                               auto_device=True
#                               ).load_trained_model()
#
# dataset = AdvTCDatasetList.PWWSAdv_AGNews10K
# text_classifier = AOTCTrainer(config=get_config(),
#                               dataset=dataset,
#                               checkpoint_save_mode=1,
#                               auto_device=True
#                               ).load_trained_model()
# # ------------------------------------------------------------------------------


# # batch inferring_tutorials returns the results, save the result if necessary using save_result=True
# inference_sets = AdvTCDatasetList.TextFoolerAdv_SST2
# results = text_classifier.batch_infer(target_file=inference_sets,
#                                       print_result=False,
#                                       save_result=True,
#                                       ignore_error=False,
#                                       )
# # print(results)
