# -*- coding: utf-8 -*-
# file: bert_classification_inference.py
# time: 2021/8/5
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import os

import findfile

from anonymous_demo import TCCheckpointManager, TCDatasetList, TADCheckpointManager
from anonymous_demo.functional.dataset.dataset_manager import AdvTCDatasetList, DatasetItem

os.environ['PYTHONIOENCODING'] = 'UTF8'

# ckpt = r'tad-sst2'
# ckpt = r'tad-sst2bae'
# ckpt = r'tad-sst2pwws'
# ckpt = r'tad-sst2textfooler'
# ckpt = r'tad-agnews10k'
# ckpt = r'tad-agnews10kbae'
# ckpt = r'tad-agnews10kpwws'
# ckpt = r'tad-agnews10ktextfooler'
# ckpt = r'tad-amazon'
# ckpt = r'tad-amazonbae'
# ckpt = r'tad-amazonpwws'
# ckpt = r'tad-amazontextfooler'

for dataset in [
    'SST2',
    'Amazon',
    'AGNews',
]:
# for dataset in [
#     'SST2BAE',
#     'SST2PWWS',
#     'SST2TextFooler',
#     'AGNews10KBAE',
#     'AGNews10KPWWS',
#     'AGNews10KTextFooler',
#     'AmazonBAE',
#     'AmazonPWWS',
#     'AmazonTextFooler',
# ]:
    inference_sets = DatasetItem(dataset, findfile.find_cwd_files([dataset, '.org', '.inference']))

    text_classifier = TADCheckpointManager.get_tad_text_classifier(checkpoint=f'TAD-{dataset}',
                                                                   auto_device=True,  # Use CUDA if available
                                                                   )

    # inference_sets = DatasetItem(dataset)
    results = text_classifier.batch_infer(target_file=inference_sets,
                                          print_result=False,
                                          save_result=False,
                                          ignore_error=False,
                                          )

    # input('Press Enter to continue...')
    #
    # text_classifier = TADCheckpointManager.get_tad_text_classifier(checkpoint=f'TAD-BERT-{dataset}',
    #                                                                auto_device=True,  # Use CUDA if available
    #                                                                )
    #
    #
    # inference_sets = DatasetItem(dataset)
    # results = text_classifier.batch_infer(target_file=inference_sets,
    #                                       print_result=False,
    #                                       save_result=False,
    #                                       ignore_error=False,
    #                                       )
    #
    input('Press Enter to continue...')
