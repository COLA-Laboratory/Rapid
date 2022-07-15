# -*- coding: utf-8 -*-
# file: bert_classification_inference.py
# time: 2021/8/5
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import os

from pyabsa import TCCheckpointManager, TCDatasetList, TADCheckpointManager
from pyabsa.functional.dataset.dataset_manager import AdvTCDatasetList, DatasetItem

os.environ['PYTHONIOENCODING'] = 'UTF8'

ckpt = r'tad-sst2bae'
# ckpt = r'tad-sst2pwws'
# ckpt = r'tad-sst2textfooler'
# ckpt = r'tad-agnews10kbae'
# ckpt = r'tad-agnews10kpwws'
# ckpt = r'tad-agnews10ktextfooler'
text_classifier = TADCheckpointManager.get_tad_text_classifier(checkpoint=ckpt,
                                                               auto_device=True,  # Use CUDA if available
                                                               )
#
inference_sets = DatasetItem('SST2BAE')
results = text_classifier.batch_infer(target_file=inference_sets,
                                      print_result=False,
                                      save_result=False,
                                      ignore_error=False,
                                      )
#
# inference_sets = DatasetItem('SST2PWWS')
# results = text_classifier.batch_infer(target_file=inference_sets,
#                                       print_result=False,
#                                       save_result=False,
#                                       ignore_error=False,
#                                       )
#
# inference_sets = DatasetItem('SST2TextFooler')
# results = text_classifier.batch_infer(target_file=inference_sets,
#                                       print_result=False,
#                                       save_result=False,
#                                       ignore_error=False,
#                                       )
#
# inference_sets = DatasetItem('AGNews10KBAE')
# results = text_classifier.batch_infer(target_file=inference_sets,
#                                       print_result=False,
#                                       save_result=True,
#                                       ignore_error=False,
#                                       )

# inference_sets = DatasetItem('AGNews10KPWWS')
# results = text_classifier.batch_infer(target_file=inference_sets,
#                                       print_result=True,
#                                       save_result=False,
#                                       ignore_error=False,
#                                       )
#
# inference_sets = DatasetItem('AGNews10KTextFooler')
# results = text_classifier.batch_infer(target_file=inference_sets,
#                                       print_result=False,
#                                       save_result=False,
#                                       ignore_error=False,
#                                       )
