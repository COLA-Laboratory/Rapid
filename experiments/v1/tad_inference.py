# -*- coding: utf-8 -*-
# file: bert_classification_inference.py
# time: 2021/8/5
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import os

from anonymous_demo import TCCheckpointManager, TCDatasetList, TADCheckpointManager
from anonymous_demo.functional.dataset.dataset_manager import AdvTCDatasetList, DatasetItem

os.environ['PYTHONIOENCODING'] = 'UTF8'

ckpt = 'tadbert_SST2_cls_acc_95.71_cls_f1_95.7_adv_det_acc_91.51_adv_det_f1_91.5'

text_classifier = TADCheckpointManager.get_tad_text_classifier(checkpoint=ckpt,
                                                               auto_device=True,  # Use CUDA if available
                                                               )

# batch inferring_tutorials returns the results, save the result if necessary using save_result=True
inference_sets = DatasetItem('SST2')
results = text_classifier.batch_infer(target_file=inference_sets,
                                      print_result=True,
                                      save_result=True,
                                      ignore_error=False,
                                      )

# inference_sets = AdvTCDatasetList.TextFoolerAdv_IMDB10K_Adv_Inference
# results = text_classifier.batch_infer(target_file=inference_sets,
#                                       print_result=True,
#                                       save_result=True,
#                                       ignore_error=False,
#                                       )
#
# inference_sets = AdvTCDatasetList.BAEAdv_SST2_Adv_Inference
# results = text_classifier.batch_infer(target_file=inference_sets,
#                                       print_result=True,
#                                       save_result=True,
#                                       ignore_error=False,
#                                       )

# inference_sets = AdvTCDatasetList.IMDB50K
# results = text_classifier.batch_infer(target_file=inference_sets,
#                                       print_result=True,
#                                       save_result=True,
#                                       ignore_error=False,
#                                       )
#
# inference_sets = AdvTCDatasetList.SST2
# results = text_classifier.batch_infer(target_file=inference_sets,
#                                       print_result=True,
#                                       save_result=True,
#                                       ignore_error=False,
#                                       )
# #
# inference_sets = AdvTCDatasetList.Yelp10K
# results = text_classifier.batch_infer(target_file=inference_sets,
#                                       print_result=True,
#                                       save_result=True,
#                                       ignore_error=False,
#                                       )
# print(results)
