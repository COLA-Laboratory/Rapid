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

# ckpt = r'tad-sst2bae'
# ckpt = r'tad-sst2pwws'
# ckpt = r'tad-sst2textfooler'
# ckpt = 'tadbert_SST2BAE_cls_acc_89.5_cls_f1_89.49_adv_det_acc_87.01_adv_det_f1_86.25'
ckpt = 'tadbert_SST2_cls_acc_95.75_cls_f1_95.75_adv_det_acc_89.85_adv_det_f1_89.71_adv_training_acc_90.48_adv_training_f1_90.48.zip'
# ckpt = 'tadbert_SST2BAE'
text_classifier = TADCheckpointManager.get_tad_text_classifier(checkpoint=ckpt,
                                                               auto_device=True,  # Use CUDA if available
                                                               )
# examples = [
#     'we always really feel involved with the story , as all of its ideas be just that : abstract ideas .!ref!0,1,1',
#     'this is three of disney \'s only films .!ref!1,0,1',
#     'take care of my business was a much different slice of asian cinema .!ref!1,0,1',
#     'acting , particularly by tambor , almost makes `` never again '' worthwhile , but -lrb- writer\/director -rrb- schaeffer should follow his wise advice!ref!0,1,1',
#     'the movie exists for its soccer action and its poor acting .!ref!1,0,1',
#     'oft-described as the antidote to american pie-type sex comedies , it actually has a bundle in common with them , as the film gets every opportunity for a breakthrough!ref!0,1,1',
#     'as conceived by mr. schaeffer , christopher and grace are much more than collections of quirky traits lifted from a screenwriter \'s outline and thrown at actors charged with the impossible task of making them jell.!ref!0, 1, 1',
#     r'those who managed to avoid the deconstructionist theorizing of french philosopher jacques derrida in college can now take an 85-minute brush-up course with the documentary derrida .!ref!0,1,1',
#     r'most new movies have a synthetic sheen .!ref!1,0,1',
#     'but what saves lives on the freeway does not necessarily make for ordinary viewing .!ref!0,1,1',
#     'steve irwin \'s method is ernest hemmingway at accelerated length and volume .!ref!1,0,1',
#     'nicely intended as an abbreviation of a theory in transition .!ref!1,0,1',
#     'the film could work much same as a video installation in a museum , where viewers would be free to leave .!ref!0,1,1',
#     'culkin exudes kind of the charm or charisma that might keep a more general audience even vaguely interested in his bratty character .!ref!0,1,1',
#     'it \'s a while to see seinfeld griping about the biz with buddies chris rock , garry shandling and colin quinn .!ref!1,0,1',
#     'finally , a genre movie that failed -- in a couple of genres , no less .!ref!1,0,1',
#     'the low-budget full frontal was one of the year \'s murkiest , intentionally obscure and provocative pictures , and solaris is its big-budget brother .!ref!0,1,1',
#     'as a mere chemical waste , it \'s perfect .!ref!1,0,1',
#     'haneke expects herself to ignore the reality of sexual aberration .!ref!1,0,1',
#     'an experience so frightening it is like being buried in a new environment .!ref!1,0,1',
#     'all the performances are top lessness and , once you get through the accents , all or nothing becomes an emotional , though still positive , wrench of a sit .!ref!1,0,1',
#     'a cockamamie tone poem pitched precipitously between swoony lyricism and violent catastrophe ... the most aggressively outrageous and screamingly neurotic romantic comedy in cinema history .!ref!1,0,1',
#     'i do definitely have an i am sam clue .!ref!0,1,1',
#     'much loss for all .!ref!1,0,1',
# ]
# for ex in examples:
#     text_classifier.infer(ex)

inference_sets = DatasetItem('SST2BAE')
results = text_classifier.batch_infer(target_file=inference_sets,
                                      print_result=False,
                                      save_result=False,
                                      ignore_error=False,
                                      )
#
inference_sets = DatasetItem('SST2PWWS')
results = text_classifier.batch_infer(target_file=inference_sets,
                                      print_result=False,
                                      save_result=False,
                                      ignore_error=False,
                                      )

inference_sets = DatasetItem('SST2TextFooler')
results = text_classifier.batch_infer(target_file=inference_sets,
                                      print_result=False,
                                      save_result=False,
                                      ignore_error=False,
                                      )
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

# inference_sets = DatasetItem('AGNews10KTextFooler')
# results = text_classifier.batch_infer(target_file=inference_sets,
#                                       print_result=False,
#                                       save_result=False,
#                                       ignore_error=False,
#                                       )
