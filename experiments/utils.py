# -*- coding: utf-8 -*-
# file: utils.py.py
# time: 15/06/2022
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import random


# def get_ensembled_tad_results(results):
#     target_dict = {}
#     for r in results:
#         target_dict[r['label']] = target_dict.get(r['label']) + 1 if r['label'] in target_dict else 1
#     if len(target_dict) == 1:
#         if list(target_dict.keys())[0] != '0':
#             return str(random.randint(0, int(list(target_dict.keys())[0])))
#         else:
#             return '1'
#     if min(target_dict.values()) <= 2:
#         return dict(zip(target_dict.values(), target_dict.keys()))[min(target_dict.values())]
#     else:
#         return dict(zip(target_dict.values(), target_dict.keys()))[max(target_dict.values())]
#
#
# def get_ensembled_tc_results(results):
#     target_dict = {}
#     for r in results:
#         target_dict[r['label']] = target_dict.get(r['label']) + 1 if r['label'] in target_dict else 1
#     if len(target_dict) == 1:
#         if list(target_dict.keys())[0] != '0':
#             return str(random.randint(0, int(list(target_dict.keys())[0])))
#         else:
#             return '1'
#     if min(target_dict.values()) <= 2:
#         return dict(zip(target_dict.values(), target_dict.keys()))[min(target_dict.values())]
#     else:
#         return dict(zip(target_dict.values(), target_dict.keys()))[max(target_dict.values())]

def get_ensembled_tad_results(results):
    target_dict = {}
    for r in results:
        target_dict[r['label']] = target_dict.get(r['label']) + 1 if r['label'] in target_dict else 1

    return dict(zip(target_dict.values(), target_dict.keys()))[max(target_dict.values())]
    # return dict(zip(target_dict.values(), target_dict.keys()))[max(target_dict.values())]


def get_ensembled_tc_results(results):
    target_dict = {}
    for r in results:
        target_dict[r['label']] = target_dict.get(r['label']) + 1 if r['label'] in target_dict else 1

    return dict(zip(target_dict.values(), target_dict.keys()))[max(target_dict.values())]
    # return dict(zip(target_dict.values(), target_dict.keys()))[max(target_dict.values())]

