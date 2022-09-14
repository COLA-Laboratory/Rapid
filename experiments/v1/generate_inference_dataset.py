# -*- coding: utf-8 -*-
# file: generate_inference_dataset.py
# time: 11/06/2022
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import os

from pyabsa.functional.dataset import detect_dataset

datasets = detect_dataset(os.getcwd(), task='text_defense')

for dataset in datasets['train']:
    with open(dataset + '.inference', mode='w', encoding='utf8') as fout:
        with open(dataset, mode='r', encoding='utf8') as fin:
            for line in fin.readlines():
                fout.write(line.replace('$LABEL$', '!ref!'))
