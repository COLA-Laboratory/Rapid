# -*- coding: utf-8 -*-
# file: train_classifier.py
# time: 18/07/2022
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2021. All Rights Reserved.


import os
import warnings

import findfile
import torch.optim

from pyabsa import TextClassificationTrainer, ClassificationConfigManager, ClassificationDatasetList, TCConfigManager
from pyabsa.functional import BERTClassificationModelList
from pyabsa.functional.dataset import DatasetItem

warnings.filterwarnings('ignore')

classification_config_english = TCConfigManager.get_tc_config_english()
classification_config_english.model = BERTClassificationModelList.BERT
classification_config_english.pretrained_bert = 'microsoft/deberta-v3-base'
# classification_config_english.pretrained_bert = 'bert-base-uncased'
classification_config_english.num_epoch = 10
classification_config_english.patience = 3
classification_config_english.evaluate_begin = 0
classification_config_english.optimizer = torch.optim.AdamW
classification_config_english.max_seq_len = 100
classification_config_english.log_step = -1
classification_config_english.dropout = 0.5
classification_config_english.learning_rate = 1e-5
classification_config_english.cache_dataset = False
classification_config_english.seed = {12, 3, 2}
classification_config_english.l2reg = 1e-8
classification_config_english.cross_validate_fold = -1

dataset = DatasetItem('SST2', 'SST2')
text_classifier = TextClassificationTrainer(config=classification_config_english,
                                            dataset=dataset,
                                            checkpoint_save_mode=1,
                                            auto_device=True
                                            ).load_trained_model()

dataset = DatasetItem('AGNews', 'AGNews')
text_classifier = TextClassificationTrainer(config=classification_config_english,
                                            dataset=dataset,
                                            checkpoint_save_mode=1,
                                            auto_device=True
                                            ).load_trained_model()

dataset = DatasetItem('Amazon', 'Amazon')
text_classifier = TextClassificationTrainer(config=classification_config_english,
                                            dataset=dataset,
                                            checkpoint_save_mode=1,
                                            auto_device=True
                                            ).load_trained_model()
