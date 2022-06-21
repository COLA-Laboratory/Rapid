# -*- coding: utf-8 -*-
# file: generate_adversarial_examples.py
# time: 03/05/2022
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import json
import os

import tqdm
from findfile import find_files

# Quiet TensorFlow.
import os

import numpy as np
import pandas
from termcolor import colored
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, pipeline, AutoModelForSequenceClassification

from textattack import Attacker
from textattack.attack_recipes import BERTAttackLi2020, BAEGarg2019, PWWSRen2019, TextFoolerJin2019, PSOZang2020, IGAWang2019, GeneticAlgorithmAlzantot2018, DeepWordBugGao2018
from textattack.attack_recipes import TextFoolerJin2019
from textattack.attack_results import SuccessfulAttackResult
from textattack.datasets import HuggingFaceDataset, Dataset
from textattack.models.wrappers import ModelWrapper, HuggingFaceModelWrapper

import os

import autocuda
from pyabsa import TCConfigManager, GloVeTCModelList, TCDatasetList, BERTTCModelList

from boost_aug import TCBoostAug, AugmentBackend

if "TF_CPP_MIN_LOG_LEVEL" not in os.environ:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

device = autocuda.auto_cuda()


class PyABSAMOdelWrapper(HuggingFaceModelWrapper):
    """ Transformers sentiment analysis pipeline returns a list of responses
        like

            [{'label': 'POSITIVE', 'score': 0.7817379832267761}]

        We need to convert that to a format TextAttack understands, like

            [[0.218262017, 0.7817379832267761]
    """

    def __init__(self, model):
        self.model = model  # pipeline = pipeline

    def __call__(self, text_inputs, **kwargs):
        outputs = []
        for text_input in text_inputs:
            raw_outputs = self.model.infer(text_input, print_result=False, **kwargs)
            outputs.append(raw_outputs['probs'])
        return outputs


class SentAttacker:

    def __init__(self, model, recipe_class=BAEGarg2019):
        # Create the model: a French sentiment analysis model.
        # see https://github.com/TheophileBlard/french-sentiment-analysis-with-bert
        # model = AutoModelForSequenceClassification.from_pretrained(huggingface_model)
        # tokenizer = AutoTokenizer.from_pretrained(huggingface_model)
        # sent_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        # model_wrapper = HuggingFaceSentimentAnalysisPipelineWrapper(sent_pipeline)
        # model_wrapper = HuggingFaceModelWrapper(model=model, tokenizer=tokenizer)
        model = model
        model_wrapper = PyABSAMOdelWrapper(model)

        recipe = recipe_class.build(model_wrapper)
        # WordNet defaults to english. Set the default language to French ('fra')
        #
        # See
        # "Building a free French wordnet from multilingual resources",
        # E. L. R. A. (ELRA) (ed.),
        # Proceedings of the Sixth International Language Resources and Evaluation (LREC’08).

        # recipe.transformation.language = "en"

        # dataset = HuggingFaceDataset("sst", split="test")
        # data = pandas.read_csv('examples.csv')
        dataset = [('', 0)]
        dataset = Dataset(dataset)

        self.attacker = Attacker(recipe, dataset)


def generate_adversarial_example(dataset, augmentor):
    # attack_recipe_name = attack_recipe.__name__
    attack_recipes = [
        BAEGarg2019,
        PWWSRen2019,
        TextFoolerJin2019,
    ]
    attackers = [SentAttacker(augmentor.text_classifier, attack_recipe) for attack_recipe in attack_recipes]

    filter_key_words = ['.py', '.md', 'readme', 'log', 'result', 'zip', '.state_dict', '.model', '.png', 'acc_', 'f1_', '.origin', '.adv', '.csv']

    dataset_file = {'train': [], 'test': [], 'valid': []}

    search_path = './'
    task = 'text_classification'
    dataset_file['train'] += find_files(search_path, [dataset, 'train', task], exclude_key=['.adv', '.org', '.defense', '.inference', 'test.', 'synthesized'] + filter_key_words)
    dataset_file['test'] += find_files(search_path, [dataset, 'test', task], exclude_key=['.adv', '.org', '.defense', '.inference', 'train.', 'synthesized'] + filter_key_words)
    dataset_file['valid'] += find_files(search_path, [dataset, 'valid', task], exclude_key=['.adv', '.org', '.defense', '.inference', 'train.', 'synthesized'] + filter_key_words)
    dataset_file['valid'] += find_files(search_path, [dataset, 'dev', task], exclude_key=['.adv', '.org', '.defense', '.inference', 'train.', 'synthesized'] + filter_key_words)

    for dat_type in [
        # 'train',
        # 'valid',
        'test'
    ]:
        data = []
        label_set = set()
        for data_file in dataset_file[dat_type]:
            print("Attack: {}".format(data_file))

            with open(data_file, mode='r', encoding='utf8') as fin:
                lines = fin.readlines()
                for line in lines:
                    text, label = line.split('$LABEL$')
                    text = text.strip()
                    label = int(label.strip())
                    data.append((text, label))
                    label_set.add(label)

            adv_data = []
            org_data = []
            count = 0.
            acc_count = 0.
            for text, label in tqdm.tqdm(data, postfix='attacking...'):
                fix_successful = 0
                for attacker in attackers:

                    result = attacker.attacker.simple_attack(text, label)
                    new_data = {}

                    if result is not None:
                        new_data['origin_text'] = result.original_result.attacked_text.text
                        new_data['origin_label'] = result.original_result.ground_truth_output

                        new_data['adv_text'] = result.perturbed_result.attacked_text.text
                        new_data['perturb_label'] = result.perturbed_result.output
                        new_data['is_adv'] = 1

                        if result.perturbed_result.output != result.original_result.ground_truth_output and \
                            result.original_result.output == result.original_result.ground_truth_output:
                            augs = augmentor.single_augment(result.original_result.attacked_text.text, result.original_result.ground_truth_output, 1)
                            if augs:
                                for aug in augs:
                                    infer_res = augmentor.text_classifier.infer(aug + '!ref!{}'.format(result.perturbed_result.ground_truth_output))
                                    if infer_res['ref_check'] == 'Correct':
                                        fix_successful += 1
                        elif result.original_result.ground_truth_output == result.original_result.output and \
                            result.original_result.ground_truth_output == result.perturbed_result.output:
                            fix_successful += 1

                if fix_successful == len(attackers):
                    acc_count += 1
            print(colored('Accuracy Under Attack: {}'.format(acc_count / len(data)), 'green'))


def prepare_augmentor(dataset):
    tc_config = TCConfigManager.get_classification_config_english()
    tc_config.model = BERTTCModelList.BERT  # 'BERT' model can be used for DeBERTa or BERT
    tc_config.num_epoch = 15
    tc_config.evaluate_begin = 0
    tc_config.max_seq_len = 100
    tc_config.pretrained_bert = 'microsoft/deberta-v3-base'
    # tc_config.pretrained_bert = r'checkpoints\mono_boost\bert_SST2_deberta-v3-base\bert_SST2_acc_95.53_f1_95.52'
    # tc_config.pretrained_bert = 'textattack/bert-base-uncased-SST-2'
    tc_config.log_step = 100
    tc_config.dropout = 0.1
    tc_config.cache_dataset = False
    tc_config.seed = 1
    tc_config.l2reg = 1e-7
    tc_config.learning_rate = 1e-5

    backend = AugmentBackend.EDA
    dataset_map = {
        'sst2': TCDatasetList.SST2,
        'agnews10k': TCDatasetList.AGNews10K,
        'yelp10k': TCDatasetList.Yelp10K,
        'imdb10k': TCDatasetList.IMDB10K
    }

    augmentor = TCBoostAug(ROOT=os.getcwd(),
                           AUGMENT_BACKEND=backend,
                           CLASSIFIER_TRAINING_NUM=1,
                           WINNER_NUM_PER_CASE=8,
                           AUGMENT_NUM_PER_CASE=16,
                           CONFIDENCE_THRESHOLD=0.8,
                           PERPLEXITY_THRESHOLD=5,
                           USE_LABEL=False,
                           device=device)
    # augmentor.tc_mono_augment(tc_config,
    #                           dataset_map[dataset.lower()],
    #                           rewrite_cache=False,
    #                           train_after_aug=False
    #                           )
    # augmentor.tc_boost_augment(tc_config,
    #                            dataset_map[dataset.lower()],
    #                            rewrite_cache=True,
    #                            train_after_aug=True
    #                            )
    augmentor.USE_LABEL = False
    # augmentor.load_augmentor('bert_SST2_acc_95.76_f1_95.76')
    augmentor.load_augmentor('TC-{}'.format(dataset))
    return augmentor


if __name__ == '__main__':

    attack_name = 'BAE'
    # attack_name = 'PWWS'
    # attack_name = 'TextFooler'

    # attack_name = 'PSO'
    # attack_name = 'IGA'
    # attack_name = 'WordBug'

    datasets = [
        # 'sst2',
        'agnews10k',
        # 'Yelp10K'
        # 'imdb10k',
    ]

    for dataset in datasets:
        augmentor = prepare_augmentor(dataset)

        attack_recipes = {
            'bae': BAEGarg2019,
            'pwws': PWWSRen2019,
            'textfooler': TextFoolerJin2019,
            'pso': PSOZang2020,
            'iga': IGAWang2019,
            'GA': GeneticAlgorithmAlzantot2018,
            'wordbugger': DeepWordBugGao2018,
        }
        generate_adversarial_example(dataset, augmentor=augmentor)
