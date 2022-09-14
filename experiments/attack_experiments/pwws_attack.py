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
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, pipeline, \
    AutoModelForSequenceClassification

from textattack import Attacker
from textattack.attack_recipes import BERTAttackLi2020, BAEGarg2019, PWWSRen2019, TextFoolerJin2019, PSOZang2020, \
    IGAWang2019, GeneticAlgorithmAlzantot2018, DeepWordBugGao2018
from textattack.attack_recipes import TextFoolerJin2019
from textattack.attack_results import SuccessfulAttackResult
from textattack.datasets import HuggingFaceDataset, Dataset
from textattack.models.wrappers import ModelWrapper, HuggingFaceModelWrapper

import os

import autocuda
from pyabsa import TCConfigManager, GloVeTCModelList, TCDatasetList, BERTTCModelList, TCCheckpointManager

if "TF_CPP_MIN_LOG_LEVEL" not in os.environ:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

device = autocuda.auto_cuda()


class PyABSAModelWrapper(HuggingFaceModelWrapper):
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
        model_wrapper = PyABSAModelWrapper(model)

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


def generate_adversarial_example(dataset, attack_recipe, text_classifier):
    attack_recipe_name = attack_recipe.__name__
    sent_attacker = SentAttacker(text_classifier, attack_recipe)

    filter_key_words = ['.py', '.md', 'readme', 'log', 'result', 'zip', '.state_dict', '.model', '.png', 'acc_', 'f1_',
                        '.origin', '.adv', '.csv']

    dataset_file = {'train': [], 'test': [], 'valid': []}

    search_path = './'
    task = 'text_classification'
    # dataset_file['train'] += find_files(search_path, [dataset, 'train', task], exclude_key=['.adv', '.org', '.defense', '.inference', 'test.', 'synthesized'] + filter_key_words)
    dataset_file['test'] += find_files(search_path, [dataset, 'test', task],
                                       exclude_key=['.adv', '.org', '.defense', '.inference', 'train.',
                                                    'synthesized'] + filter_key_words)
    # dataset_file['valid'] += find_files(search_path, [dataset, 'valid', task], exclude_key=['.adv', '.org', '.defense', '.inference', 'train.', 'synthesized'] + filter_key_words)
    # dataset_file['valid'] += find_files(search_path, [dataset, 'dev', task], exclude_key=['.adv', '.org', '.defense', '.inference', 'train.', 'synthesized'] + filter_key_words)

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

            print(label_set)

            count = 0
            acc_count = 0
            for text, label in tqdm.tqdm(data, postfix='attacking...'):
                result = sent_attacker.attacker.simple_attack(text, label)
                count += 1
                if result.original_result.ground_truth_output == result.original_result.output and \
                        result.original_result.ground_truth_output == result.perturbed_result.output:
                    acc_count += 1

                print(colored('Accuracy: {}%'.format(acc_count / count * 100), 'cyan'))
            print(colored('Accuracy Under Attack: {}'.format(acc_count / len(data)), 'green'))


if __name__ == '__main__':

    # attack_name = 'BAE'
    attack_name = 'PWWS'
    # attack_name = 'TextFooler'

    # attack_name = 'PSO'
    # attack_name = 'IGA'
    # attack_name = 'WordBug'

    datasets = [
        # 'sst2',
        # 'agnews10k',
        # 'Yelp10K'
        # 'imdb10k',
        'Amazon'

    ]

    for dataset in datasets:
        text_classifier = TCCheckpointManager.get_text_classifier('bert_{}'.format(dataset))
        attack_recipes = {
            'bae': BAEGarg2019,
            'pwws': PWWSRen2019,
            'textfooler': TextFoolerJin2019,
            'pso': PSOZang2020,
            'iga': IGAWang2019,
            'GA': GeneticAlgorithmAlzantot2018,
            'wordbugger': DeepWordBugGao2018,
        }
        generate_adversarial_example(dataset, attack_recipe=attack_recipes[attack_name.lower()],
                                     text_classifier=text_classifier)
