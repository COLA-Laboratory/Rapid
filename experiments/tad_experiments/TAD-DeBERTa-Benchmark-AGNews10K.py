# -*- coding: utf-8 -*-
# file: TAD-Benchmark-AGNews10K.py
# time: 03/05/2022
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import argparse

import tqdm
from findfile import find_files

from termcolor import colored
from textattack.attack_results import SuccessfulAttackResult

from textattack import Attacker
from textattack.attack_recipes import (BAEGarg2019,
                                       PWWSRen2019,
                                       TextFoolerJin2019,
                                       PSOZang2020,
                                       IGAWang2019,
                                       GeneticAlgorithmAlzantot2018,
                                       DeepWordBugGao2018,
                                       BERTAttackLi2020)
from textattack.datasets import Dataset
from textattack.models.wrappers import PyTorchModelWrapper

import os

import autocuda
from pyabsa import TADCheckpointManager


def get_ensembled_tad_results(results):
    target_dict = {}
    for r in results:
        target_dict[r['label']] = target_dict.get(r['label']) + 1 if r['label'] in target_dict else 1

    return dict(zip(target_dict.values(), target_dict.keys()))[max(target_dict.values())]


# Quiet TensorFlow.
if "TF_CPP_MIN_LOG_LEVEL" not in os.environ:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class PyABSAModelWrapper(PyTorchModelWrapper):
    def __init__(self, model):
        self.model = model

    def __call__(self, text_inputs, **kwargs):
        outputs = []
        for text_input in text_inputs:
            raw_outputs = self.model.infer(text_input, print_result=False, **kwargs)
            outputs.append(raw_outputs['probs'])
        return outputs


class SentAttacker:

    def __init__(self, model, recipe_class=BAEGarg2019):
        model = model
        model_wrapper = PyABSAModelWrapper(model)

        recipe = recipe_class.build(model_wrapper)
        # WordNet defaults to english. Set the default language to French ('fra')
        # recipe.transformation.language = "en"

        _dataset = [('', 0)]
        _dataset = Dataset(_dataset)

        self.attacker = Attacker(recipe, _dataset)


def run_TAD_benchmark(_dataset: str, _attack_recipe, _defense_attacker: str):
    sent_attacker = SentAttacker(tad_classifier, _attack_recipe)

    filter_key_words = ['.py', '.md', 'readme', 'log', 'result', 'zip', '.state_dict', '.model', '.png', 'acc_', 'f1_',
                        '.origin', '.adv', '.csv']

    dataset_file = {'train': [], 'test': [], 'valid': []}

    search_path = './'
    task = 'text_defense'
    dataset_file['train'] += find_files(search_path, [_dataset, 'train', task],
                                        exclude_key=['.adv', '.org', '.defense', '.inference', 'test.',
                                                     'synthesized'] + filter_key_words)
    dataset_file['test'] += find_files(search_path, [_dataset, 'test', task],
                                       exclude_key=['.adv', '.org', '.defense', '.inference', 'train.',
                                                    'synthesized'] + filter_key_words)
    dataset_file['valid'] += find_files(search_path, [_dataset, 'valid', task],
                                        exclude_key=['.adv', '.org', '.defense', '.inference', 'train.',
                                                     'synthesized'] + filter_key_words)
    dataset_file['valid'] += find_files(search_path, [_dataset, 'dev', task],
                                        exclude_key=['.adv', '.org', '.defense', '.inference', 'train.',
                                                     'synthesized'] + filter_key_words)

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
                for line in lines[:100]:
                    text, label = line.split('$LABEL$')
                    text = text.strip()
                    label = int(label.strip())
                    data.append((text, label))
                    label_set.add(label)

            all_count = 0.
            acc_count = 0.

            defense_count = 0.
            defense_acc_count = 0.

            detection_count = 0.
            detection_acc_count = 0.

            it = tqdm.tqdm(data, postfix='evaluating on {} ...'.format(data_file))
            for text, label in it:
                result = sent_attacker.attacker.simple_attack(text, label)
                all_count += 1
                if result.original_result.ground_truth_output == result.original_result.output and \
                        result.original_result.ground_truth_output == result.perturbed_result.output:
                    acc_count += 1

                elif result.perturbed_result.output != result.original_result.ground_truth_output and \
                        result.original_result.output == result.original_result.ground_truth_output:
                    defense_count += 1
                    detection_count += 1
                    res = tad_classifier.infer(
                        result.perturbed_result.attacked_text.text + '!ref!{}'.format(
                            result.original_result.ground_truth_output),
                        print_result=False,
                        defense=args.defense_attacker
                    )
                    if res['is_adv_label'] == '1':
                        detection_acc_count += 1
                    fixed_label = res['label']
                    if fixed_label == str(result.original_result.ground_truth_output):
                        defense_acc_count += 1

                    if fixed_label == str(result.original_result.ground_truth_output):
                        acc_count += 1
                it.postfix = colored('Restored Accuracy: {}%'.format(round(acc_count / all_count * 100, 2)), 'cyan')
                it.update()

            summary = (
                '-----------------------------------------------------\n'
                'TAD Benchmark Result: \n'
                'Attack Recipe: {}\n'
                'Dataset: {}\n'
                'Restored Accuracy: {}%\n'
                'Accuracy of AdvExample Detection: {}%\n'
                'Accuracy of AdvExample Defense: {}%\n'
                '-----------------------------------------------------\n').format(
                _attack_recipe.__name__,
                args.dataset,
                round(acc_count / len(data), 2) * 100,
                round(detection_acc_count / detection_count, 2) * 100,
                round(defense_acc_count / defense_count, 2) * 100
            )
            print(colored(summary, 'green'))
            with open('./result/{}_{}_TAD_result.txt'.format(args.dataset, args.tad_model), mode='w',
                      encoding='utf8') as fout:
                fout.write(summary)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        default='AGNews10K',
                        required=False,
                        type=str,
                        choices=['AGNews10K', 'SST2'],
                        help='Choose a dataset from AGNews10K, SST2 to run attack with TAD (Text Adversarial-attack Defense)',
                        )
    parser.add_argument("--tad_model",
                        type=str,
                        choices=['TAD-BERT', 'TAD-DeBERTa'],
                        help='Backbone model of TAD',
                        default='TAD-DeBERTa'
                        )
    parser.add_argument("--defense_attacker",
                        default='PWWS',
                        required=False,
                        type=str,
                        choices=['BAE', 'PWWS', 'TextFooler', 'pso', 'iga', 'pso', 'GA', 'wordbugger'],
                        help='Choose a defense attacker used in the TAD model from BAE, PWWS, TextFooler, pso, iga, pso, GA, wordbugger',
                        )
    args = parser.parse_args()

    attack_recipes = {
        # 'bae': BAEGarg2019,
        # 'pwws': PWWSRen2019,
        # 'textfooler': TextFoolerJin2019,
        # 'wordbugger': DeepWordBugGao2018,
        # 'pso': PSOZang2020,
        'iga': IGAWang2019,
        'GA': GeneticAlgorithmAlzantot2018,
        # 'custom': CustomAttackRecipe  # Your Own Attacker implemented in TextAttack
    }

    for attack_recipe in attack_recipes.keys():
        tad_classifier = TADCheckpointManager.get_tad_text_classifier(
            f'TAD-{args.dataset}' if args.tad_model == 'TAD-DeBERTa' else f'TAD-BERT-{args.dataset}',
            auto_device=autocuda.auto_cuda()
        )
        print('-----------------------------------------------------')
        print('Dataset: {}'.format(args.dataset))
        print('Attack Recipe: {}'.format(attack_recipe))
        print('Defense Attacker: {}'.format(args.defense_attacker))
        print('Target model: {}'.format(args.tad_model))
        print('-----------------------------------------------------')
        run_TAD_benchmark(args.dataset, _attack_recipe=attack_recipes[attack_recipe.lower()],
                          _defense_attacker=args.defense_attacker)
