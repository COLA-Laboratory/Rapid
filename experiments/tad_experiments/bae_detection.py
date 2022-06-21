# -*- coding: utf-8 -*-
# file: generate_adversarial_examples.py
# time: 03/05/2022
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import tqdm
from findfile import find_files

from termcolor import colored
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, pipeline, AutoModelForSequenceClassification

from textattack import Attacker
from textattack.attack_recipes import BERTAttackLi2020, BAEGarg2019, PWWSRen2019, TextFoolerJin2019, PSOZang2020, IGAWang2019, GeneticAlgorithmAlzantot2018, DeepWordBugGao2018
from textattack.attack_results import SuccessfulAttackResult
from textattack.datasets import Dataset
from textattack.models.wrappers import HuggingFaceModelWrapper

import os

import autocuda
from pyabsa import TADCheckpointManager

# Quiet TensorFlow.
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



if "TF_CPP_MIN_LOG_LEVEL" not in os.environ:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# device = autocuda.auto_cuda()
# from textattack.augmentation import EasyDataAugmenter as Aug
#
# # Alter default values if desired
# eda_augmenter = Aug(pct_words_to_swap=0.3, transformations_per_example=2)

# import nlpaug.augmenter.word as naw
#
# bert_augmenter = naw.ContextualWordEmbsAug(
#     model_path='roberta-base', action="substitute", aug_p=0.3, device=autocuda.auto_cuda())

# raw_augs = augmenter.augment(text)


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
        model = model
        model_wrapper = PyABSAMOdelWrapper(model)

        recipe = recipe_class.build(model_wrapper)
        # WordNet defaults to english. Set the default language to French ('fra')

        # recipe.transformation.language = "en"

        _dataset = [('', 0)]
        _dataset = Dataset(_dataset)

        self.attacker = Attacker(recipe, _dataset)


def generate_adversarial_example(dataset, attack_recipe):
    attack_recipe_name = attack_recipe.__name__
    sent_attacker = SentAttacker(tad_classifier, attack_recipe)

    filter_key_words = ['.py', '.md', 'readme', 'log', 'result', 'zip', '.state_dict', '.model', '.png', 'acc_', 'f1_', '.origin', '.adv', '.csv']

    dataset_file = {'train': [], 'test': [], 'valid': []}

    search_path = './'
    task = 'text_defense'
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

            count = 0.
            acc_count = 0.
            it = tqdm.tqdm(data, postfix='testing ...')
            for text, label in it:
                result = sent_attacker.attacker.simple_attack(text, label)
                count += 1
                if result.original_result.ground_truth_output == result.original_result.output and \
                    result.original_result.ground_truth_output == result.perturbed_result.output:
                    acc_count += 1

                elif result.perturbed_result.output != result.original_result.ground_truth_output and \
                    result.original_result.output == result.original_result.ground_truth_output:

                    infer_res = tad_classifier.infer(
                        result.perturbed_result.attacked_text.text + '!ref!{},{},{}'.format(result.original_result.ground_truth_output, 1, result.perturbed_result.output),
                        print_result=False,
                        # defense='bae'
                    )
                    if infer_res['is_adv_label'] == '1':
                        examples = []
                        res = sent_attacker.attacker.simple_attack(result.perturbed_result.attacked_text.text, int(infer_res['label']))
                        examples.append(res.perturbed_result.attacked_text.text)
                        # for enum_label in tad_classifier.opt.label_to_index.values():
                        #     if enum_label != -100 and enum_label != int(infer_res['label']):
                        #         res = sent_attacker.attacker.simple_attack(result.perturbed_result.attacked_text.text, int(infer_res['label']))
                        #         # examples = bert_augmenter.augment(res.perturbed_result.attacked_text.text, 10, num_thread=os.cpu_count())
                        #         # examples = eda_augmenter.augment(res.perturbed_result.attacked_text.text) + [res.perturbed_result.attacked_text.text]
                        #         examples = [res.perturbed_result.attacked_text.text]
                        infer_results = []
                        for ex in examples:
                            infer_results.append(
                                tad_classifier.infer(
                                    ex + '!ref!{},{},{}'.format(result.original_result.ground_truth_output, 1, result.perturbed_result.output),
                                    print_result=False
                                )
                            )
                        fixed_label = get_ensembled_tad_results(infer_results)
                    else:
                        fixed_label = infer_res['label']

                    if fixed_label == str(result.original_result.ground_truth_output):
                        acc_count += 1
                    #     print(colored('Success', 'green'))
                    # else:
                    #     print(colored('Failed', 'red'))
                it.postfix = colored('Accuracy: {}%'.format(acc_count / count * 100), 'cyan')
                it.update()
            print(colored('Accuracy Under Attack: {}'.format(acc_count / len(data)), 'green'))


if __name__ == '__main__':

    attack_name = 'BAE'
    # attack_name = 'PWWS'
    # attack_name = 'TextFooler'

    # attack_name = 'PSO'
    # attack_name = 'IGA'
    # attack_name = 'WordBug'

    datasets = [
        'sst2',
        'agnews10k',
        # 'Yelp10K',
        # 'imdb10k',
    ]

    for dataset in datasets:
        tad_classifier = TADCheckpointManager.get_tad_text_classifier(
            # f'tadbert_SST2_cls_acc_95.75_cls_f1_95.75_adv_det_acc_89.85_adv_det_f1_89.71_adv_training_acc_90.48_adv_training_f1_90.48.zip',
            # f'tadbert_{dataset}{attack_name}',
            f'tadbert_{dataset}',
            # f'TAD-{dataset}',
            auto_device=autocuda.auto_cuda()
        )

        attack_recipes = {
            'bae': BAEGarg2019,
            'pwws': PWWSRen2019,
            'textfooler': TextFoolerJin2019,
            'pso': PSOZang2020,
            'iga': IGAWang2019,
            'GA': GeneticAlgorithmAlzantot2018,
            'wordbugger': DeepWordBugGao2018,
        }
        generate_adversarial_example(dataset, attack_recipe=attack_recipes[attack_name.lower()])

    # for attack_name in [
    #     'BAE',
    #     'PWWS',
    #     'TextFooler'
    # ]:
    #     datasets = [
    #         'sst2',
    #         # 'agnews10k',
    #         # 'Yelp10K',
    #         # 'imdb10k',
    #     ]
    #
    #     for dataset in datasets:
    #         tad_classifier = TADCheckpointManager.get_tad_text_classifier(
    #             # f'tadbert_SST2_cls_acc_95.75_cls_f1_95.75_adv_det_acc_89.85_adv_det_f1_89.71_adv_training_acc_90.48_adv_training_f1_90.48.zip',
    #             # f'tadbert_{dataset}{attack_name}',
    #             f'TAD-{dataset}',
    #             auto_device=autocuda.auto_cuda()
    #         )
    #
    #         attack_recipes = {
    #             'bae': BAEGarg2019,
    #             'pwws': PWWSRen2019,
    #             'textfooler': TextFoolerJin2019,
    #             'pso': PSOZang2020,
    #             'iga': IGAWang2019,
    #             'GA': GeneticAlgorithmAlzantot2018,
    #             'wordbugger': DeepWordBugGao2018,
    #         }
    #         generate_adversarial_example(dataset, attack_recipe=attack_recipes[attack_name.lower()])
