# -*- coding: utf-8 -*-
# file: bert_classification_inference.py
# time: 2021/8/5
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import os
import random
import signal

import findfile
import numpy as np
import torch
import tqdm
from decorator import contextmanager
from metric_visualizer import MetricVisualizer
from termcolor import colored

from pyabsa import TCCheckpointManager, TCDatasetList, TADCheckpointManager
from pyabsa.functional.dataset.dataset_manager import AdvTCDatasetList, DatasetItem, detect_dataset
from textattack import Attacker
from textattack.attack_recipes import BAEGarg2019, PWWSRen2019, TextFoolerJin2019, PSOZang2020, IGAWang2019, \
    GeneticAlgorithmAlzantot2018, DeepWordBugGao2018, CLARE2020
from textattack.attack_results import SuccessfulAttackResult
from textattack.datasets import Dataset
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.attack_args import AttackArgs


@contextmanager
def timeout(duration: float):
    """
    A context manager that raises a `TimeoutError` after a specified time.

    Parameters
    ----------
    duration: float,
        the time duration in seconds,
        should be non-negative,
        0 for no timeout

    References
    ----------
    https://stackoverflow.com/questions/492519/timeout-on-a-function-call

    """
    if np.isinf(duration):
        duration = 0
    elif duration < 0:
        raise ValueError("duration must be non-negative")
    elif duration > 0:  # granularity is 1 second, so round up
        duration = max(1, int(duration))

    def timeout_handler(signum, frame):
        raise TimeoutError(f"block timedout after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)


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
        model = model
        model_wrapper = PyABSAModelWrapper(model)

        recipe = recipe_class.build(model_wrapper)
        # WordNet defaults to english. Set the default language to French ('fra')
        # recipe.transformation.language = "en"

        _dataset = [('', 0)]
        _dataset = Dataset(_dataset)

        self.attacker = Attacker(recipe, _dataset)


os.environ['PYTHONIOENCODING'] = 'UTF8'


def transfer_adversarial_attack_detection_and_defense(infer_files, tad_classifier, attacker):
    mv = MetricVisualizer()
    data = []
    for data_file in infer_files:
        print(colored("Attack: {}".format(data_file), 'green'))

        with open(data_file, mode='r', encoding='utf8') as fin:
            lines = fin.readlines()
            for line in lines:
                text, label = line.split('$LABEL$')
                text = text.strip()
                label = int(label.strip())
                data.append((text, label))

        all_num = 1e-10
        def_num = 1e-10
        acc_count = 0.
        def_acc_count = 0.
        det_acc_count = 0.
        random.shuffle(data)
        it = tqdm.tqdm(data[:100], postfix='testing ...')
        for text, label in it:
            try:
                result = attacker.attacker.simple_attack(text, label)
            except Exception as e:
                # del attacker
                torch.cuda.empty_cache()
                # attacker = SentAttacker(source_classifier, attack_recipes[target_attacker.lower()])
                print(e)
                continue
            if isinstance(result, SuccessfulAttackResult):
                infer_res = tad_classifier.infer(
                    result.perturbed_result.attacked_text.text + '!ref!{},{},{}'.format(
                        result.original_result.ground_truth_output, 1, result.perturbed_result.output),
                    print_result=False,
                    defense='pwws'
                )
                def_num += 1
                if infer_res['label'] == str(result.original_result.ground_truth_output):
                    def_acc_count += 1
                if infer_res['is_adv_label'] == '1':
                    det_acc_count += 1
            else:
                infer_res = tad_classifier.infer(
                    result.original_result.attacked_text.text + '!ref!{},{},{}'.format(
                        result.original_result.ground_truth_output, 1, result.perturbed_result.output),
                    print_result=False,
                )
            all_num += 1

            if infer_res['label'] == str(result.original_result.ground_truth_output):
                acc_count += 1
            it.postfix = colored('Det Acc:{}|TAD Acc: {}|Res Acc: {}'.format(
                round(det_acc_count / def_num * 100, 2),
                round(def_acc_count / def_num * 100, 2),
                round(acc_count / all_num * 100, 2)), 'green')
            it.update()
        mv.add_metric('Detection Accuracy', det_acc_count / def_num * 100)
        mv.add_metric('Defense Accuracy', def_acc_count / def_num * 100)
        mv.add_metric('Restored Accuracy', acc_count / all_num * 100)


# defense detection transfer experiment
if __name__ == '__main__':
    attack_recipes = {
        'bae': BAEGarg2019,
        'pwws': PWWSRen2019,
        'textfooler': TextFoolerJin2019,
        'pso': PSOZang2020,
        'iga': IGAWang2019,
        'ga': GeneticAlgorithmAlzantot2018,
        'wordbugger': DeepWordBugGao2018,
        'clare': CLARE2020,

    }
    for dataset in [
        'SST2',
        'Amazon',
        'AGNews10K',
    ]:
        for source_attacker in [
            # 'BAE',
            # 'PWWS',
            # 'TextFooler',
            'Multi-Attack',
        ]:
            for target_attacker in [
                # 'clare',
                # 'PSO',
                'GA',
                # 'wordbugger',
            ]:
                print(colored(
                    f'\n------------------- {dataset}{source_attacker} -> {dataset}{target_attacker} (DeBERTa) -------------------\n',
                    'green'))
                source_classifier = TADCheckpointManager.get_tad_text_classifier(checkpoint=f'TAD-{dataset}',
                                                                                 auto_device=True,
                                                                                 # Use CUDA if available
                                                                                 )

                # print(colored(f'\n------------------- {dataset}{source_attacker} -> {dataset}{target_attacker} (DeBERTa) -------------------\n', 'green'))
                # print(colored(f'******************** AdvDet Accuracy ********************', 'green'))
                # inference_sets = DatasetItem(target_attacker, findfile.find_cwd_files([target_attacker, dataset, '.adv', '.inference']))
                # results = source_classifier.batch_infer(target_file=inference_sets,
                #                                         # defense=True,
                #                                         print_result=False,
                #                                         save_result=False,
                #                                         ignore_error=False,
                #                                         )
                # print(colored(f'\n******************** AdvDet Accuracy ********************\n', 'green'))

                print(colored(f'\n******************** Restored Accuracy ********************\n', 'green'))
                testing_set = detect_dataset(DatasetItem(dataset), task='classification')['test']
                transfer_adversarial_attack_detection_and_defense(testing_set, source_classifier,
                                                                  SentAttacker(source_classifier,
                                                                               attack_recipes[target_attacker.lower()]))
                print(colored(f'\n******************** Restored Accuracy ********************\n', 'green'))
                print(colored(
                    f'\n------------------- {dataset}{source_attacker} -> {dataset}{target_attacker} (DeBERTa) -------------------\n',
                    'green'))
