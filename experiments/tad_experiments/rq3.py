# -*- coding: utf-8 -*-
# file: tsne_plot.py
# time: 19/02/2022
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import os
import pickle
import random
from time import sleep

import autocuda
import findfile
import numpy as np
import tikzplotlib
import torch
import tqdm
from matplotlib import pyplot as plt
from metric_visualizer import MetricVisualizer
from numpy import ndarray
from pandas import DataFrame

from anonymous_demo import TADCheckpointManager
from anonymous_demo.functional.dataset import DatasetItem
from textattack.models.wrappers import HuggingFaceModelWrapper

from textattack import Attacker
from textattack.datasets import Dataset

from textattack.attack_recipes import BAEGarg2019, PWWSRen2019, TextFoolerJin2019, PSOZang2020, IGAWang2019, \
    GeneticAlgorithmAlzantot2018, DeepWordBugGao2018, CLARE2020

from textattack.attack_results import SuccessfulAttackResult
from transformers import AutoTokenizer
from shapely.geometry import Polygon  # 多边形
from sklearn.manifold import TSNE

from sentence_transformers import util

from findfile import find_cwd_files

mv = MetricVisualizer()

tex_tsne_template = r"""
    \documentclass{article}
    \usepackage{pgfplots}
    \usepackage{tikz}
    \usepackage{caption}
    \usetikzlibrary{intersections}
    \usepackage{helvet}
    \usepackage[eulergreek]{sansmath}
    \usepackage{amsfonts,amssymb,amsmath,amsthm,amsopn}	% math related

    \begin{document}
        \pagestyle{empty}
        \pgfplotsset{ compat=1.3,every axis/.append style={
            grid = major,
            thick,
            font=\normalsize,
            ticks=none,
            line width = 1pt,
            tick style = {line width = 0.8pt}}}
        \pgfplotsset{every plot/.append style={thin}}

        \begin{figure}
        \centering

        $tikz_code$

        \end{figure}

    \end{document}
    """

tex_sim_template = r"""
    \documentclass{article}
    \usepackage{pgfplots}
    \usepackage{tikz}
    \usepackage{caption}
    \usetikzlibrary{intersections}
    \usepackage{helvet}
    \usepackage[eulergreek]{sansmath}
    \usepackage{amsfonts,amssymb,amsmath,amsthm,amsopn}	% math related

    \begin{document}
        \pagestyle{empty}
        \pgfplotsset{ compat=1.3,every axis/.append style={
            grid = major,
            thick,
            font=\normalsize,
            line width = 1pt,
            tick style = {line width = 1pt}}}
        \pgfplotsset{every plot/.append style={thin}}

        \begin{figure}
        \centering

        $tikz_code$

        \end{figure}

    \end{document}
    """

name_map = {
    'normal': 'Natural Example',
    'adversary': 'Adversarial Example',
    'restored': 'Repaired Example',
    # 'restored': 'Restored',
}

color_map = {
    # 'test': random.choice(mv.COLORS),
    # 'cross': random.choice(mv.COLORS),
    # 'mono': random.choice(mv.COLORS),
    # 'train': random.choice(mv.COLORS),
    # 'classic': random.choice(mv.COLORS)
    'normal': 'lawngreen',
    'adversary': 'red',
    'restored': 'cyan',
    # 'train': 'violet',
    # 'classic': 'darkviolet'
}

marker_map = {
    # 'test': random.choice(mv.MARKERS),
    # 'cross': random.choice(mv.MARKERS),
    # 'mono': random.choice(mv.MARKERS),
    # 'train': random.choice(mv.MARKERS),
    # 'classic': random.choice(mv.MARKERS)
    # 'normal': '.',
    # 'adversary': '^',
    # 'restored': 'P',
    # 'train': 'd',
    # # 'classic': '*'
    'normal': '.',
    'adversary': '.',
    'restored': '.',
}


def remove_outliers(data: ndarray):
    data = DataFrame(data)
    a = data.quantile(0.75)
    b = data.quantile(0.25)
    c = data
    c[(c >= (a - b) * 1.5 + a) | (c <= b - (a - b) * 1.5)] = np.nan
    c.fillna(c.median(), inplace=True)
    return np.array(c)


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


def cal_sentence_pair_similarity(embedding1, embedding2):
    cosine_score = util.pytorch_cos_sim(embedding1, embedding2)
    cosine_score = torch.sum(cosine_score)
    return cosine_score


def plot_embedding(normal_data, adversary_data, restored_data, avg_sim_score_normal, avg_sim_score_adv,
                   avg_sim_score_restored, dataset, attacker_name):
    ax = plt.subplot()
    # overlap1 = Cal_area_2poly(normal_data, restored_data)
    # overlap2 = Cal_area_2poly(normal_data, adversary_data)
    ax.scatter(normal_data[:, 0],
               normal_data[:, 1],
               label=name_map['normal'],
               c=color_map['normal'],
               marker=marker_map['normal'],
               )

    ax.scatter(adversary_data[:, 0],
               adversary_data[:, 1],
               label=name_map['adversary'],
               c=color_map['adversary'],
               marker=marker_map['adversary'],
               )

    ax.scatter(restored_data[:, 0],
               restored_data[:, 1],
               label=name_map['restored'],
               c=color_map['restored'],
               marker=marker_map['restored'],
               )

    plt.xticks([])
    plt.yticks([])
    plt.ylabel('{} ({})'.format(dataset.replace('TSNE', ''), attacker_name.upper()), fontsize=18)
    # plt.ylabel('', fontsize=18)
    # plt.xlabel(
    #     f'$\Delta$_{"{res}"}: {round(avg_sim_score_adv, 2)}% Res-$\Delta$_{"{res}"}: {round(avg_sim_score_restored, 2)}%')
    plt.legend(fontsize=14, loc=2)

    tikz_code = tikzplotlib.get_tikz_code()
    tex_src = tex_tsne_template.replace('$tikz_code$', tikz_code)
    tex_src = tex_src.replace(', only marks]', ',mark size=2, only marks]')
    fout = open('{}-{}.tex'.format(dataset, attacker_name), mode='w', encoding='utf8')
    fout.write(tex_src)
    fout.close()

    plt.show()
    plt.savefig('{}-{}.pdf'.format(dataset, attacker_name), dpi=1000)


def plot_dist_similarity_ecdf(normal_data, simscore1, adversary_data, simscore2, restored_data, simscore3, dataset, attacker):
    import seaborn as sns
    ax1 = plt.subplot()
    ax2 = plt.subplot()
    ax3 = plt.subplot()
    sns.ecdfplot(normal_data, linewidth=2, color='lightgreen', label='Natural', ax=ax1)
    sns.ecdfplot(adversary_data, linewidth=2, color='red', label='Adversary', ax=ax2)
    sns.ecdfplot(restored_data, linewidth=2, color='cyan', label='Repaired Adversary', ax=ax3)
    # plt.ylabel('Cumulative Distribution', fontsize=18)
    plt.xlabel('$\Delta_{adv}$: ' + str(round(simscore2, 2)) + '% $\Delta_{res}$: ' + '{}%'.format(round(simscore3, 2)), fontsize=18)
    plt.ylabel('{} ({})'.format(dataset, attacker.upper()), fontsize=18)
    plt.legend(fontsize=18, loc=3)
    plt.minorticks_on()
    plt.grid()
    tikz_code = tikzplotlib.get_tikz_code()
    tex_src = tex_sim_template.replace('$tikz_code$', tikz_code)
    with open('ecdf_{}_{}.tex'.format(dataset, attacker), mode='w', encoding='utf8') as fout:
        fout.write(tex_src)

    plt.show()


def tsne_plot(ckpt: str, attacker_name, num_points=1000):
    device = autocuda.auto_cuda()
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
    if not hasattr(__name__, 'tad_classifier'):
        tad_classifier = TADCheckpointManager.get_tad_text_classifier(ckpt)
        attacker = SentAttacker(tad_classifier, attack_recipes[attacker_name.lower()])
        tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')

    dataset = ckpt.split('-')[-1]

    normal_data = []
    adversary_data = []
    restored_data = []
    sim_score_normal = []
    sim_score_adversary = []
    sim_score_restored = []
    cache_path = '{}-{}-{}.pkl'.format(dataset, attacker_name, num_points)
    if os.path.exists(cache_path):
        with open(cache_path, mode='rb') as f:
            normal_data, adversary_data, restored_data, sim_score_normal, sim_score_adversary, sim_score_restored = pickle.load(
                f)
    else:
        # if True:
        normal_set = DatasetItem(attacker_name, find_cwd_files([attacker_name, dataset, 'test', '.org', '.inference']))
        for f in normal_set:
            with open(f, mode='r', encoding='utf8') as fin:
                lines = fin.readlines()
                for i in range(0, len(lines)):
                    text, labels = lines[i].strip().split('!ref!')[0], lines[i].strip().split('!ref!')[1]
                    result = attacker.attacker.simple_attack(text, int(labels.split(',')[0].strip()))
                    if len(adversary_data) >= num_points:
                        break
                    if isinstance(result, SuccessfulAttackResult):
                        infer_res = tad_classifier.infer(result.perturbed_result.attacked_text.text + '!ref!' + labels,
                                                         defense='pwws', print_result=True)
                        # if infer_res['label'] == str(result.original_result.ground_truth_output):
                        if infer_res['is_adv_label'] == '1':
                            normal_data.append(text)
                            adversary_data.append(result.perturbed_result.attacked_text.text)
                            restored_data.append(infer_res['restored_text'])

                            normal_ids = tokenizer(text, max_length=80, padding='max_length', truncation=True,
                                                   return_tensors="pt")
                            adversary_ids = tokenizer(result.perturbed_result.attacked_text.text, max_length=80,
                                                      padding='max_length', truncation=True, return_tensors="pt")
                            restored_ids = tokenizer(infer_res['restored_text'], max_length=80, padding='max_length',
                                                     truncation=True, return_tensors="pt")

                            embedding1 = tad_classifier.model(inputs=[normal_ids.input_ids.to(device)])['sent_logits']
                            embedding2 = tad_classifier.model(inputs=[adversary_ids.input_ids.to(device)])[
                                'sent_logits']
                            embedding3 = tad_classifier.model(inputs=[restored_ids.input_ids.to(device)])['sent_logits']

                            sim_score_normal.append(cal_sentence_pair_similarity(embedding1[0], embedding1[0]).item())
                            sim_score_adversary.append(
                                cal_sentence_pair_similarity(embedding1[0], embedding2[0]).item())
                            sim_score_restored.append(cal_sentence_pair_similarity(embedding1[0], embedding3[0]).item())

        pickle.dump(
            (normal_data, adversary_data, restored_data, sim_score_normal, sim_score_adversary, sim_score_restored),
            open(cache_path, mode='wb'))

    sim_score1 = np.sum(sim_score_normal) / np.sum(sim_score_normal)
    sim_score2 = np.sum(sim_score_adversary) / np.sum(sim_score_normal)
    sim_score3 = np.sum(sim_score_restored) / np.sum(sim_score_normal)
    print('average similarity score of original text and original text: {}'.format(sim_score1))
    print('average similarity score of original text and adversarial text: {}'.format(sim_score2))
    print('average similarity score of original text and restored text: {}'.format(sim_score3))
    plot_dist_similarity_ecdf(sim_score_normal, sim_score1, sim_score_adversary, sim_score2, sim_score_restored, sim_score3, dataset, attacker_name)

    random.shuffle(normal_data)
    random.shuffle(adversary_data)
    random.shuffle(restored_data)
    output = None
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, len(normal_data + adversary_data + restored_data), 10)):
            ids = tokenizer((normal_data + adversary_data + restored_data)[i:i + 10], max_length=80,
                            padding='max_length', truncation=True, return_tensors="pt")
            ids = ids.to(device)

            outputs = tad_classifier.model(inputs=[ids.input_ids])
            feat = outputs['last_hidden_state'].cpu()
            feat = feat.view(10, -1)
            try:
                output = np.concatenate((output, feat), axis=0) if output is not None else feat
            except Exception as e:
                print(e)

    manifold = TSNE(n_components=2, init='random', learning_rate=200, perplexity=30)

    result = manifold.fit_transform(output)
    normal_res = result[:len(normal_data)]
    adversary_res = result[len(normal_data):len(normal_data) + len(adversary_data)]
    restored_res = result[len(normal_data) + len(adversary_data):]

    return plot_embedding(normal_res,
                          adversary_res,
                          restored_res,
                          sim_score1,
                          sim_score2,
                          sim_score3,
                          manifold.__class__.__name__ + dataset,
                          attacker_name)


def Cal_area_2poly(data1, data2):
    """
    任意两个图形的相交面积的计算
    :param data1: 当前物体
    :param data2: 待比较的物体
    :return: 当前物体与待比较的物体的面积交集
    """

    poly1 = Polygon(data1).convex_hull  # Polygon：多边形对象
    poly2 = Polygon(data2).convex_hull
    if not poly1.intersects(poly2):
        inter_area = 0  # 如果两四边形不相交
    else:
        inter_area = poly1.intersection(poly2).area  # 相交面积
    return 2 * inter_area / (poly2.area + poly1.area) * 100


if __name__ == '__main__':
    ckpts = [
        # 'TAD-SST2',
        # 'TAD-AGNews10K',
        'TAD-Amazon',
    ]
    attacker_names = [
        'pwws',
        'textfooler',
        'bae',
    ]
    REPEAT = 1
    for ckpt in ckpts:
        for attacker_name in attacker_names:
            for i in range(REPEAT):
                random.seed(i)
                tsne_plot(ckpt, attacker_name, num_points=500)

for f in findfile.find_cwd_files('.tex', exclude_key='.pdf', recursive=1):
    os.system('pdflatex {}'.format(f))

for f in findfile.find_cwd_files('.pdf', exclude_key='.tex', recursive=1):
    os.system('pdfcrop {} {}'.format(findfile.find_cwd_file([f]), findfile.find_cwd_file([f])))

for f in findfile.find_cwd_files(or_key=['.log', '.aux', '.out'], recursive=1):
    os.remove(f)
