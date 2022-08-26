# -*- coding: utf-8 -*-
# file: tsne_plot.py
# time: 19/02/2022
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import os
import pickle
import random

import autocuda
import numpy as np
import tikzplotlib
import torch
import tqdm
from findfile import find_cwd_files, find_cwd_dirs
from matplotlib import pyplot as plt
from metric_visualizer import MetricVisualizer
from numpy import ndarray
from pandas import DataFrame
from pyabsa import TADCheckpointManager
from pyabsa.functional.dataset import DatasetItem
from scipy import stats
from sklearn.cluster import KMeans
from textattack.models.wrappers import HuggingFaceModelWrapper

from textattack import Attacker
from textattack.datasets import Dataset

from textattack.attack_recipes import BAEGarg2019, PWWSRen2019, TextFoolerJin2019, PSOZang2020, IGAWang2019, GeneticAlgorithmAlzantot2018, DeepWordBugGao2018, CLARE2020

from textattack.attack_results import SuccessfulAttackResult
from transformers import DebertaV2Model, AutoConfig, AutoTokenizer
from shapely.geometry import Polygon  # 多边形
from sklearn.manifold import TSNE

mv = MetricVisualizer()
from findfile import find_cwd_files, find_cwd_dirs

tex_template = r"""
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
            xscale=0.85,
            line width = 1pt,
            tick style = {line width = 0.8pt}}}
        \pgfplotsset{every plot/.append style={thin}}

        \begin{figure}
        \centering

        $tikz_code$

        \end{figure}

    \end{document}
    """

name_map = {
    'normal': 'Normal',
    'adversary': 'Adversary',
    'restored': 'Restored',
}

color_map = {
    # 'test': random.choice(mv.COLORS),
    # 'cross': random.choice(mv.COLORS),
    # 'mono': random.choice(mv.COLORS),
    # 'train': random.choice(mv.COLORS),
    # 'classic': random.choice(mv.COLORS)
    'normal': 'orange',
    'adversary': 'lawngreen',
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
    'normal': '.',
    'adversary': '^',
    'restored': 'P',
    # 'train': 'd',
    # 'classic': '*'
}

order_map = {
    'normal': 0,
    'adversary': 2,
    'restored': 3,
    # 'train': 1,
    # 'classic': 4

}


def remove_outliers(data: ndarray):
    data = DataFrame(data)
    a = data.quantile(0.75)
    b = data.quantile(0.25)
    c = data
    c[(c >= (a - b) * 1.5 + a) | (c <= b - (a - b) * 1.5)] = np.nan
    c.fillna(c.median(), inplace=True)
    return np.array(c)


def plot_embedding(normal_data, adversary_data, restored_data, dataset, attacker_name):
    # data = np.concatenate((src_data, tgt_data), 0)
    # x_min, x_max = np.min(data, 0), np.max(data, 0)
    # src_data = (src_data - x_min) / (x_max - x_min)
    # tgt_data = (tgt_data - x_min) / (x_max - x_min)

    normal_data = remove_outliers(normal_data)
    adversary_data = remove_outliers(adversary_data)
    restored_data = remove_outliers(restored_data)

    ax = plt.subplot()
    # overlap = Cal_area_2poly(normal_data, adversary_data)
    # skew = stats.skew(np.concatenate((src_data, tgt_data), axis=0))
    # skew = stats.skew(adversary_data)
    # skew = round(float(abs(skew[0]) + abs(skew[1])), 2)
    # kurtosis = stats.kurtosis(tgt_data)
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

    # for i in range(data.shape[0]):
    #     plt.text(data[i, 0], data[i, 1], str(label[i]),
    #              color=colors[label[i]],
    #              fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    # plt.xlabel('Overlap: {}% Skewness: {}'.format(round(overlap, 2), skew), fontsize=18)
    plt.ylabel('t-SNE'.format(), fontsize=18)
    plt.legend(fontsize=14, loc=2)
    # plt.title(title + ' Src_dense: {} Tgt_dense'.format(round(src_dense, 2)), round(tgt_dense, 2))
    # plt.savefig('{}--{}-{}-{}-Overlap-{}.pdf'.format(title, order_map[tgt_label], src_label, tgt_label, overlap), dpi=1000)
    # plt.title(title + ' Overlap: {}'.format(round(overlap, 2)))
    tikz_code = tikzplotlib.get_tikz_code()
    tex_src = tex_template.replace('$tikz_code$', tikz_code)

    fout = open('{}-{}.tex'.format(dataset, attacker_name), mode='w', encoding='utf8')
    fout.write(tex_src)
    fout.close()

    plt.show()
    plt.savefig('{}-{}.pdf'.format(dataset, attacker_name), dpi=1000)


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

def tsne_plot(ckpt: str, attacker_name):
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
    tad_classifier = TADCheckpointManager.get_tad_text_classifier(ckpt)
    attacker = SentAttacker(tad_classifier, attack_recipes[attacker_name.lower()])

    dataset = ckpt.split('-')[-1]

    normal_data = []
    adversary_data = []
    restored_data = []

    cache_path = '{}-{}.pkl'.format(dataset, attacker_name)
    if os.path.exists(cache_path):
        with open(cache_path, mode='rb') as f:
            normal_data, adversary_data, restored_data = pickle.load(f)
    else:
        normal_set = DatasetItem(attacker_name, find_cwd_files([attacker_name, dataset, '.org', '.inference']))
        for f in normal_set:
            with open(f, mode='r', encoding='utf8') as fin:
                lines = fin.readlines()
                for i in range(0, len(lines)):
                    text, labels = lines[i].strip().split('!ref!')[0], lines[i].strip().split('!ref!')[1]
                    normal_data.append(text)
                    try:
                        result = attacker.attacker.simple_attack(text, int(labels.split(',')[0].strip()))
                    except Exception as e:
                        print(e)
                        continue
                    if isinstance(result, SuccessfulAttackResult):
                        adversary_data.append(result.perturbed_result.attacked_text.text)
                        infer_res = tad_classifier.infer(result.perturbed_result.attacked_text.text + '!ref!' + labels, defense='pwws', print_result=True)
                        # if infer_res['label'] == str(result.original_result.ground_truth_output):
                        if infer_res['is_adv_label'] == '1':
                            restored_data.append(result.perturbed_result.attacked_text.text)

        pickle.dump((normal_data, adversary_data, restored_data), open(cache_path, mode='wb'))

    random.shuffle(normal_data)
    random.shuffle(adversary_data)
    random.shuffle(restored_data)

    # normal_data = normal_data
    # adversary_data = adversary_data[:len(normal_data)]
    # normal_data = normal_data[:500]
    # adversary_data = adversary_data[:500]
    # adversary_data = adversary_data

    pretrained_config = AutoConfig.from_pretrained('microsoft/deberta-v3-base')
    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
    PTM = DebertaV2Model(pretrained_config).to(tad_classifier.opt.device)

    output = None
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, len(normal_data + adversary_data + restored_data), 16)):
            ids = tokenizer((normal_data + adversary_data + restored_data)[i:i + 16], max_length=80, padding='max_length', truncation=True, return_tensors="pt")
            ids = ids.to(device)

            feat = PTM(**ids)['last_hidden_state']
            # feat = feat.view(16, -1).cpu().numpy()
            length = np.count_nonzero(feat.cpu().numpy(), axis=1)
            feat = feat.cpu().sum(dim=1)
            feat = feat / length
            try:
                output = np.concatenate((output, feat), axis=0) if output is not None else feat
            except:
                pass
    tsne = TSNE(n_components=2, init='pca', learning_rate=200, perplexity=50)
    result = tsne.fit_transform(output)
    return plot_embedding(result[:len(normal_data)],
                          result[len(normal_data):len(adversary_data)],
                          result[len(adversary_data):],
                          dataset,
                          attacker_name)


def Cal_area_2poly(data1, data2):
    """
    任意两个图形的相交面积的计算
    :param data1: 当前物体
    :param data2: 待比较的物体
    :return: 当前物体与待比较的物体的面积交集
    """

    a = Polygon(data1)
    b = a.convex_hull
    poly1 = Polygon(data1).convex_hull  # Polygon：多边形对象
    poly2 = Polygon(data2).convex_hull
    if not poly1.intersects(poly2):
        inter_area = 0  # 如果两四边形不相交
    else:
        inter_area = poly1.intersection(poly2).area  # 相交面积
    return 2 * inter_area / (poly2.area + poly1.area) * 100


if __name__ == '__main__':
    ckpts = [
        'TAD-SST2',
        'TAD-Amazon',
        'TAD-AGNews',
    ]
    attacker_names = [
        'pwws',
        'textfooler',
    ]
    REPEAT = 3
    for i in range(REPEAT):
        for ckpt in ckpts:
            for attacker_name in attacker_names:
                random.seed(i)
                tsne_plot('TAD-SST2', attacker_name)
