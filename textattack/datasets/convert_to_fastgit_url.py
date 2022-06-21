# -*- coding: utf-8 -*-
# file: conver_to_fastgit_url.py
# time: 10/04/2022
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://hub.fastgit.xyz/yangheng95
# Copyright (C) 2021. All Rights Reserved.

from findfile import find_cwd_files

for py in find_cwd_files('.py', exclude_key='convert'):
    with open(py, mode='r', encoding='utf8') as fin:
        lines = fin.readlines()
        for i in range(len(lines)):
            lines[i] = lines[i].replace('github.com', 'hub.fastgit.xyz')
            lines[i] = lines[i].replace('raw.githubusercontent.org', 'raw.fastgit.org')
            lines[i] = lines[i].replace('assets.github.com', 'assets.fastgit.org')
    with open(py, mode='w', encoding='utf8') as fout:
        fout.writelines(lines)
