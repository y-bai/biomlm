#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: Yong Bai, yong.bai@hotmail.com
@Time: 2024/4/21 8:56
@License: (C) Copyright 2013-2024. 
@File: corpus_files_gen.py
@Desc:

"""

import os
from typing import List

from datasets import load_from_disk, DatasetDict


def _gen_corpus_file(
        root_dir: str,
        txt_suffix: str,
        dataset_names: List[str] = ['chm13_t2t_20000_200', 'crgd_t2t_20000_200'],
        also_downsampling: bool = True,
        num_downsamples: int=400000,
):
    """Generate train corpus for training tokens with an extremely large data

    Parameters
    ----------
    root_dir : str
        _description_
    dataset_names : List[str], optional
        _description_, by default ['chm13_t2t_20000_200', 'crgd_t2t_20000_200']
    init_train_corpus : bool, optional
        _description_, by default True
    """

    out_dir = os.path.join(root_dir, 'tokens/train_corpus_txt')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    seqs = []
    for _name in dataset_names:
        _dir = os.path.join(root_dir, f'data/raw_dataset_{_name}')
        print(f"{''.join(['>'] * 3)} working on {_name}")
        # load dataset
        datasetdict: DatasetDict = load_from_disk(_dir)
        print(datasetdict)

        # only retrive train corpus
        seqs.extend(datasetdict['train']['sequence'])

    if also_downsampling:
        print(f"{''.join([' '] * 5)} ---- downsampling training corpus...")

        init_file_name = os.path.join(out_dir, f"train_corpus_{num_downsamples//1000}K_{txt_suffix}.txt")
        with open(init_file_name, 'w', encoding='utf-8') as init_f:
            for n_example in range(num_downsamples):
                init_f.write(seqs[n_example] + '\n')

    out_file_name = os.path.join(out_dir, f"train_corpus_{txt_suffix}.txt")

    print(f"generating whole training corpus...")
    with open(out_file_name, 'w', encoding='utf-8') as f:
        for _seq in seqs:
            f.write(_seq + '\n')
    print("======Done")


if __name__ == '__main__':
    # root_dir = r'/home/share/huadjyin/home/baiyong01/projects/biomlm/biomlm'
    root_dir = r'/jdfssz1/ST_HEALTH/P18Z10200N0124/AI/user/baiyong/projects/biomlm'

    # T2T
    # dataset_dir_names = ['chm13_t2t_20000_200', 'crgd_t2t_20000_200']

    # multi
    dataset_dir_names = ['multi_20000_200']

    # txt_suffix = 't2t_20000_200'
    txt_suffix = 'multi_20000_200' # need to downsample

    # For T2T
    # _gen_corpus_file(root_dir, txt_suffix, dataset_names=dataset_dir_names, also_downsampling=False,
    #                  num_downsamples=400000)
    _gen_corpus_file(root_dir, txt_suffix, dataset_names=dataset_dir_names, also_downsampling=True, num_downsamples=400000)


