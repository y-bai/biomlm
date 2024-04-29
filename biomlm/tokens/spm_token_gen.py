#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		spm_token_gen.py
@Time    :   	2024/04/21 12:13:58
@Author  :   	Yong Bai 
@Contact :   	baiyong at genomics.cn
@License :   	(C)Copyright 2023-2024, Yong Bai

                Licensed under the Apache License, Version 2.0 (the "License");
                you may not use this file except in compliance with the License.
                You may obtain a copy of the License at

                    http://www.apache.org/licenses/LICENSE-2.0

                Unless required by applicable law or agreed to in writing, software
                distributed under the License is distributed on an "AS IS" BASIS,
                WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
                See the License for the specific language governing permissions and
                limitations under the License.

@Desc    :   	None
# https://github.com/google/sentencepiece#train-sentencepiece-model
"""

import os, shutil
from datetime import datetime
import time

import sentencepiece as spm


def _gen_token(
        vocab_size, 
        root_dir, 
        train_corpus_fname,
        species: str = "T2T",
        token_type: str="SPM_BPE",
        seq_len_type:str='20000_200',
    ):

    print('#########################################')
    print(f'vocab_size: {vocab_size}')

    # load dataset from disk

    saved_dir = os.path.join(root_dir, f'tokens/{seq_len_type}/{species}/{token_type}/{vocab_size}')
    # if os.path.isdir(saved_dir):
    #     shutil.rmtree(saved_dir)
    os.makedirs(saved_dir, exist_ok=True)
    print(f"tokens saved path: {saved_dir}")

    if token_type == "SPM_BPE":
        model_type = 'bpe'
    elif token_type == "SPM_Unigram":
        model_type = 'unigram'  # default

    model_prefix = os.path.join(saved_dir, 'spm_vocab')

    input_fname = os.path.join(root_dir, f'tokens/train_corpus_txt/{train_corpus_fname}')

    print(f"{token_type} starts, "+ datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
    start_t = time.time()
    spm.SentencePieceTrainer.Train(
        input=input_fname,
        vocab_size=vocab_size,
        model_prefix=model_prefix,
        model_type=model_type,
        character_coverage=1.0,
        bos_id=0,
        unk_id=1,
        eos_id=2,
        unk_piece='<UNK>',
        bos_piece='<BOS>',
        eos_piece='<EOS>',
        train_extremely_large_corpus=True,
        num_threads=32,
        num_sub_iterations=8,
        max_sentence_length=int(seq_len_type.split('_')[0]),
        add_dummy_prefix=False, # https://github.com/google/sentencepiece/issues/488
    )
    end_t = time.time()
    print(f"running duration: {end_t-start_t} s")
    print(f"{token_type} finished, "+ datetime.now().strftime("%m/%d/%Y %H:%M:%S"))

    print('---------------Done-----------------')


if __name__ == "__main__":
    species = 'T2T'
    token_type = 'SPM_BPE' # BPE
    train_corpus_fname = 'train_corpus_t2t_20000_200.txt'
    seq_len_type = '20000_200'

    # root_dir = r'/home/share/huadjyin/home/baiyong01/projects/biomlm/biomlm'
    root_dir = r'/jdfssz1/ST_HEALTH/P18Z10200N0124/AI/user/baiyong/projects/biomlm'

    vocab_sizes = [5008, 10008, 50008, 100008, 150008, 200008]

    _gen_token(
        vocab_sizes[0],
        root_dir,
        species=species,
        train_corpus_fname=train_corpus_fname,
        token_type=token_type,
        seq_len_type=seq_len_type
    )