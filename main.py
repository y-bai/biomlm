#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   main.py
@Time    :   2024/01/16 21:18:23
@Author  :   Yong Bai 
@Contact :   baiyong at genomics.cn
@License :   (C)Copyright 2023-2024, Yong Bai
@Desc    :   None
'''

import os
from datasets import load_from_disk
import sentencepiece as spm

def _batch_iterator(data_path, batch_size=100):
    
    num_examples = 10000
    data_dict = load_from_disk(data_path)
    train_dataset = data_dict['train'].select(range(num_examples))
    for i in range(0, num_examples, batch_size):
        yield train_dataset[i : i + batch_size]['sequence']


if __name__ == '__main__':
    i_ds_name = 'chm13_t2t_20000_200'
    root_dir = r'/home/share/huadjyin/home/baiyong01/projects/biomlm/biomlm'
    data_path = os.path.join(root_dir, f'datasets/raw_dataset_{i_ds_name}')

    _data_iterator = _batch_iterator(data_path)

    # https://github.com/google/sentencepiece/blob/master/doc/options.md
    spm.SentencePieceTrainer.Train(
        # sentence_iterator=_data_iterator,
        # input_format='text',
        input='/home/share/huadjyin/home/baiyong01/projects/biomlm/biomlm/tokens/train_corpus/init_train_corpus/init_train_corpus_chm13_t2t_20000_200.txt',
        vocab_size=1000,
        character_coverage=1.0,
        model_prefix='sp_vocab',
        bos_id=0,
        unk_id=1,
        eos_id=2,
        unk_piece='<UNK>',
        bos_piece='<BOS>',
        eos_piece='<EOS>',
        train_extremely_large_corpus=True,
        num_threads=10,
        max_sentence_length=20000,
        add_dummy_prefix=False, # https://github.com/google/sentencepiece/issues/488
        # hard_vocab_limit=False,
        # unk_surface='<UNK>'
    )

    # seq = 'CTATACCAAGATCTCTCCCCAGAAACAAACCCAAATCTTACTATATGTTATGGCACGCTATGATGATGAGCAGCGATGAGCAGCCGAAGCCTCAAGGAAGGGATGCTTTTGTAAAACAAGACTTGTGGAATATAACATGTGAAAGTAAAGCCCACGGCAGAGCTCCCTCCTCAGCACACGGGGAGCAGACAGGAAGTTTTTCCTCACCTTCCTCAATGGCCTGCAGCCACGTCTCCCAGGTCAGTCTTAA'
    # tokenizer = spm.SentencePieceProcessor(model_file="sp_vocab.model")
    # print(tokenizer._alpha)
    # print(tokenizer._nbest_size)
    # print(tokenizer._enable_sampling) # default is False, if Ture, then enconded str is different every time
    # # return _sentencepiece.SentencePieceProcessor__EncodeAsPieces(self, text, enable_sampling, nbest_size, alpha, add_bos, add_eos, reverse, emit_unk_piece)
    # # https://github.com/google/sentencepiece/blob/master/python/src/sentencepiece/__init__.py#L471
    # print(tokenizer.encode(seq, out_type=str, add_bos=True, add_eos=True))
    # print(tokenizer.encode(seq, out_type=int, add_bos=True, add_eos=True))
