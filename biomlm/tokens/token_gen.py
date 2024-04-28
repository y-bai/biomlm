#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :      token_gen.py
@Time    :      2024/01/16 21:59:02
@Author  :      Yong Bai 
@Contact :      baiyong at genomics.cn
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

@Desc    :      None

"""
from pathlib import Path
import sys
import os
from typing import List

import os, shutil
from datetime import datetime
import time
import json
from collections import OrderedDict
import copy

from datasets import load_from_disk, Dataset, DatasetDict

sys.path.append(
    str(Path(__file__).resolve().parents[1]) #
) 
from models.tokenizer import (
    BioSeqBaseBPETokenizer, 
    BioSeqBaseUnigramTokenizer
)

def _batch_iterator(ds_dicts: List[DatasetDict], batch_size: int=1000, max_num_examples:int=-1):
    for i_ds_dict in ds_dicts:
        i_ds = i_ds_dict['train']
        num_sample = len(i_ds) if max_num_examples == -1 else max_num_examples
        for i in range(0, num_sample, batch_size):
            yield i_ds[i : i + batch_size]['sequence']

def _gen_token(
        vocab_size, 
        root_dir, 
        raw_dataset_names: List[str] = ['chm13_t2t_20000_200'],
        species: str = "T2T",
        token_type: str="BPE",
        seq_len_type:str='20000_200',
    ):
    """To train tokenizer with huge dataset where 

    Parameters
    ----------
    vocab_size : _type_
        _description_
    root_dir : _type_
        _description_
    species : str, optional
        _description_, by default "T2T"
    raw_dataset_name : List[str], optional
        _description_, by default ['chm13_t2t_20000_200', 'crgd_t2t_20000_200']
    token_type : str, optional
        _description_, by default "BPE"
    """

    print('#########################################')
    print(f'vocab_size: {vocab_size}')

    # load dataset from disk

    ds_dicts = []
    total_len = 0
    for i_ds_name in raw_dataset_names:
        i_path = os.path.join(root_dir, f'datasets/raw_dataset_{i_ds_name}')
        print(f'>>>loading raw dataset: {i_path}')
        i_ds_dict = load_from_disk(i_path)
        total_len += len(i_ds_dict['train'])
        ds_dicts.append(i_ds_dict)
    
    # raw_iterabledatasets:IterableDataset = concatenate_datasets(iterabledataset_list)
    # raw_iterabledatasets = raw_iterabledatasets._resolve_features()

    saved_dir = os.path.join(root_dir, f'tokens/{seq_len_type}/{species}/{token_type}/{vocab_size}')
    # if os.path.isdir(saved_dir):
    #     shutil.rmtree(saved_dir)
    os.makedirs(saved_dir, exist_ok=True)
    print(f"init tokens saved path: {saved_dir}")

    data_iterator = _batch_iterator(
        ds_dicts,
        batch_size=1000,
        max_num_examples=-1 if species=='T2T' else 400000)

    if token_type == 'BPE':
        print("BPE starts, "+ datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
        start_t = time.time()
        
        _tokenizer = BioSeqBaseBPETokenizer()
        _tokenizer.train_from_iterator(
            data_iterator,
            vocab_size=vocab_size,
            min_frequency=2,
            max_token_length=16, # actural 15
            length=total_len if species=='T2T' else 400000, # In order to improve the look of our progress bars, we can specify the total length of the dataset
            special_tokens= ["<BOS>", "<UNK>", "<EOS>"],
        )
        _tokenizer.save_model(saved_dir)
        end_t = time.time()
        print(f"running duration: {end_t-start_t} s")
        print("BPE finished, "+ datetime.now().strftime("%m/%d/%Y %H:%M:%S"))

    if token_type == 'Unigram':
        print("Unigram starts, "+ datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
        start_t = time.time()

        _tokenizer = BioSeqBaseUnigramTokenizer()
        _tokenizer.train_from_iterator(
            data_iterator,
            vocab_size=vocab_size,
            length=total_len if species=='T2T' else 400000,
            special_tokens= ["<BOS>", "<UNK>", "<EOS>"],
        )
        _tokenizer.save_model(saved_dir)
        end_t = time.time()
        print(f"running duration: {end_t-start_t} s")
        print("Unigram finished, "+ datetime.now().strftime("%m/%d/%Y %H:%M:%S"))

    print('---------------Done-----------------')

    # # load all dataset couses memory leak. 
    # +++++++++++++++++++++Old methods
    # if token_type == "BPE": 
    #     data_len = get_corpus_len(dataset) 
    #     data_iterator = get_corpus_iterator(dataset, batch_size=1000)
    # else:
        # for Unigram.
        # This is to reduce the dadaset, otherwise error raised when using Unigram model: 
        # pyo3_runtime.PanicException: called `Result::unwrap()` on an `Err` value: Interna
    # if seq_len <= 1000:
    #     sel_range = range(2400000) if token_type == 'BPE' else range(2100000) # This the almost max number of samples for token training
    # else:
    #     sel_range = range(1200000)
    # dt = {'seq': dataset["train"].shuffle(seed=42).select(sel_range)}
    # data_len = get_corpus_len(dt)
    # data_iterator = get_corpus_iterator(dt, batch_size=1000)

    # print(f"data_len: {data_len}")
    
    # saved_dir = os.path.join(root_dir, f'mambaDNA/tokens/{species}/{seq_len//1000}K/{token_type}/{vocab_size}')

    # if os.path.isdir(saved_dir):
    #     shutil.rmtree(saved_dir)
    # os.makedirs(saved_dir)

    # if token_type == 'BPE':
    #     print("BPE strats, "+ datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
    #     start_t = time.time()
        
    #     dna_tokenizer = BioSeqBPETokenizer()
    #     dna_tokenizer.train_from_iterator(
    #         data_iterator,
    #         vocab_size=vocab_size,
    #         min_frequency=2,
    #         max_token_length=16, # should be 15
    #         length=data_len,
    #         special_tokens= ["<BOS>", "<UNK>", "<EOS>", "<MASK>"],
    #     )
    #     dna_tokenizer.save_model(saved_dir)
        
    #     ####################
    #     # save the model
    #     # dna_tokenizer.save_pretained(saved_dir)
        
    #     end_t = time.time()
    #     print(f"running duration: {end_t-start_t} s")
    #     print("BPE finished, "+ datetime.now().strftime("%m/%d/%Y %H:%M:%S"))

    # if token_type == 'Unigram':
    #     print("Unigram strats, "+ datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
    #     start_t = time.time()

    #     dna_tokenizer = BioSeqUnigramTokenizer()
    #     dna_tokenizer.train_from_iterator(
    #         data_iterator,
    #         vocab_size=vocab_size,
    #         length=data_len,
    #         special_tokens= ["<BOS>", "<UNK>", "<EOS>", "<MASK>"],
    #     )
    #     dna_tokenizer.save_model(saved_dir)
    #     # dna_tokenizer.save_pretained(saved_dir)

    #     end_t = time.time()
    #     print(f"running duration: {end_t-start_t} s")
    #     print("Unigram finished, "+ datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
    # +++++++++++++++++++++


def _change_special_token(
        vocab_size, root_dir, 
        seq_len: int=1000, 
        species: str="T2T", 
        token_type: str="BPE",
        n_original_special_token: int=3
):

    token_dir = os.path.join(root_dir, f'mambaDNA/tokens/{species}/{seq_len//1000}K/{token_type}/{vocab_size}')

    if token_type == "BPE":
        vocab_file = os.path.join(token_dir, 'vocab.json')
        if os.path.exists(vocab_file):
            with open(vocab_file, 
                        mode='rt', 
                        encoding="utf-8"
                ) as vocab_f:    
                vocab_json = json.load(vocab_f, object_pairs_hook=OrderedDict)
            
            # remove the first 7 keys, ie, ["<CLS>", "<PAD>", "<SEP>", "<UNK>", "<MASK>", "<BOS>", "<EOS>"]
            # and add the updated keys 
            keys_list = ["<BOS>", "<UNK>", "<EOS>", "<MASK>","<PAD>"] + list(vocab_json.keys())[n_original_special_token:]
            vals = list(range(len(keys_list)))
            update_vocab = OrderedDict(zip(keys_list, vals))

            with open(vocab_file, 
                        mode='w', 
                        encoding="utf-8"
                ) as vocab2_f:    
                json.dump(update_vocab, vocab2_f)
    
    if token_type == "Unigram":
        vocab_file = os.path.join(token_dir, 'unigram.json')
        if os.path.exists(vocab_file): 
            with open(vocab_file, 
                    mode='rt', 
                    encoding="utf-8"
                ) as vocab_f:
                
                vocab_json = json.load(vocab_f, object_pairs_hook=OrderedDict)
            vocab_json.update(
                {
                    'unk_id': 1, 
                    'vocab': [['<BOS>', 0.0], ['<UNK>', 0.0], ['<EOS>', 0.0], ['<MASK>', 0.0],['<PAD>', 0.0]] + vocab_json['vocab'][n_original_special_token:]
                }
            )
            with open(vocab_file, 
                    mode='w', 
                    encoding="utf-8"
                ) as vocab2_f:    
                json.dump(vocab_json, vocab2_f)


def _clear_files(
        vocab_size, root_dir, 
        seq_len: int=1000, 
        species: str="T2T", 
        token_type: str="BPE"
):
    src_token_dir = os.path.join(root_dir, f'mambaDNA/tokens/{species}/{seq_len//1000}K/{token_type}/{vocab_size}')  # 3008
    # des_token_dir = os.path.join(root_dir, f'mambaDNA/tokens/{species}/{seq_len//1000}K/{token_type}/{vocab_size}')

    if os.path.isdir(src_token_dir):
        # # remove unwanted files
        for fname in ['special_tokens_map.json', 'tokenizer_config.json', 'tokenizer.json']:
            _fname = os.path.join(src_token_dir, fname)
            if os.path.exists(_fname):
                os.remove(_fname)
        # shutil.move(src_token_dir, des_token_dir)


if __name__ == "__main__":

    # nohup /home/HPCBase/tools/anaconda3/bin/python /home/share/huadjyin/home/baiyong01/projects/biomlm/biomlm/token_gen.py --seqlen 1000 --species T2T --token BPE > bpe_t2t_1k.out &

    # parser = argparse.ArgumentParser()

    # parser.add_argument(
    #     "--seqlen",
    #     default=1000,
    #     type=int,
    #     required=True,
    #     help="sequence len, could be 1000 or 2000",
    # )

    # parser.add_argument(
    #    "--species",
    #     default="T2T",
    #     type=str,
    #     required=True,
    #     help="sequence species type, could be T2T or Multi_species", 
    # )

    # parser.add_argument(
    #     "--token",
    #     default="BPE",
    #     type=str,
    #     required=True,
    #     help="tokeniner model, could be BPE or Unigram", 
    # )

    # args = parser.parse_args()

    # seq_len = args.seqlen
    # species = args.species
    # token_type = args.token

    species = 'T2T'
    token_type = 'BPE' # BPE
    raw_datasets = ['chm13_t2t_20000_200', 'crgd_t2t_20000_200']
    seq_len_type='20000_200'

    root_dir = r'/home/share/huadjyin/home/baiyong01/projects/biomlm/biomlm'

    vocab_sizes =  [5008, 10008, 50008, 100008, 150008, 200008]

    # >>>NOTE: Ungram is not working on huge dataset, because it raise the errer:
    # thread '<unnamed>' panicked at /root/.cargo/registry/src/index.crates.io-6f17d22bba15001f/esaxx-rs-0.1.10/src/lib.rs:83:26:
    # called `Result::unwrap()` on an `Err` value: TryFromIntError(())
    # note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
    # Traceback (most recent call last):
    # File "token_gen.py", line 324, in <module>
    #     _gen_token(
    # File "token_gen.py", line 126, in _gen_token
    #     _tokenizer.train_from_iterator(
    # File "/jdfssz1/ST_HEALTH/P18Z10200N0124/AI/user/baiyong/projects/biomlm/tokenizer/_unigram_tokenizer.py", line 161, in train_from_iterator
    #     self._tokenizer.train_from_iterator(
    # pyo3_runtime.PanicException: called `Result::unwrap()` on an `Err` value: TryFromIntError(())

    # >>> NOTE: run 'BPE' with huge training dataset on the machine with huge memory.
    _gen_token(
        vocab_sizes[0],
        root_dir,
        species=species,
        raw_dataset_names=raw_datasets,
        token_type=token_type,
        seq_len_type=seq_len_type
    )

    # clear and modify the token folder and files
    # for vocab_size in vocab_sizes:
    #     _clear_files(
    #         vocab_size, root_dir, 
    #         seq_len=seq_len, 
    #         species=species, 
    #         token_type=token_type) 

    # update token
    # for vocab_size in vocab_sizes:
    #     _change_special_token(
    #         vocab_size, root_dir, 
    #         seq_len=seq_len, 
    #         species=species, 
    #         token_type=token_type,
    #         n_original_special_token=4)

    # BioSeqPreTrainedTokenizerFast model
    # for vocab_size in vocab_sizes:
    #     _save_pretrained_tokenizer(
    #         vocab_size, root_dir, 
    #         seq_len=seq_len, 
    #         species=species, 
    #         token_type=token_type)
    
    # for vocab_size in [3009]:
    #     _load_pretrained_tokenizer(
    #         vocab_size, root_dir, 
    #         seq_len=seq_len, 
    #         species=species, 
    #         token_type=token_type)


