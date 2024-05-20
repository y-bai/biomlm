#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		_tokenizer_map.py
@Time    :   	2024/04/08 17:15:07
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

"""
from itertools import chain

from typing import List, Optional, Union
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import PreTrainedTokenizerFast
from datasets import IterableDataset, Dataset, IterableDatasetDict, DatasetDict
from transformers.utils import logging

logger = logging.get_logger(__name__)

class BioSeqTokenizerMapMinxi:
    pass
        
class BioSeqTokenizerMap(BioSeqTokenizerMapMinxi):
    """Using the tokenizer to tokenize the dataset with `map` function

        Parameters
        ----------
        tokenizer : Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
            tokenizer
    """
    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        max_length: int = 512,
        stride: int = 10,
        min_len_frac: float = 0.8,
        streaming: bool=True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.streaming = streaming
        self.min_len_frac = min_len_frac
    
    def do_map(
        self,
        input_dataset: Union[IterableDatasetDict, DatasetDict],
        dataset_col_remove: Optional[List[str]] = None,
        dataset_col_tokenize: Optional[str] = None,
        padding: Union[bool, str] = "max_length",
        truncation: Union[bool, str] = True,
        return_overflowing_tokens: bool=True, 
        load_from_cache_file: bool=True,
        num_proc: int = 4,
    ):
        def tokenize_fn(untokenized_dataset):
            token_encodes = self.tokenizer(
                untokenized_dataset[dataset_col_tokenize],
                max_length=self.max_length,
                padding=False if self.streaming else padding,
                truncation=False if self.streaming else truncation,
                # return_overflowing_tokens has different behavior between slow tokenizer and fast tokenizer
                # see: https://github.com/huggingface/transformers/issues/23001
                return_overflowing_tokens=False if self.streaming else return_overflowing_tokens,
                stride= 0 if self.streaming else self.stride,
                add_special_tokens=True, # default value in the __call__ function
                # return_special_tokens_mask=False if self.streaming else True, # for performance improvement
            )
            # return token_encodes
            return {
                "input_ids": token_encodes["input_ids"],
                # "token_type_ids": token_type_batch,
                "attention_mask": token_encodes["attention_mask"],
            }
        
        _min_frac = 0 if self.streaming else self.min_len_frac
        min_tokenized_seq_len = int(self.max_length * _min_frac)

        def filter_fn(tokenized_dataset):
            return_vals = []
            for attention_mask in tokenized_dataset["attention_mask"]:
                return_vals.append(
                    sum(attention_mask) >= min_tokenized_seq_len
                )
            return return_vals

        if not self.streaming:
            # do map and filter
            _tokenized_ds = input_dataset.map(
                tokenize_fn,
                batched=True, 
                remove_columns=dataset_col_remove,
                load_from_cache_file=load_from_cache_file, 
                num_proc=num_proc,
                desc="Running tokenizer on raw dataset (map), no streaming",
            ).filter(
                filter_fn,
                batched=True,
                load_from_cache_file=load_from_cache_file,
                num_proc=num_proc,
                desc="Running tokenizer on raw dataset (filter), no streaming",
            )
        else:
            _tokenized_ds = input_dataset.map(
                tokenize_fn, 
                batched=True, 
                remove_columns=dataset_col_remove,
            ).filter(
                filter_fn,
                batched=True
            )
        self.clm_tokenized_ds = _tokenized_ds

        # if streaming mode, both add_bos_token and add_eos_token should be False at begin.
        # print(f"tokenizer.add_bos_token: {self.tokenizer.add_bos_token}")
        # print(f"tokenizer.add_eos_token: {self.tokenizer.add_eos_token}")

        return self
    
    def get_tokenized_dataset(self):
        return self.clm_tokenized_ds

    def get_chunked_tokenized_dataset(
        self,
        add_bos_token:bool=False,
        add_eos_token:bool=True, 
    ):
        """Defaultly, tokenized seq are not chunked by return_overflowing_tokens
        when using stream mode. Therefore, we group the tokenized dataset and 
        then chunk it with padding.
        """
        if not self.streaming:
            logger.warn('>>>>>NO need for chunking, as not using streaming mode. Returning original tokenized data.')
            return self.get_tokenized_dataset()
        
        # update self.tokenizer 
        self.tokenizer.add_bos_token = add_bos_token
        self.tokenizer.add_eos_token = add_eos_token
        
        # print(f"updated tokenizer.add_bos_token to: {self.tokenizer.add_bos_token}")
        # print(f"updated tokenizer.add_eos_token to: {self.tokenizer.add_eos_token}")
        
        bos_token_id = [self.tokenizer.bos_token_id] if self.tokenizer.add_bos_token else []
        eos_token_id = [self.tokenizer.eos_token_id] if self.tokenizer.add_eos_token else []

        bos_attn_mask = [1] if self.tokenizer.add_bos_token else []
        eos_attn_mask = [1] if self.tokenizer.add_eos_token else []

        _chunk_length = self.max_length - len(bos_token_id + eos_token_id) 

        
        def group_and_chunk_fn(tokenized_dataset):
            
            # print(f"tokenized_dataset.keys() in the group_and_chunk_fn: {list(tokenized_dataset.keys())}")
            # >>>['input_ids', 'attention_mask']
            # - This indicates that the tokenized_dataset is Dataset or IterableDataset 
            # from key of 'train', or 'test' in the  DatasetDict or IterableDatasetDict (here: self.clm_tokenized_ds).
            # - The key of 'train', or 'test' has been specified when 
            # calling `self.clm_tokenized_ds.map()`.
            # - Once specified, we can deal with the Dataset or IterableDataset (here: tokenized_dataset) with features that 
            # determined by `model_input_names` specified in the `tokenizer` class (either slow and fast).

            # print(f"The number of samples in tokenized_dataset['input_ids']: {len(tokenized_dataset['input_ids'])}")
            # print(f"The number of samples in tokenized_dataset['attention_mask']: {len(tokenized_dataset['attention_mask'])}")
            # >>> 500 (debug mode)
            # - These output the number of samples in the Dataset or IterableDataset (here: tokenized_dataset).
            # - The above two outputs of the number of samples in `input_ids` and the number of `attention_mask`
            # should be the same, otherwise Error will be raised.
            # - In the `streaming` mode, the length of each sample may not be the same.

            # - This is just for debug that was used to return the samples in the 
            # tokenized_dataset["input_ids"] and tokenized_dataset["attention_mask"].
            # - The output `input_ids` or `attention_mask` will have the 2d list data structure. Each sample
            # will have different length. ie, [[sample_1 with len_1], [sample_2 with len_2],..., [sample_n with len_n]]
            # return {"input_ids":tokenized_dataset["input_ids"],
            #         "attention_mask":tokenized_dataset["attention_mask"]}

            # - Originally, each sample in tokenized_dataset[k] (where `k` would be `input_ids` or `attention_mask`)
            # will be a list with different length.
            # >>>
            # - For example: the first sample in tokenized_dataset["input_ids"][0][0:6] is 
            # [0, 6, 4319, 4319, 254, 7994], and tokenized_dataset["attention_mask"][0][0:6] is
            # [1, 1, 1, 1, 1, 1].
            concat_token_ids = {k: list(chain(*tokenized_dataset[k])) for k in tokenized_dataset.keys()}
            # - But, after calling above, the output (ie, `concat_token_ids`) becomes:
            # >>>
            # {'input_ids': 0, 'attention_mask': 1}
            # {'input_ids': 6, 'attention_mask': 1}
            # {'input_ids': 4319, 'attention_mask': 1}
            # {'input_ids': 4319, 'attention_mask': 1}
            # {'input_ids': 254, 'attention_mask': 1}
            # {'input_ids': 7994, 'attention_mask': 1}
            #
            # - However, when call the following, the outputs are:
            # >>>
            # input_ids: [0, 6, 4319, 4319, 254, 7994, 7994, 1226, 5148, 6301, 7994, 7994, ...],
            # attention_mask: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...]
            # and its length is sum of all sample lengths in `input_ids` (concat result): 1501487.
            #
            # for k, v in concat_token_ids.items():
            #     print(f"{k}:{v}")
            # 
            
            # total sequence length after concatenation.
            concat_token_ids_length = len(concat_token_ids[list(tokenized_dataset.keys())[0]])
            # print(f"concat_token_ids_length: {concat_token_ids_length}")
            # >>> e.g., 1501487 (all sample account)
            #

            # Overlap sliding window:
            # - self.stride = overlap_size, not the step size,
            # which is defined in the `__call__` of tokenizer (slow or fast).
            step_size = _chunk_length - self.stride
            num_chunks = (concat_token_ids_length - _chunk_length) // step_size + 1

            # chunked total length
            _total_length = step_size * num_chunks + self.stride
            _remain_length = concat_token_ids_length - _total_length

            _remain_length_with_stride = _remain_length + self.stride
            
            result_chunks = {}
            for k, t in concat_token_ids.items():
                chunked_ids = []
                for i in range(num_chunks):
                    _start_pos = i * step_size
                    _end_pos = _start_pos + _chunk_length 
                    if k == 'input_ids':
                        chunk_sequence = bos_token_id + t[_start_pos:_end_pos] + eos_token_id
                    elif k == 'attention_mask':
                        chunk_sequence = bos_attn_mask + t[_start_pos:_end_pos] + eos_attn_mask
                    else:
                        raise KeyError("Only support model_input_names = ['input_ids', 'attention_mask'].")
                    chunked_ids.append(chunk_sequence)

                if _remain_length_with_stride >= _chunk_length * self.min_len_frac:
                    if self.tokenizer.padding_side == 'right':
                        if k == 'input_ids':
                            _last_chunk = (bos_token_id + t[-_remain_length_with_stride:] + eos_token_id 
                                           + [self.tokenizer.pad_token_id] * (_chunk_length - _remain_length_with_stride))
                        elif k == 'attention_mask':
                            _last_chunk = (bos_attn_mask + t[-_remain_length_with_stride:] + eos_attn_mask 
                                           + [0] * (_chunk_length - _remain_length_with_stride))
                        else: 
                            raise KeyError("Only support model_input_names = ['input_ids', 'attention_mask'].")
                    else:
                        if k == 'input_ids':
                            _last_chunk = ([self.tokenizer.pad_token_id] * (_chunk_length - _remain_length_with_stride) 
                                           + bos_token_id + t[-_remain_length_with_stride:] + eos_token_id)
                        elif k == 'attention_mask':
                            _last_chunk = ([0] * (_chunk_length - _remain_length_with_stride) 
                                           + bos_attn_mask + t[-_remain_length_with_stride:] + eos_attn_mask)
                        else: 
                            raise KeyError("Only support model_input_names = ['input_ids', 'attention_mask'].")
                    chunked_ids.append(_last_chunk)

                result_chunks[k] = chunked_ids
            
            return result_chunks
        
        # print(f"self.clm_tokenized_ds.keys(): {list(self.clm_tokenized_ds.keys())}")
        # >>>['train', 'test']
        # self.clm_tokenized_ds is IterableDatasetDict as it was generated by `do_map` function with `streaming` mode.
        _tokenized_ds = self.clm_tokenized_ds.map(
            group_and_chunk_fn, 
            batched=True, 
        )
        # To solve the `Unknown` features in the IterableDataset issue:
        # https://github.com/huggingface/datasets/issues/3888#issuecomment-1330495533
        for k in _tokenized_ds.keys():
            _tokenized_ds[k] = _tokenized_ds[k]._resolve_features()

        return _tokenized_ds

        

        

        

