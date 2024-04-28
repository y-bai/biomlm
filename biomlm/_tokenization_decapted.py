#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		_tokenization.py
@Time    :   	2024/03/07 11:33:04
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

@Desc    :   	Input data tokenization for model 

"""

import os
from itertools import chain
from typing import Any, List, Optional, Union
from datasets import (
    DatasetDict,
    Dataset,
    IterableDatasetDict,
    IterableDataset,
)

from transformers import PreTrainedTokenizerFast
from .models.utils import verify_exist


class BioSeqTokenizer:
    """
    Tokenizate sequences in `datasets` for causal models using the 
    pretrained tokenizer. 

    """

    def __init__(
        self,
        pretrained_model_name_or_path: Optional[str] = None,
        local_files_only: Optional[bool] = True,
        model_max_length: Optional[int] = 512,
        padding_side: Optional[str] = "left",
        **kwargs,
    ):
        """
        load and instantiate the `PreTrainedTokenizerFast`. 

        kwargs will be passed to the `PreTrainedTokenizerFast.from_pretrained` method, which will be 
        further passed to the  `PreTrainedTokenizerFast.__init__` method.

        Parameters
        ----------
        pretrained_model_name_or_path: Optional[str], optional
            path of the pretrained tokenizer, by default None

        local_files_only: Optional[bool], optional
            whether load the tokenizer from local, by default True

        model_max_length: Optional[int], optional
            the maximum number of tokens in the sequence, by default 512.
        
        padding_side: Optional[str], optional
            The side on which the model should have padding applied, by default "right". 

            This becuase, although we use causal model, we tokenize the input sequences along with the 
            arguments of `return_overflowing_tokens=True` and `stride=6` (default). Therefore, we set
            `padding_side="right"`
        
        
        """
        verify_exist(pretrained_model_name_or_path)
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
            pretrained_model_name_or_path,
            local_files_only=local_files_only,
            padding_side=padding_side,
            model_max_length=model_max_length,
            **kwargs,
        )


    def __call__(
        self,
        dataset: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset],
        max_length: Optional[int] = 512,
        truncation: Optional[Union[str, bool]] = True,
        padding: Optional[Union[str, bool]] = "max_length",
        stride: Optional[int] = 10,
        return_overflowing_tokens: Optional[bool] = True,
        min_ctx_fraction: Optional[float] = 0.8,
        num_proc: int = 1,
        remove_columns: Optional[Union[str, List[str], None]] = None,
        load_from_cache_file: bool = False,
        **kwargs,
    ):
        """Tokenize the sequences in the input dataset and convert them 
        into token IDs. The token IDs for the input seuqence include
        <BOS> and <EOS>.

        kwargs parameters will be passed to `PreTrainedTokenizerFast.__call__` function.

        Parameter details see https://huggingface.co/docs/transformers/v4.38.2/en/main_classes/tokenizer 

        Parameters
        ----------
        dataset : Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]
            dataset generated by `load_dataset()`

        max_length : Optional[int], optional
            Controls the maximum length (in the number of tokens) to use by one of the truncation/padding
            parameters. by default 512, including the <BOS> and <EOS> tokens. This parameter will overwirte the 
            parameter `model_max_length` in `PreTrainedTokenizerFast.__init__` method.

            Details see `PreTrainedTokenizerFast.__call__` function.

            alias: n_ctx: defined in causal models
                   model_max_length: defined in the `PreTrainedTokenizerFast`

        truncation : Optional[Union[str, bool]], optional
            truncate the long sequence on the right, by default True

            Details see `PreTrainedTokenizerFast.__call__` function.

        padding : Optional[Union[str, bool]], optional
             Activates and controls padding, by default "max_length"
             Details see `PreTrainedTokenizerFast.__call__` function.

        stride: Optional[int], optional
            The value of this argument defines the number of overlapping tokens, by default 6.

            This argument should be along with max_length when `return_overflowing_tokens=True`.
            Example see https://huggingface.co/learn/nlp-course/en/chapter6/3b.

        return_overflowing_tokens : Optional[bool], optional
            whether tokenize the whole input and split it into several chunks, by default True

            For the reason, see 
            https://huggingface.co/learn/nlp-course/en/chapter7/6
            https://huggingface.co/learn/nlp-course/en/chapter6/3b

        min_ctx_fraction: Optional[float], optional
            minimum fraction of length of context in the chunked sequences. 
            This is used for check if the chunked sequences have enough tokens to input LM models. by default 0.8
        
        remove_columns: Optional[Union[str, List[str], None]], optional
            columns that will be removed from the returned tokenized dataset, by default None.
            This argument will be passed to the `dataset.map` function.
             
        """

        def tokenize_fn(untokenized_dataset):

            token_encodes = self.tokenizer(
                untokenized_dataset["sequence"],
                max_length=max_length,
                padding=padding,
                truncation=truncation,
                return_overflowing_tokens=return_overflowing_tokens,
                stride=stride,
                **kwargs,
            )

            input_batch = []
            attention_mask_batch = []
            # token_type_batch = []
            # overflow_to_sample_mapping_batch = []
            for input_ids, attention_mask in zip(
                token_encodes["input_ids"],
                # token_encodes["token_type_ids"],
                token_encodes["attention_mask"]
                # token_encodes["overflow_to_sample_mapping"]
            ):
                # filter out the short chunks
                if sum(attention_mask) >= int(max_length * min_ctx_fraction):
                    
                    input_batch.append(input_ids)
                    # token_type_batch.append(token_type_ids)
                    # Generally, the attention mask is made so that it accepts 0s and 1s. 
                    # Putting a 1 indicates that this token should be attended to, 
                    # while putting a 0 indicates a value that the token should not be attended to.
                    attention_mask_batch.append(attention_mask)
                    # overflow_to_sample_mapping_batch.append(overflow_to_sample_mapping)

            return {
                "input_ids": input_batch,
                # "token_type_ids": token_type_batch,
                "attention_mask": attention_mask_batch,
                # "overflow_to_sample_mapping": overflow_to_sample_mapping_batch
            }
        
        # when `return_overflowing_tokens = True `
        # Tokenize the whole input and split it into several chunks, and return 3d Python list.
        # The returned fields include:
        #      ['input_ids', 'token_type_ids', 'attention_mask', 'overflow_to_sample_mapping'] 
        # Setting `batched=True`can convert the 3d list into 2d list
        tokenized_datasets = dataset.map(
            tokenize_fn, 
            batched=True, 
            remove_columns=remove_columns,
            load_from_cache_file=load_from_cache_file, 
            num_proc=num_proc # os.cpu_count()
        )

        return tokenized_datasets
    
    def save_pretrained(self, output_dir, filename_prefix=None):
        self.tokenizer.save_pretrained(output_dir, filename_prefix=filename_prefix)


    
    
