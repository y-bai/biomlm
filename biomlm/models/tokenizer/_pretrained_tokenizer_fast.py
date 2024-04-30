#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		_pretrained_tokenizer_fast.py
@Time    :   	2024/03/14 14:41:53
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

@Desc    :   	

legacy implementation!

"""

from typing import List, Optional
from transformers import PreTrainedTokenizerFast
from tokenizers import processors

from ._bpe_tokenizer import BioSeqBPETokenizer
from ._unigram_tokenizer import BioSeqUnigramTokenizer

def save_pretrained_tokenizer_fast(
    token_model_name: str = "BPE", # or "Unigram"
    vocab_filename: str = None,
    merges_filename: str = None,
    saved_dir:str = None,
    bos_token: str = "<BOS>",
    eos_token: str = "<BOS>",
    pad_token: str = "<BOS>",
    unk_token: str = "<UNK>",
    mask_token: str = "<MASK>",
    **kwargs,
):
    """
    
    bos_token, eos_token, and pad_token are setting following:
        https://huggingface.co/state-spaces/mamba-2.8b-hf/blob/main/config.json
    """
    if token_model_name == "BPE":
        init_tokenier = BioSeqBPETokenizer.from_file(
            vocab_filename=vocab_filename,
            merges_filename=merges_filename,
        )
    if token_model_name == "Unigram":
        init_tokenier = BioSeqUnigramTokenizer.from_file(
            vocab_filename=vocab_filename
        )
    
    _save_pretrained(
        init_tokenier._tokenizer,
        save_dir=saved_dir,
        bos_token=bos_token,
        eos_token=eos_token,
        pad_token=pad_token,
        unk_token=unk_token,
        mask_token=mask_token,
        filename_prefix=None,
        do_lower_case = False,     
        remove_space =True,
        add_prefix_space = False, 
        trim_offsets = True,
        **kwargs,
    )

def _save_pretrained(
    _tokenizer_object,
    save_dir: str = None,
    bos_token: str = "<BOS>",
    eos_token: str = "<BOS>",
    pad_token: str = "<BOS>",
    unk_token: str = "<UNK>",
    mask_token: str = "<MASK>",
    filename_prefix: Optional[str] = None,
    do_lower_case: Optional[bool] = False,     
    remove_space: Optional[bool] =True,
    add_prefix_space: Optional[bool] = False, 
    trim_offsets: Optional[bool] = True,
    **kwargs,
):
    """
    Save the tokenizer using the `PreTrainedTokenizerFast` HF API,
    `processors.TemplateProcessing` is added to the tokenizer.

    Parameters can be referred to:
    https://huggingface.co/docs/transformers/v4.38.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast

    Parameters
    ----------
    save_dir : str, 
        _description_, by default None
    filename_prefix : str
        _description_, by default None
    do_lower_case : Optional[bool], optional
        _description_, by default False
    remove_space : Optional[bool], optional
        _description_, by default True
    add_prefix_space : Optional[bool], optional
        _description_, by default False
    trim_offsets : Optional[bool], optional
        _description_, by default True

    Raises
    ------
    ValueError
        _description_
    """
    _tokenizer_object.post_processor = processors.TemplateProcessing(
        single=f"{bos_token} $A {eos_token}",
        pair=f"{bos_token} $A {eos_token}:0 $B:1 {eos_token}:1",
        special_tokens=[
            (bos_token, _tokenizer_object.token_to_id(bos_token)),
            (eos_token, _tokenizer_object.token_to_id(eos_token)),
        ],
    )
    
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=_tokenizer_object,
        bos_token=bos_token,
        eos_token=eos_token,
        pad_token=pad_token,
        unk_token=unk_token,
        mask_token=mask_token,
        do_lower_case=do_lower_case,
        remove_space=remove_space,
        add_prefix_space=add_prefix_space,
        trim_offsets=trim_offsets,
        **kwargs,
    )
    if save_dir is None:
        raise ValueError("save_dir: None is found.")
    
    hf_tokenizer.save_pretrained(save_dir, filename_prefix=filename_prefix)

