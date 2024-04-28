#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		_sentencepiece_tokenizer.py
@Time    :   	2024/04/17 13:13:09
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
reference: 

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/xlnet/tokenization_xlnet.py

"""

from typing import Any, Dict, Optional
import sentencepiece as spm

from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer

from transformers.utils import logging

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "spm_vocab.model"}

class BioSeqSPMTokenizer(PreTrainedTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES

    model_input_names = ["input_ids", "attention_mask"]

    padding_side: str = "right"
    truncation_side: str = "right"

    def __init__(
        self,
        vocab_file,
        do_lower_case=False,
        bos_token="<BOS>",
        eos_token="<EOS>",
        unk_token="<UNK>",
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        # default spm_kwargs values:
        # see: https://github.com/google/sentencepiece/blob/master/python/src/sentencepiece/__init__.py#L471
        # spm_kwargs = {
        #     "add_bos": False,
        #     "add_eos": False,
        #     "reverse": False,
        #     "emit_unk_piece": False,
        #     "enable_sampling": False, # default is False, if Ture, then enconded str is different every time
        #     "nbest_size": -1,
        #     "alpha":0.16,
        # }

        self.do_lower_case = do_lower_case
        self.vocab_file = vocab_file
        
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)

        super().__init__(
            do_lower_case=do_lower_case,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sp_model_kwargs=self.sp_model_kwargs,
            **kwargs,
        )
        

        



    

# def train_jp(vocab_size):
#   model_prefix = "cc100_jp" + "_vocab_" + str(vocab_size)
#   spm.SentencePieceTrainer.train(input=tempfile_path
#       , model_prefix=model_prefix
#       , vocab_size=vocab_size
#       , character_coverage = 0.9995
#       , num_threads=60
#       , train_extremely_large_corpus=True
#   )
# train_jp(8000)

# https://github.com/google/sentencepiece/blob/master/python/README.md
# spm.SentencePieceTrainer.train(
#       sentence_iterator=response, model_writer=model, vocab_size=1000)