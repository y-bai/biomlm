#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		_sentencepiece_tokenizer_fast.py
@Time    :   	2024/04/29 11:55:13
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

https://github.com/huggingface/transformers/blob/main/src/transformers/models/xlnet/tokenization_xlnet_fast.py

"""
import os
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple

from transformers.utils import logging
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, processors, normalizers, Regex
from transformers.convert_slow_tokenizer import SpmConverter

from ._sentencepiece_tokenizer import BioSeqSPMTokenizer

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "spm_vocab.model", "tokenizer_file": "tokenizer.json"}

def check_number_comma(piece: str) -> bool:
    return len(piece) < 2 or piece[-1] != "," or not piece[-2].isdigit()

class BioSeqSPMConverter(SpmConverter):
    def vocab(self, proto):
        return [
            (piece.piece, piece.score) if check_number_comma(piece.piece) else (piece.piece, piece.score - 100)
            for piece in proto.pieces
        ]

    def normalizer(self, proto):
        list_normalizers = [
            normalizers.Replace("``", '"'),
            normalizers.Replace("''", '"'),
        ]
        # if not self.original_tokenizer.keep_accents:
        #     list_normalizers.append(normalizers.NFKD())
        #     list_normalizers.append(normalizers.StripAccents())
        if self.original_tokenizer.do_lower_case:
            list_normalizers.append(normalizers.Lowercase())

        precompiled_charsmap = proto.normalizer_spec.precompiled_charsmap

        if precompiled_charsmap:
            list_normalizers.append(normalizers.Precompiled(precompiled_charsmap))

        list_normalizers.append(normalizers.Replace(Regex(" {2,}"), " "))
        return normalizers.Sequence(list_normalizers)

    def post_processor(self):

        bos_token = str(self.original_tokenizer.bos_token)
        eos_token = str(self.original_tokenizer.eos_token)

        # print(f"bos_token: {bos_token}, eos_token: {eos_token}")

        add_bos = self.original_tokenizer.add_bos_token
        add_eos = self.original_tokenizer.add_eos_token

        single = f"{(bos_token+':0 ') if add_bos else ''}$A:0{(' ' + eos_token +':0') if add_eos else ''}"
        pair = f"{single} $B:1{(' ' + eos_token +':1') if add_eos else ''}"

        special_tokens = []
        if add_bos:
            special_tokens.append((bos_token, self.original_tokenizer.convert_tokens_to_ids(bos_token)))
        if add_eos:
            special_tokens.append((eos_token, self.original_tokenizer.convert_tokens_to_ids(eos_token)))

        return processors.TemplateProcessing(
            single=single,
            pair=pair,
            special_tokens=special_tokens,
        )

class BioSeqSPMTokenizerFast(PreTrainedTokenizerFast):
    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: str,
        tokenizer_file: Optional[str] = None,
        bos_token: str = "<BOS>",
        eos_token: str = "<EOS>",     
        unk_token: str = "<UNK>",
        model_max_length: int = 512,
        padding_side: str="right",
        truncation_side: str="right",
        add_bos_token: bool=False,
        add_eos_token: bool=True,
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        add_prefix_space: bool = False, 
        do_lower_case: bool=False,
        **kwargs,
    ):

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

        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token

        self.add_prefix_space = add_prefix_space
        self.do_lower_case = do_lower_case
        
        self.vocab_file = vocab_file
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        if tokenizer_file is None:

            slow_tokenizer = BioSeqSPMTokenizer(
                vocab_file=vocab_file,
                bos_token=bos_token,
                eos_token=eos_token,
                unk_token=unk_token,
                add_bos_token=add_bos_token,
                add_eos_token=add_eos_token,
                do_lower_case=do_lower_case,
                add_prefix_space=add_prefix_space,
                sp_model_kwargs=self.sp_model_kwargs,
                **kwargs
            )
            tokenizer_object = BioSeqSPMConverter(slow_tokenizer).converted()
        else:
            tokenizer_object = None
        
        super().__init__(
            tokenizer_object=tokenizer_object,
            tokenizer_file=tokenizer_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            do_lower_case=do_lower_case,
            add_prefix_space=add_prefix_space,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            padding_side=padding_side,
            truncation_side=truncation_side,
            model_max_length=model_max_length,
            **kwargs,
        )

    @property
    def can_save_slow_tokenizer(self) -> bool:
        return os.path.isfile(self.vocab_file) if self.vocab_file else False
    
    def build_inputs_with_special_tokens(
        self, 
        token_ids_0: List[int], 
        token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens when `add_special_tokens`=True in `__call__` function, which will call `tokenize` function 
        with the same of `add_special_tokens` parameter. A sequence has the following format:

        - single sequence: `<BOS> X <EOS>`
        - pair of sequences: `<BOS> A <EOS> B <EOS>`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """

        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = bos_token_id + token_ids_0 + eos_token_id

        if token_ids_1 is not None:
            output = output + token_ids_1 + eos_token_id

        return output
    

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task. An 
        sequence pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1
        | first sequence      | second sequence |
        ```

        if token_ids_1 is None, only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of ids.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """

        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []
        
        output = len(bos_token_id + token_ids_0 + eos_token_id) * [0]

        if token_ids_1 is not None:
            output += len(token_ids_1 + eos_token_id) * [1]
        
        return output

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )

        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        return (out_vocab_file,)