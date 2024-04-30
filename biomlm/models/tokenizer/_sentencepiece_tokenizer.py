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
import os
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union

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
        bos_token="<BOS>",
        eos_token="<EOS>",
        unk_token="<UNK>",
        add_bos_token: bool=False,
        add_eos_token: bool=True,
        do_lower_case: bool=False,
        add_prefix_space: bool = False,
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

        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token

        self.do_lower_case = do_lower_case
        self.add_prefix_space = add_prefix_space


        self.vocab_file = vocab_file
        # load the pretrained vocab model
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file) 

        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token


        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            do_lower_case=do_lower_case,
            add_prefix_space=add_prefix_space,
            sp_model_kwargs=self.sp_model_kwargs,
            **kwargs,
        )

    @property
    def vocab_size(self):
        return self.sp_model.vocab_size()
    
    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab
    
    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d

        # for backward compatibility
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(self.vocab_file)
    
    def preprocess_seq(self, inputs):
        if self.do_lower_case:
            inputs = inputs.lower()
        return inputs
    
    def _tokenize(self, raw_seq: str):
        """
        Converts a string into a sequence of tokens (string), using the tokenizer. Split in words for word-based
        vocabulary or sub-words for sub-word-based vocabularies (BPE/SentencePieces/WordPieces).

        Do NOT take care of added tokens.
        """
        raw_seq = self.preprocess_seq(raw_seq)
        pieces = self.sp_model.encode(raw_seq, out_type=str)

        return pieces
    
    def _convert_token_to_id(self, token):
        return self.sp_model.PieceToId(token)
    
    def _convert_id_to_token(self, index: int) -> str:
        return self.sp_model.IdToPiece(index)

    # def _decode(self, token_ids: List[int], skip_special_tokens: bool = False, clean_up_tokenization_spaces: bool = None, spaces_between_special_tokens: bool = True, **kwargs) -> str:
    #     return super()._decode(token_ids, skip_special_tokens, clean_up_tokenization_spaces, spaces_between_special_tokens, **kwargs)

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
    
    def get_special_tokens_mask(
        self, 
        token_ids_0: List[int], 
        token_ids_1: Optional[List[int]] = None, 
        already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )
        
        bos_token_id = [1] if self.add_bos_token else []
        eos_token_id = [1] if self.add_eos_token else []

        if token_ids_1 is None:
            return bos_token_id + ([0] * len(token_ids_0)) + eos_token_id
        
        return (
            bos_token_id
            + ([0] * len(token_ids_0))
            + eos_token_id
            + ([0] * len(token_ids_1))
            + eos_token_id
        )
    
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


    def prepare_for_tokenization(self, seq, is_split_into_words=False, **kwargs):
        add_prefix_space = kwargs.pop("add_prefix_space", self.add_prefix_space)
        if (is_split_into_words or add_prefix_space) and (len(seq) > 0 and not seq[0].isspace()):
            seq = " " + seq
        return (seq, kwargs)
    
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