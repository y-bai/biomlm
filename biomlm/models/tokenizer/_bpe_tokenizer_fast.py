#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		_bpe_tokenizer_fast.py
@Time    :   	2024/04/07 02:51:16
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
import os
from typing import Any, Dict, Optional, Tuple
from transformers.utils import logging
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, processors

from transformers.convert_slow_tokenizer import Converter

from ._bpe_tokenizer import BioSeqBaseBPETokenizer, BioSeqBPETokenizer

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt", "tokenizer_file": "tokenizer.json"}

class BioSeqBPEConverter(Converter):
    def converted(self) -> Tokenizer:
        # vocab_file = self.original_tokenizer.vocab_file
        # merges_file = self.original_tokenizer.merges_file

        # base_tokenizer = BioSeqBaseBPETokenizer.from_file(
        #         vocab_filename=vocab_file,
        #         merges_filename=merges_file,
        #         unk_token=str(self.original_tokenizer.unk_token),
        #         **self.original_tokenizer.base_model_kwarg,
        # )

        base_tokenizer = self.original_tokenizer.base_tokenier

        bos_token = str(self.original_tokenizer.bos_token)
        eos_token = str(self.original_tokenizer.eos_token)

        # print(f"bos_token: {bos_token}, eos_token: {eos_token}")

        add_bos = self.original_tokenizer.add_bos_token
        add_eos = self.original_tokenizer.add_eos_token

        # print(f"add_bos: {add_bos}, eos_token: {add_eos}")
        # print(f"bos_token_id: {base_tokenizer.token_to_id(bos_token)}")
        # print(f"eos_token_id: {base_tokenizer.token_to_id(eos_token)}")

        single = f"{(bos_token+':0 ') if add_bos else ''}$A:0{(' ' + eos_token +':0') if add_eos else ''}"
        pair = f"{single} $B:1{(' ' + eos_token +':1') if add_eos else ''}"

        special_tokens = []
        if add_bos:
            special_tokens.append((bos_token, base_tokenizer.token_to_id(bos_token)))
        if add_eos:
            special_tokens.append((eos_token, base_tokenizer.token_to_id(eos_token)))

        # print(f"single: \n{single}")
        # print(f"pair: \n{pair}")
        # print(f"special_tokens: \n{special_tokens}")
        
        if not add_bos and not add_eos:
            # XXX trim_offsets=False actually means this post_processor doesn't
            # really do anything.
            base_tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
        else:
            base_tokenizer.post_processor = processors.TemplateProcessing(
                single=single,
                pair=pair,
                special_tokens=special_tokens
            )
        # https://github.com/huggingface/tokenizers/issues/1105
        return base_tokenizer._tokenizer

class BioSeqBPETokenizerFast(PreTrainedTokenizerFast):

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: str,
        merges_file: str,
        tokenizer_file: Optional[str] = None,
        bos_token: str = "<BOS>",
        eos_token: str = "<EOS>",   
        # pad_token: str = "<PAD>",   
        unk_token: str = "<UNK>",
        # mask_token: str = "<MASK>",
        model_max_length: int = 512,
        padding_side: str="right",
        truncation_side: str="right",
        add_bos_token: bool=False,
        add_eos_token: bool=True,
        base_model_kwarg: Optional[Dict[str, Any]] = None,
        add_prefix_space: bool = False, 
        do_lower_case: bool=False,
        **kwargs,
    ):
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token

        self.add_prefix_space = add_prefix_space
        self.do_lower_case = do_lower_case
        
        self.vocab_file = vocab_file
        self.merges_file = merges_file

        # convert slow tokenizer
        if tokenizer_file is None:
            self.base_model_kwarg = {
                "add_prefix_space": add_prefix_space,
                "use_regex": False,
                "lowercase": do_lower_case,
                "dropout": None,
                "unicode_normalizer": None,
                "continuing_subword_prefix": None,
                "end_of_word_suffix": None,
            } if base_model_kwarg is None else base_model_kwarg
            
            #
            # https://github.com/huggingface/tokenizers/issues/1105
            # ~/.local/lib/python3.9/site-packages/transformers/tokenization_utils_base.py 
            # 3006         )
            # ...
            # --> 451                 self._tokenizer.enable_truncation(**target)
            # 452 
            # 453         if padding_strategy == PaddingStrategy.DO_NOT_PAD:
            # 
            # TypeError: enable_truncation() got an unexpected keyword argument 'direction' 
            #
            # tokenizer_object = BioSeqBaseBPETokenizer.from_file(
            #     vocab_filename=vocab_file,
            #     merges_filename=merges_file,
            #     unk_token=unk_token,
            #     **self.base_model_kwarg,
            # )

            slow_tokenizer = BioSeqBPETokenizer(
                vocab_file=vocab_file,
                merges_file=merges_file,
                bos_token=bos_token,
                eos_token=eos_token,   
                # pad_token=pad_token,   
                unk_token=unk_token,
                # mask_token=mask_token,
                add_bos_token=add_bos_token,
                add_eos_token=add_eos_token,
                base_model_kwarg=self.base_model_kwarg,
                add_prefix_space=add_prefix_space, 
                do_lower_case=add_prefix_space,
                padding_side=padding_side,
                truncation_side=truncation_side,
                **kwargs)
            tokenizer_object = BioSeqBPEConverter(slow_tokenizer).converted()
        else:
            tokenizer_object = None
        
        super().__init__(
            tokenizer_object=tokenizer_object,
            tokenizer_file=tokenizer_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            # pad_token=pad_token,
            # mask_token=mask_token,
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

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )

        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return

        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)



    