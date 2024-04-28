#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :      _bpe_tokenizer.py
@Time    :      2024/02/22 16:44:42
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

@Desc    :      tokenize the bio sequence using Byte-level BPE 

This is adapted from:
https://github.com/huggingface/tokenizers/blob/main/bindings/python/py_src/tokenizers/implementations/byte_level_bpe.py

"""

from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from tokenizers import (
    Tokenizer, 
    trainers, 
    pre_tokenizers, 
    decoders,
    models,
)
from tokenizers.normalizers import (
    Lowercase,
    Sequence, 
    unicode_normalizer_from_str
)
from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer
from transformers.utils import logging

from ._base_tokenizer import BaseTokenizer
from ..utils import verify_exist 

logger = logging.get_logger(__name__)

class BioSeqBaseBPETokenizer(BaseTokenizer):
    """
    bio sequence token generation based on Byte-level BPE.
    """
    def __init__(
            self,
            vocab: Optional[Union[str, Dict[str, int]]] = None,
            merge: Optional[Union[str, Dict[Tuple[int, int], Tuple[int, int]]]] = None,
            unk_token: Optional[str] = '<UNK>',
            add_prefix_space: bool = False,
            use_regex: bool = False,
            lowercase: bool = False,
            dropout: Optional[float] = None,
            unicode_normalizer: Optional[str] = None,
            continuing_subword_prefix: Optional[str] = None,
            end_of_word_suffix: Optional[str] = None,
    ):

        if vocab is not None and merge is not None:
            tokenizer = Tokenizer(
                models.BPE(
                    vocab,
                    merge,
                    dropout=dropout,
                    continuing_subword_prefix=continuing_subword_prefix or "",
                    end_of_word_suffix=end_of_word_suffix or "",
                    unk_token=unk_token,
                )
            ) 
        else:
            # Initialze an empty tokenizer
            tokenizer = Tokenizer(models.BPE(unk_token=unk_token))
        
        # Check for Unicode normalization first (before everything else)
        normalizers = []

        if unicode_normalizer:
            normalizers += [unicode_normalizer_from_str(unicode_normalizer)]

        if lowercase:
            normalizers += [Lowercase()]

        # Create the normalizer structure
        if len(normalizers) > 0:
            if len(normalizers) > 1:
                tokenizer.normalizer = Sequence(normalizers)
            else:
                tokenizer.normalizer = normalizers[0]

        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=add_prefix_space,
            use_regex = use_regex
        )
        tokenizer.decoder = decoders.ByteLevel()
        # tokenizer.post_processor = processors.ByteLevel(trim_offsets=trim_offsets)

        parameters = {
            "model": "ByteLevelBPE",
            "add_prefix_space": add_prefix_space,
            "use_regex": use_regex,
            "lowercase": lowercase,
            "dropout": dropout,
            "unicode_normalizer": unicode_normalizer,
            "continuing_subword_prefix": continuing_subword_prefix,
            "end_of_word_suffix": end_of_word_suffix,
        }

        super().__init__(tokenizer, parameters)
    
    @staticmethod
    def from_file(vocab_filename: str, merges_filename: str, **kwargs):
        verify_exist(vocab_filename) 
        verify_exist(merges_filename)
        vocab, merges = models.BPE.read_file(vocab_filename, merges_filename)
        return BioSeqBaseBPETokenizer(vocab, merges, **kwargs)
    
    def train(
            self,
            files: Union[str, List[str]],
            vocab_size: int = 52009,
            min_frequency: int = 2,
            show_progress: bool = True,
            initial_alphabet: List[str] = ['A', 'C', 'G', 'T', 'N'],
            special_tokens: List[Union[str, AddedToken]] = ["<BOS>", "<UNK>", "<EOS>"],
            max_token_length: Optional[int] = None,
    ):
        """Train the model using the given files"""

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            show_progress=show_progress,
            special_tokens=special_tokens,
            initial_alphabet=initial_alphabet,
            max_token_length=max_token_length,
        )
        if isinstance(files, str):
            files = [files]
        self._tokenizer.train(files, trainer=trainer)
    
    def train_from_iterator(
        self,
        iterator: Union[Iterator[str], Iterator[Iterator[str]]],
        vocab_size: int = 52009,
        min_frequency: int = 2,
        show_progress: bool = True,
        initial_alphabet: List[str] = ['A', 'C', 'G', 'T', 'N'],
        special_tokens: List[Union[str, AddedToken]] = ["<BOS>", "<UNK>", "<EOS>"],
        max_token_length: Optional[int] = None,
        length: Optional[int] = None,
    ):
        """Train the model using the given iterator"""

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            show_progress=show_progress,
            special_tokens=special_tokens,
            initial_alphabet=initial_alphabet,
            max_token_length=max_token_length,
        )
        self._tokenizer.train_from_iterator(
            iterator,
            trainer=trainer,
            length=length,
        )


"""
>>>tokenizer = BioSeqBaseBPETokenizer()
>>>tokenizer.train_from_iterator(
    data_iterator,
    vocab_size=vocab_size,
    length=data_len,
    special_tokens= ["<BOS>", "<UNK>", "<EOS>", "<MASK>"],
>>>)
>>>tokenizer.save_model(saved_dir)

Above code will generate two files: `vocab.json` and `merges.txt` as pretrained vocab.
see `token_gen.py`.
"""
VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

class BioSeqBPETokenizer(PreTrainedTokenizer):
    """
    Pretrained tokenizer to tokenizate sequences in `datasets` for causal models using 

    """
    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    padding_side: str = "right"
    truncation_side: str = "right"

    def __init__(
        self,
        vocab_file: str,
        merges_file: str,
        bos_token: str = "<BOS>",
        eos_token: str = "<EOS>",   
        pad_token: str = "<PAD>",   
        unk_token: str = "<UNK>",
        mask_token: str = "<MASK>",
        add_bos_token: bool=False,
        add_eos_token: bool=True,
        ## unset, since Error: unsupported operand type(s) for ** or pow(): 'int' and 'dict'
        ## when initialize Fast mode, this becuase two model_max_length will be used by the parent class.
        # model_max_length: int=512,
        base_model_kwarg: Optional[Dict[str, Any]] = None,
        add_prefix_space: bool = False, 
        do_lower_case: bool=False,
        **kwargs,
    ):  
        self.vocab_file = vocab_file
        self.merges_file = merges_file
        self.base_model_kwarg = {
            "add_prefix_space": add_prefix_space,
            "use_regex": False,
            "lowercase": do_lower_case,
            "dropout": None,
            "unicode_normalizer": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
        } if base_model_kwarg is None else base_model_kwarg
        self.base_tokenier = BioSeqBaseBPETokenizer.from_file(
                vocab_filename=vocab_file,
                merges_filename=merges_file,
                unk_token=unk_token,
                **self.base_model_kwarg,
        )

        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token

        self.add_prefix_space = add_prefix_space
        self.do_lower_case = do_lower_case

        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
        # In bioseq, mask token does not behave like a normal word, i.e. not include the space before it
        # mask_token = (
        #     AddedToken(mask_token, lstrip=True, rstrip=False, normalized=False)
        #     if isinstance(mask_token, str)
        #     else mask_token
        # )
        mask_token = AddedToken(mask_token, lstrip=False, rstrip=False) if isinstance(mask_token, str) else mask_token

        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            mask_token=mask_token,
            do_lower_case=do_lower_case,
            add_prefix_space=add_prefix_space,
            # model_max_length=model_max_length
            **kwargs,
        )


    @property
    def vocab_size(self):
        return self.base_tokenier.get_vocab_size()

    def get_vocab(self) -> Dict[str, int]:
        return self.base_tokenier.get_vocab()
    
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
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
    
    def _tokenize(
            self, 
            raw_seq: str,
            pair: Optional[str] = None,
            is_pretokenized: bool = False,
            add_special_tokens: bool = False,
    ):
        """Tokenize a sequnce"""

        return self.base_tokenier.encode(raw_seq, pair, is_pretokenized, add_special_tokens).tokens
    
    def _convert_token_to_id(self, token):
        return self.base_tokenier.token_to_id(token)
    
    def _convert_id_to_token(self, index: int) -> str:
        return self.base_tokenier.id_to_token(index)
    
    def _add_tokens(self, new_tokens: Union[List[str], List[AddedToken]], special_tokens: bool = False) -> int:
        if not special_tokens:
            num_added_tokens = self.base_tokenier.add_tokens(new_tokens)
        else:
            num_added_tokens = self.base_tokenier.add_special_tokens(new_tokens)

        added_tokens: Dict[int, AddedToken] = self.base_tokenier.get_added_tokens_decoder()
        for token_id, token in added_tokens.items():
            if special_tokens and isinstance(token, AddedToken):
                # doing token.special=True changes the normalization! will fix in rust
                # this is important and the only reason why the AddedTokens in each class are normalized by default
                token.__setstate__({"special": True, "normalized": token.normalized})

            if not token.special and token.normalized and getattr(self, "do_lower_case", False):
                # Normalize if requested
                token.content = token.content.lower()
            
            if token.special and str(token) not in self.all_special_tokens:
                self._additional_special_tokens.append(token)
            
            # the setter automatically updates the reverse map
            self._added_tokens_decoder[token_id] = token
            self._added_tokens_encoder[token.content] = token_id
            if self.verbose:
                logger.info(f"Adding {token} to the vocabulary")
        
        self._update_trie()
        return num_added_tokens
    
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

    def prepare_for_tokenization(self, seq, is_split_into_words=False, **kwargs):
        add_prefix_space = kwargs.pop("add_prefix_space", self.add_prefix_space)
        if (is_split_into_words or add_prefix_space) and (len(seq) > 0 and not seq[0].isspace()):
            seq = " " + seq
        return (seq, kwargs)
            
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        return tuple(self.base_tokenier.save_model(save_directory, filename_prefix))



