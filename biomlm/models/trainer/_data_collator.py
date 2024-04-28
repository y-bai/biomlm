#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		_data_loader.py
@Time    :   	2024/03/08 23:22:06
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

from dataclasses import dataclass
from typing import Any, Dict, List, Union

# import torch
# from torch.utils.data import DataLoader
# from datasets import (
#     DatasetDict,
#     Dataset,
#     IterableDatasetDict,
#     IterableDataset,
# )
# from transformers import PreTrainedTokenizerBase
from transformers import DataCollatorForLanguageModeling


@dataclass
class BioSeqDataCollatorCausalLM(DataCollatorForLanguageModeling):
    """ 
    refer:
    https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py

    """
    replaced_padding_id: int = -100
    return_tensors: str = 'pt'

    def __call__(
            self, 
            examples: List[Union[List[int], Any, Dict[str, Any]]],
        ) -> Dict[str, Any]:
        """rewrite the `DataCollatorForLanguageModeling.__call__` function
        to enable redefine the padding_token_id instead of using the default value of -100

        """

        batch = super().__call__(examples)

        # NOTE
        # `DataCollatorForLanguageModeling`` set the pad tokens (0) to -100 to be
        # ignored by the CrossEntropy loss, thus we don't need to recover it.
        # So in our training, we do NOT use `BioSeqDataCollator` to recover pad_id,
        # instead, we still use `DataCollatorForLanguageModeling`. 
        # See `run_bioseqmamba_causal.py`.
        input_ids = batch["input_ids"].clone()
        input_ids[input_ids == -100] = self.replaced_padding_id
        batch["input_ids"] = input_ids
        # 
        # Tensor should use `.clone()`
        # See: https://discuss.pytorch.org/t/copy-deepcopy-vs-clone/55022/7
        batch["labels"] = input_ids.clone() 
        # print(f"batch['input_ids']: {batch['input_ids']}")
        # print(f"batch['labels']:{batch['labels']}")
        
        return batch

# class BioSeqDataLoader(DataLoader):
#     """ Dataloader, the data_collator is from `BioSeqDataCollator`
    
#     """
#     def __init__(
#             self, 
#             dataset: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset], 
#             tokenizer: PreTrainedTokenizerBase, 
#             mlm: Optional[bool] = False, 
#             replaced_padding_id: Optional[int] = -100,
#             **kwargs
#     ):
#         data_collator  = BioSeqDataCollator(
#             tokenizer, 
#             mlm=mlm, 
#             replaced_padding_id=replaced_padding_id)
        
#         super().__init__(dataset, collate_fn=data_collator, **kwargs)

# def data_collator(batch):

#     input_ids = [item["input_ids"] for item in batch]
#     attention_masks = [item["attention_mask"] for item in batch]
#     labels = [item["input_ids"] for item in batch]

#     # convert lists to pt tensors
#     input_ids = torch.tensor(input_ids)
#     attention_masks = torch.tensor(attention_masks)
#     labels = torch.tensor(labels)

#     return {
#         "input_ids": input_ids,
#         "attention_mask": attention_masks,
#         "labels": labels,
#     }