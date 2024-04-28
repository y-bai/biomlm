#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		_seq_generate.py
@Time    :   	2024/03/20 15:43:53
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

@Desc    :   	Bioseq generation using the causal large model

"""

from typing import Optional
import torch
import torch.nn.functional as F

def seq_generate(
        model,
        tokenizer,
        prompt: str,
        n_tokens_to_gen: int = 50,
        sample: bool = True,
        top_k: Optional[int] = 40
):
    model.eval()
    input_ids = tokenizer(prompt).input_ids

    for token_n in range(n_tokens_to_gen):
        with torch.no_grad():
            idx_to_input = input_ids
            next_token_logits = model(idx_to_input)["logits"][:, -1]

        probs = F.softmax(next_token_logits, dim=-1)
        (batch, vocab_size) = probs.shape

        if top_k is not None:
            (values, indices) = torch.topk(probs, k=top_k)
            probs[probs < values[:, -1, None]] = 0
            probs = probs / probs.sum(axis=1, keepdims=True)
        if sample:
            next_indices = torch.multinomial(probs, num_samples=1)
        else:
            next_indices = torch.argmax(probs, dim=-1)[:, None]
        input_ids = torch.cat([input_ids, next_indices], dim=1)
    
    output_completions = [tokenizer.decode(output.tolist()) for output in input_ids][0]

    return output_completions
            
