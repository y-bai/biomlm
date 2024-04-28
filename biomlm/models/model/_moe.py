#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		_moe.py
@Time    :   	2024/03/20 16:48:05
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

@Desc    :   	Switch Transformer MoE design

Similar to  https://arxiv.org/pdf/2401.04081.pdf

https://huggingface.co/docs/transformers/en/model_doc/switch_transformers
https://huggingface.co/blog/moe
https://etc.cuit.columbia.edu/news/basics-language-modeling-transformers-switch-transformer

https://github.com/huggingface/transformers/blob/v4.39.0/src/transformers/models/switch_transformers/modeling_switch_transformers.py

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# copied from transformers/models/switch_transformers/modeling_switch_transformers.py
def router_z_loss_func(router_logits: torch.Tensor) -> float:
    r"""
    Compute the router z-loss implemented in PyTorch.

    The router z-loss was introduced in [ST-MoE: Designing Effective Sparse Expert Models](https://arxiv.org/abs/2202.08906).
    It encourages router logits to remain small in an effort to improve stability.

    Args:
        router_logits (`float`):
            Input logits of shape [batch_size, sequence_length, num_experts]

    Returns:
        Scalar router z-loss.
    """
    num_groups, tokens_per_group, _ = router_logits.shape
    log_z = torch.logsumexp(router_logits, dim=-1)
    z_loss = log_z**2
    return torch.sum(z_loss) / (num_groups * tokens_per_group)

# copied from transformers/models/switch_transformers/modeling_switch_transformers.py
def load_balancing_loss_func(router_probs: torch.Tensor, expert_indices: torch.Tensor) -> float:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        router_probs (`torch.Tensor`):
            Probability assigned to each expert per token. Shape: [batch_size, seqeunce_length, num_experts].
        expert_indices (`torch.Tensor`):
            Indices tensor of shape [batch_size, seqeunce_length] identifying the selected expert for a given token.

    Returns:
        The auxiliary loss.
    """
    num_experts = router_probs.shape[-1]

    # cast the expert indices to int64, otherwise one-hot encoding will fail
    if expert_indices.dtype != torch.int64:
        expert_indices = expert_indices.to(torch.int64)

    if len(expert_indices.shape) == 2:
        expert_indices = expert_indices.unsqueeze(2)

    expert_mask = F.one_hot(expert_indices, num_experts)

    # For a given token, determine if it was routed to a given expert.
    expert_mask = torch.max(expert_mask, axis=-2).values

    # cast to float32 otherwise mean will fail
    expert_mask = expert_mask.to(torch.float32)
    tokens_per_group_and_expert = torch.mean(expert_mask, axis=-2)

    router_prob_per_group_and_expert = torch.mean(router_probs, axis=-2)
    return torch.mean(tokens_per_group_and_expert * router_prob_per_group_and_expert) * (num_experts**2)


class SwithMoETop1Router(nn.Module):
    r"""
    Switch layer.

    Switch layer, proposed in Switch Transformer (https://arxiv.org/abs/2101.03961), is a simplified 
    strategy where input is routed to only a single expert, i.e. k=1. This simplification preserves model 
    quality, reduces routing computation and performs better.

    https://sh-tsang.medium.com/brief-review-switch-transformers-scaling-to-trillion-parameter-models-with-simple-and-efficient-880edbbf4890
    
    """
    def __init__(
        self,
    ) -> None:
        super().__init__()




class DeprecatedExpert(nn.Module):
    def __init__(
        self, 
        d_input,
        d_hidden,
        d_output,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.ffn = nn.Sequential(
            nn.Linear(d_input, d_hidden, **factory_kwargs),
            nn.ReLU(),
            nn.Linear(d_hidden, d_output, **factory_kwargs),
        )
    
    def forward(self, x):
        return self.ffn(x)
    
class DeprecatedMixtureOfExperts(nn.Module):
    def __init__(
        self, 
        d_input,
        d_hidden,
        d_output,
        n_experts,
        top_k_experts=1,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.n_experts = n_experts
        self.top_k_experts = top_k_experts

        # Router: MLP to generate logits for expert selection
        self.router = nn.Linear(d_input, n_experts, **factory_kwargs)

        # Experts: a list of expert (FFNs)
        self.experts = nn.ModuleList(
            [
                DeprecatedExpert(d_input, d_hidden, d_output, **factory_kwargs) 
                for _ in range(n_experts)
            ]
        )

    def forward(self, x):
        batch_size, seq_len, d_input = x.shape
        x_flat = x.view(-1, d_input) # Flatten to [B*SEQLEN, d_input]

        # Routing tokens to experts
        router_logits = self.router(x_flat)

        topk_logits, topk_indices = router_logits.topk(
            self.top_k, dim=1
        )

        topk_gates = F.softmax(
            topk_logits, dim=1
        )  # Normalizing the top-k logits

        # Initializing the output
        output_flat = torch.zeros(
            batch_size * seq_len,
            self.experts[0].network[-1].out_features,
            device=x.device,
        )

        # Distributing tokens to the experts and aggregating the results
        for i in range(self.top_k):
            expert_index = topk_indices[:, i]
            gate_value = topk_gates[:, i].unsqueeze(1)

            expert_output = torch.stack(
                [
                    self.experts[idx](x_flat[n])
                    for n, idx in enumerate(expert_index)
                ]
            )

            output_flat += gate_value * expert_output

        # Reshape the output to the original input shape [B, SEQLEN, expert_output_dim]
        output = output_flat.view(batch_size, seq_len, -1)
        return output