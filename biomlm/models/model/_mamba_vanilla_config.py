#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		_mamba_vanilla_config.py
@Time    :   	2024/03/19 16:39:00
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

    [1] Gu, Albert, and Tri Dao. "Mamba: Linear-time sequence modeling with selective state spaces." 
    arXiv preprint arXiv:2312.00752 (2023). https://arxiv.org/abs/2312.00752

"""

from transformers import PretrainedConfig

class BioSeqMambaVanlinaConfig(PretrainedConfig):
    model_type = "bioseq_mamba_vanlina"

    def __init__(
        self,
        d_model=1024,
        n_layer=48,
        vocab_size=3009,
        ssm_cfg=None,
        rms_norm=True,
        initializer_cfg=None,
        fused_add_norm=True,
        residual_in_fp32=True,
        norm_epsilon=1e-5,
        tie_embeddings=True,
        pad_vocab_size_multiple=8, 
        **kwargs
    ):
        self.d_model=d_model
        self.n_layer=n_layer
        self.vocab_size=vocab_size
        self.ssm_cfg=ssm_cfg
        self.rms_norm=rms_norm
        self.initializer_cfg=initializer_cfg
        self.fused_add_norm=fused_add_norm
        self.residual_in_fp32=residual_in_fp32
        self.norm_epsilon=norm_epsilon
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
        self.tie_embeddings=tie_embeddings

        super().__init__(**kwargs)


