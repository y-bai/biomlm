#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Author  :   	Albert Gu, Tri Dao. 
@License :   	(C)Copyright 2023-2024, Albert Gu, Tri Dao.

                Licensed under the Apache License, Version 2.0 (the "License");
                you may not use this file except in compliance with the License.
                You may obtain a copy of the License at

                    http://www.apache.org/licenses/LICENSE-2.0

                Unless required by applicable law or agreed to in writing, software
                distributed under the License is distributed on an "AS IS" BASIS,
                WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
                See the License for the specific language governing permissions and
                limitations under the License.

@Desc    :   	Causal LM for bio sequence.

Adapted from 
https://github.com/state-spaces/mamba/blob/v1.2.0/mamba_ssm/models/mixer_seq_simple.py
"""

import os
import json
from collections import namedtuple
from functools import partial
from typing import Optional

import torch
import torch.nn as nn

from mamba_ssm.utils.generation import GenerationMixin

from ._mamba_vanilla_mixer import BioSeqMixerModel
from ._mamba_vanilla_config import BioSeqMambaVanlinaConfig
from ._mamba_init_weights import _init_weights
from ..utils import load_config_hf, load_state_dict_hf

class BioSeqForCausalLM(nn.Module, GenerationMixin):

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
        device=None, 
        dtype=None,
        **kwargs,
    ) -> None:
        
        # this is only for save the config
        self.config = BioSeqMambaVanlinaConfig(
            d_model=d_model,
            n_layer=n_layer,
            vocab_size=vocab_size,
            ssm_cfg=ssm_cfg,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            norm_epsilon=norm_epsilon,
            pad_vocab_size_multiple = pad_vocab_size_multiple,
            tie_embeddings=tie_embeddings,
            **kwargs,
        ) 

        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()

        # init_vocab_size = vocab_size
        
        self.backbone = BioSeqMixerModel(
            d_model=d_model,
            n_layer=n_layer,
            vocab_size=vocab_size,
            ssm_cfg=ssm_cfg if ssm_cfg is not None else {},
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            norm_epsilon=norm_epsilon,
            pad_vocab_size_multiple = pad_vocab_size_multiple,
            tie_embeddings=tie_embeddings,
            **factory_kwargs,
        )

        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)
        # self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs, **kwargs)
        # self.lm_head = nn.Linear(d_model, init_vocab_size, bias=False, **factory_kwargs)

        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

        # Tie output projection to embedding weights.
        # See "Weight Tying" paper
        self.tie_weights()

    def tie_weights(self):
        if self.config.tie_embeddings:
            self.lm_head.weight = self.backbone.embedding.weight
    
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(
            self, 
            input_ids, 
            position_ids=None,
            attention_mask=None,
            labels=None,
            inference_params=None, 
            num_last_tokens=0
    ):
        """
        "position_ids", labels and "attention_mask" are just to be compatible with Transformer generation. 
        We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        max_seq_len = input_ids.shape[-1]
        self.config.max_length = max_seq_len

        hidden_states = self.backbone(input_ids, inference_params=inference_params)
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        lm_logits = self.lm_head(hidden_states)
        # CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        # return CausalLMOutput(logits=lm_logits)
        return {"logits": lm_logits, 'hidden_states': hidden_states}

    @classmethod
    def from_pretrained(
        cls, 
        pretrained_model_name_path, 
        config_file_name: Optional[str]='config.json', 
        model_file_name: Optional[str]='pytorch_model.bin', 
        local_files_only=True, 
        device=None, 
        dtype=None, 
        **kwargs
    ):
        config = load_config_hf(
            pretrained_model_name_path, 
            file_name=config_file_name, 
            local_files_only=local_files_only)
        
        model = cls(device=device, dtype=dtype, **config, **kwargs)
        model.load_state_dict(load_state_dict_hf(
            pretrained_model_name_path, 
            file_name=model_file_name, 
            local_files_only=local_files_only,
            device=device, dtype=dtype))
        
        return model

    def save_pretrained(self, save_directory,
                        congif_file_name="config.json", 
                        model_file_name="pytorch_model.bin"):
        """
        Minimal implementation of save_pretrained for MambaLMHeadModel.
        Save the model and its configuration file to a directory.
        """
        # Ensure save_directory exists
        if not os.path.exists(save_directory):
            # multiple GPUs will raise FileExistsError if exist_ok=False(default value)
            os.makedirs(save_directory, exist_ok=True)

        # Save the model's state_dict
        model_path = os.path.join(save_directory, model_file_name)
        torch.save(self.state_dict(), model_path)

        # Save the configuration of the model
        config_path = os.path.join(save_directory, congif_file_name)
        self.config.to_json_file(config_path)
        # with open(config_path, 'w', encoding='utf-8') as f:
        #     model_cfg ={
        #         "d_model": self.config.d_model,
        #         "n_layer": self.config.n_layer,
        #         "vocab_size": self.config.vocab_size,
        #         "rms_norm": self.config.rms_norm,
        #         "residual_in_fp32": self.config.residual_in_fp32,
        #         "fused_add_norm": self.config.fused_add_norm,
        #         "pad_vocab_size_multiple": self.config.pad_vocab_size_multiple,
        #         "norm_epsilon": self.config.norm_epsilon,
        #         "tie_embeddings": self.config.tie_embeddings,
        #         "ssm_cfg": self.config.ssm_cfg, # dict
        #         "initializer_cfg": self.config.initializer_cfg, # dict
        #     }
        #     json.dump(model_cfg, f)
        #     # json.dump(self.config.to_json_string(), f)
