#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		_model_init_pretrained.py
@Time    :   	2024/05/17 11:49:43
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
from typing import Optional
import torch
import torch.nn as nn
from transformers import MambaForCausalLM, MambaConfig, PreTrainedModel

class MambaCache:
    """
    Arguments:
        config: MambaConfig
        batch_size: int
        dtype: torch.dtype
        device: torch.device

    Attributes:
        seqlen_offset: int
        dtype: torch.dtype
        conv_states: Dict[int, torch.Tensor] # layer_idx -> [batch_size, intermediate_size, conv_kernel_size]
        ssm_states: Dict[int, torch.Tensor] # layer_idx -> [batch_size, intermediate_size, ssm_state_size]
    """

    def __init__(
        self, config: MambaConfig, batch_size: int, dtype: torch.dtype = torch.float16, device: Optional[str] = None
    ):
        self.seqlen_offset = 0
        self.dtype = dtype
        intermediate_size = config.intermediate_size
        ssm_state_size = config.state_size
        conv_kernel_size = config.conv_kernel

        self.conv_states = {
            i: torch.zeros(batch_size, intermediate_size, conv_kernel_size, device=device, dtype=dtype)
            for i in range(config.num_hidden_layers)
        }
        self.ssm_states = {
            i: torch.zeros(batch_size, intermediate_size, ssm_state_size, device=device, dtype=dtype)
            for i in range(config.num_hidden_layers)
        }

class BioSeqMambaForCausalLM(PreTrainedModel):
    config_class = MambaConfig

    def __init__(
        self,
        config: MambaConfig = None,
        model_name_or_path: str = None,
        local_files_only: bool = True,
        **kwargs,
    ):  
        super().__init__(config, **kwargs)

        if model_name_or_path is None:
            # train from scrath
            self.model = MambaForCausalLM(config)
        else:
            # retained
            self.model = MambaForCausalLM.from_pretrained(
                model_name_or_path,
                local_files_only=local_files_only,
            )
            self.model.resize_token_embeddings(config.vocab_size)

        # rewrite config
        self.config = self.model.config
        self.config._name_or_path = 't2t_' + f"{config.vocab_size}_{config.hidden_size}"
        self.config.architectures = config.architectures
        self.config.model_type = config.model_type
        self.config.bos_token_id = config.bos_token_id
        self.config.eos_token_id = config.eos_token_id
        self.config.pad_token_id = config.pad_token_id

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,  # for now we need this for generation
    ):
        return self.model(
            input_ids = input_ids,
            labels=labels,
            return_dict=return_dict,
            **kwargs,  # for now we need this for generation
        )
    
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








