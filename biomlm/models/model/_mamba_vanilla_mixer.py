# Copyright (c) 2023, Albert Gu, Tri Dao.

"""
Adapted from 
https://github.com/state-spaces/mamba/blob/v1.2.0/mamba_ssm/models/mixer_seq_simple.py
"""

from functools import partial
import json
import os
from typing import Optional

import torch
import torch.nn as nn

from mamba_ssm.modules.mamba_simple import Mamba, Block

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from ._mamba_init_weights import _init_weights
from ._mamba_vanilla_config import BioSeqMambaVanlinaConfig
from ..utils import load_config_hf, load_state_dict_hf

def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


class BioSeqMixerModel(nn.Module):
    """ foundation model, Mamba based
    
    """
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        vocab_size: int,
        ssm_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        pad_vocab_size_multiple=8,
        tie_embeddings=True,
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
        )
        # self.config ={
        #     "d_model": d_model,
        #     "n_layer": n_layer,
        #     "vocab_size": vocab_size,
        #     "rms_norm": rms_norm,
        #     "residual_in_fp32": residual_in_fp32,
        #     "fused_add_norm": fused_add_norm,
        #     "pad_vocab_size_multiple": pad_vocab_size_multiple,
        #     "norm_epsilon": norm_epsilon,
        #     "tie_embeddings": tie_embeddings,

        #     "ssm_cfg": ssm_cfg,
        #     "initializer_cfg": initializer_cfg,
        # }
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)

        self.residual_in_fp32 = residual_in_fp32

        self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, inference_params=None):
        hidden_states = self.embedding(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        return hidden_states
    
    @classmethod
    def from_pretrained(
        cls, 
        pretrained_model_name_path, 
        config_file_name: Optional[str] ='mixer_config.json', 
        model_file_name: Optional[str] ='mixer_pytorch_model.bin', 
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
                        congif_file_name="mixer_config.json", 
                        model_file_name="mixer_pytorch_model.bin"):
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


