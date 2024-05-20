#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		_trainer.py
@Time    :   	2024/03/12 17:14:05
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

import torch
import torch.nn as nn
# from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# import transformers
from transformers import Trainer
from transformers.optimization import get_scheduler
from transformers.modeling_utils import unwrap_model
from transformers.trainer_pt_utils import LabelSmoother

# from ._trainer_optimization import CosineAnnealingWarmupRestarts

class BioSeqMambaCausalLMTrainer(Trainer):

    # def _load_optimizer_and_scheduler(self, checkpoint):
    #     """Not load"""
    #     return

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.args.label_smoothing_factor != 0:
            return super().compute_loss(model, inputs, return_outputs=return_outputs)
        else:
            return self.ce_loss(model, inputs, return_outputs=return_outputs)
    
    # def compute_loss(self, model, inputs, return_outputs=False):
    #     """
    #     How the loss is computed by Trainer. By default, all models return the loss in the first element.

    #     Subclass and override for custom behavior.

    #     Adaped from transformers.trainer.compute_loss
    #     """
    #     labels = inputs.pop("labels")
    #     outputs = model(**inputs)
    #     # Save past state if it exists
    #     # TODO: this needs to be fixed and made cleaner later.
    #     if self.args.past_index >= 0:
    #         self._past = outputs[self.args.past_index]

    #     if labels is not None:
    #         loss_cls = LabelSmoother(epsilon=self.args.label_smoothing_factor)
    #         loss = loss_cls(outputs, labels, shift_labels=True)
    #     else:
    #         if isinstance(outputs, dict) and "loss" not in outputs:
    #             raise ValueError(
    #                 "The model did not return a loss from the inputs, only the following keys: "
    #                 f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
    #             )
    #         # We don't use .loss here since the model may return tuples instead of ModelOutput.
    #         loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

    #     return (loss, outputs) if return_outputs else loss
    
    def ce_loss(self, model, inputs, return_outputs: bool = False):
        """ override loss function

        """
        input_ids = inputs.pop("input_ids")
        labels = inputs.pop("labels")
        clm_output = model(input_ids, labels, return_dict=True)
        
        clm_loss = None
        if "loss" in clm_output:
            clm_loss = clm_output["loss"] 

        if clm_loss is not None:
            return (clm_loss, clm_output) if return_outputs else clm_loss
        else:
            clm_logits = clm_output["logits"]
            labels = labels.to(clm_logits.device)
            # labels = inputs.pop("labels")
            # labels = inputs.get("labels")
            # print(f"loss_clm_logits: {clm_logits.shape}")
            # print(f"loss_labels: {labels.shape}")
            # labels: torch.Size([2, 511])
            # clm_logits: torch.Size([2, 512, 3016])

            shift_logits = clm_logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing_factor)
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            return (lm_loss, clm_output) if return_outputs else lm_loss
    
    # Overried the `create_scheduler` function.
    def create_scheduler(
        self, 
        num_training_steps: int, 
        optimizer: torch.optim.Optimizer = None
    ):
        if self.lr_scheduler is None:
            n_warm_steps = self.args.get_warmup_steps(num_training_steps)
            _no_warm_steps = num_training_steps - n_warm_steps
            if self.args.lr_scheduler_type.startswith("cosine"):
                _scheduler_specific_kwargs = {
                    "num_cycles": max(10, _no_warm_steps//self.args.num_steps_per_cycle)}
                    # "num_cycles": 2}
            else: 
                _scheduler_specific_kwargs = {}

            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=n_warm_steps,
                num_training_steps=num_training_steps,
                scheduler_specific_kwargs=_scheduler_specific_kwargs,
            )
            self._created_lr_scheduler = True

        return self.lr_scheduler
    
    # To solve the issue:
    # RuntimeError: Some tensors share memory, this will lead to duplicate memory on disk and potential 
    # differences when loading them again: [{'lm_head.weight', 'backbone.embedding.weight'}].
    # A potential way to correctly save your model is to use `save_model`.
    # More information at https://huggingface.co/docs/safetensors/torch_shared_tensors
    def save_model(self, output_dir, _internal_call:bool = False):
        self.model.save_pretrained(output_dir)
        # self.model.backbone.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

