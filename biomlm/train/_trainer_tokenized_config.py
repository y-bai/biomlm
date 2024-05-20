#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		_trainer_config.py
@Time    :   	2024/03/11 13:06:53
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

@Desc    :   	parameter configuration for Mamba and training

"""

from dataclasses import dataclass, field
import os
from typing import List, Optional, Union
from enum import Enum
from transformers import TrainingArguments

class TokenModel(Enum):
    BPE = 'BPE'
    UNIGRAM = 'Unigram'
    SPM_BPE = 'SPM_BPE'
    SPM_UNIGRAM = 'SPM_Unigram'  # default

class SpeciesType(Enum):
    T2T = 'T2T'
    MULTI_SPECIES = 'Multi_species'
    ONEKG = '1000G'

# full sequence without chunking 
RAW_DATASET_DIRNAMES={
    'T2T': ['raw_dataset_chm13_t2t', 'raw_dataset_crgd_t2t'], # After tokenizing data invividually, combine them together
    # 'T2T': ['raw_dataset_chm13_t2t'], # tokenizing data invividually 
    'Multi_species': ['raw_dataset_multi'],
    '1000G': [], # TODO
}

# -----------
# for tokenization
VOCAB_SIZE = 50008
TOKEN_MODEL = TokenModel.BPE
SPECIES = SpeciesType.T2T
# SPECIES = SpeciesType.MULTI_SPECIES

RESUME_FROM_CHECKPOINT = False
MAX_LEN = 512 # 512 or 1024: max number of tokens in the sequence that feed into the model

PROJECT_ROOT_PATH=r"/home/share/huadjyin/home/baiyong01/projects/biomlm/"
# for raw datasets generation
USE_STREAMING = False
# alway True, otherwise pay more attention 
# to dataset.map in _tokenizer_map.py
USE_FAST_TOKENIZER = True
# for dataset mapping operation.
NUM_PROCS = 10

@dataclass
class BioSeqMambaModelConfig:

    d_model: int = field(
        default=1024,
        metadata={
            "help": (
                "The maximum dimension for the input token embedding."
            )
        },
    )
    n_layer: int = field(
        default=48,
        metadata={
            "help": (
                "The number of mamba blocks."
            )
        },
    )
    vocab_size: Optional[int] = field(
        default=VOCAB_SIZE,
        metadata={
            "help": (
                "The size of vocab."
            )
        },
    )

    ####################
    d_state: int = field(
        default=16,
        metadata={
            "help": (
                "SSM state expansion factor, (`N` in [1] Algorithm 2), default value from MambaConfig, "
                "No changes are needed unless there are special circumstances."
            )
        },
    )  
    
    expand: int = field(
        default=2,
        metadata={
            "help": (
                "SSM block expansion factor, (`E` in [1] Section 3.4), "
                "No changes are needed unless there are special circumstances."
            )
        },
    ) 

    dt_rank: str = field(
        default="auto",
        metadata={
            "help": (
                "SSM parameter, See [1] Section 3.6 Parameterization of ∆, Union[int, str] = 'auto' "
                "No changes are needed unless there are special circumstances."
            )
        },
    )

    dt_min: float = field(
        default=0.001,
        metadata={
            "help": (
                "SSM parameter, Range for initializing dt bias so that F.softplus(dt_bias) is "
                "between dt_min and dt_max No changes are needed unless there are special circumstances."
            )
        },
    )

    dt_max: float = field(
        default=0.1,
        metadata={
            "help": (
                "SSM parameter, Range for initializing dt bias so that F.softplus(dt_bias) is between dt_min and dt_max "
                "No changes are needed unless there are special circumstances."
            )
        },
    )

    dt_init: str = field(
        default="random",
        metadata={
            "help": (
                "SSM parameter for dt projection to preserve variance at initialization. constant or random "
                "No changes are needed unless there are special circumstances."
            )
        },
    )

    dt_scale: float = field(
        default=1.0,
        metadata={
            "help": (
                "SSM parameter for dt projection to preserve variance at initialization "
                "No changes are needed unless there are special circumstances."
            )
        },
    )

    dt_init_floor: float = field(
        default=1e-4,
        metadata={
            "help": (
                "SSM parameter for dt "
                "No changes are needed unless there are special circumstances."
            )
        },
    )

    d_conv: int = field(
        default=4,
        metadata={
            "help": (
                "width of 1d conv in SSM "
                "No changes are needed unless there are special circumstances."
            )
        },
    )

    conv_bias: bool = field(
        default=True,
        metadata={
            "help": (
                "whether add bias to the  1d conv in SSM "
                "No changes are needed unless there are special circumstances."
            )
        },
    )

    bias: bool = field(
        default=False,
        metadata={
            "help": (
                "whether add bias to the our project layer in SSM "
                "No changes are needed unless there are special circumstances."
            )
        },
    )

    use_fast_path: bool = field(
        default=True,
        metadata={
            "help": (
                "whether use fast path in SSM "
                "No changes are needed unless there are special circumstances."
            )
        },
    )

    ####################
    # for init model weight in SSM
    # see https://huggingface.co/state-spaces/mamba-2.8b-hf/blob/main/config.json
    # and https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/config_mamba.py
    initializer_range: float = 0.1  # 0.1, or 0.01 or 0.02
    rescale_prenorm_residual: bool = False
    n_residuals_per_layer: int = 1

    # for computation improvement
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8                        # for falsh attention
    tie_embeddings: bool = True

    # only for mixer
    norm_epsilon: float = 1e-5

@dataclass
class BioSeqDataSetConfig:

    dataset_name: Optional[str] = field(
        default=SPECIES.value,
        metadata={
            "help": (
                "Raw data type, currently could be 'T2T', 'Multi_species' or '1000G'."
            )
        },
    )

    raw_dataset_names = RAW_DATASET_DIRNAMES[SPECIES.value]

    raw_dataset_dirs: Optional[Union[List[str], str]] = None

    def __post_init__(self):
        
        if self.raw_dataset_dirs is None:
            self.raw_dataset_dirs = [os.path.join(PROJECT_ROOT_PATH, f'biomlm/datasets/{i_dir}') for i_dir in RAW_DATASET_DIRNAMES[SPECIES.value]]
        elif isinstance(self.raw_dataset_dirs, str):
            self.raw_dataset_dirs = [self.raw_dataset_dirs]

@dataclass
class BioSeqTokenizationConfig:

    token_model_type: str = TOKEN_MODEL

    model_max_length: Optional[int] = field(
        default=MAX_LEN,
        metadata={
            "help": (
                "context length for sequence, i.e., the max number of tokens "
                "in a sequence when inputting the model. "
            )
        }
    )

    padding_side: Optional[str] = "right"  # predict on left
    
    bos_token: str = "<BOS>"
    eos_token: str = "<EOS>"
    unk_token: str = "<UNK>"

    add_bos_token: bool=False
    add_eos_token: bool=False

    overwrite_cache: bool = field(
        default=False, 
        metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    use_fast_tokenizer: bool = field(
        default=USE_FAST_TOKENIZER,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )

    # too large will cause memery out leakage
    # RuntimeError: One of the subprocesses has abruptly died during map operation.To debug the error, disable multiprocessing.
    token_num_proc: Optional[int] = NUM_PROCS   

    truncation: Optional[bool] = True
    padding: Optional[str] = "max_length"
    stride: Optional[int] = 16
    return_overflowing_tokens: Optional[bool] = True

    token_min_ctx_fraction: Optional[float] = 1 # when training, we only use the full sequence without padding.
    tokenizer_pretrained_dir: Optional[str] = None

    tokenized_dataset_root_dir: Optional[str] = None
    tokenized_dataset_dir: Optional[str] = None
    tokenized_shuffled_dataset_dir: Optional[str] = None
    
    use_streaming: bool = USE_STREAMING

    def __post_init__(self):

        if self.tokenizer_pretrained_dir is None:
            if SPECIES in [SpeciesType.T2T, SpeciesType.ONEKG]:
                t_type = 'T2T'
            else:
                t_type = SPECIES.value
            self.tokenizer_pretrained_dir = os.path.join(
                PROJECT_ROOT_PATH, 
                f"biomlm/tokens/20000_200/{t_type}/{TOKEN_MODEL.value}/{VOCAB_SIZE}"
            )
        if self.tokenized_dataset_root_dir is None:
            self.tokenized_dataset_root_dir = os.path.join(
                    PROJECT_ROOT_PATH, 
                    f"biomlm/datasets/{SPECIES.value}_{TOKEN_MODEL.value}_{VOCAB_SIZE}_{MAX_LEN}"
                )
        if self.tokenized_dataset_dir is None:
            if len(RAW_DATASET_DIRNAMES[SPECIES.value]) > 1:
                # combining tokenized dataset together
                self.tokenized_dataset_dir = os.path.join(
                    PROJECT_ROOT_PATH, 
                    f"biomlm/datasets/{SPECIES.value}_{TOKEN_MODEL.value}_{VOCAB_SIZE}_{MAX_LEN}/COMB"
                )
            else:
                # tokenizing dataset individually
                self.tokenized_dataset_dir = os.path.join(
                    PROJECT_ROOT_PATH, 
                    f"biomlm/datasets/{SPECIES.value}_{TOKEN_MODEL.value}_{VOCAB_SIZE}_{MAX_LEN}/{RAW_DATASET_DIRNAMES[SPECIES.value][0]}"
                )
        
        if self.tokenized_shuffled_dataset_dir is None:
            self.tokenized_shuffled_dataset_dir = os.path.join(
                PROJECT_ROOT_PATH, 
                f"biomlm/datasets/{SPECIES.value}_{TOKEN_MODEL.value}_{VOCAB_SIZE}_{MAX_LEN}/SHUFFLED"
            )


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is only for debug
@dataclass
class BioSeqMambaCausalLMTrainingConfigDebug(TrainingArguments):

    output_dir: str = os.path.join(
        PROJECT_ROOT_PATH, 
        f"biomlm/outputs/{SPECIES.value}_{TOKEN_MODEL.value}_{VOCAB_SIZE}_{MAX_LEN}/")
    overwrite_output_dir: bool = True

    # The weight decay to apply (if not zero) to all layers 
    # except all bias and LayerNorm weights in AdamW optimizer.
    weight_decay: float = 1e-4          # 1e-4          
    adam_beta1:float = 0.9              # default for AdamW
    adam_beta2:float = 0.99             #0.999
    adam_epsilon:float = 1e-8
    
    # NOTE: such setting, almost 15 minutes for 100 step (training + evaluation)
    #
    # update: training: 100 steps: 3 minutes, eval: 80 second
    #
    # 3010 run 20 epoch done
    # num_train_epochs:float = 45        # full train: 10 (every run) 
    
    # 
    # https://discuss.huggingface.co/t/streaming-dataset-into-trainer-does-not-implement-len-max-steps-has-to-be-specified/32893/7
    max_steps:int = 800  #   
    #   
    # NOTE: to complete the whole training data during one epoch, 
    # there needs  steps = (n_whole_training_samples / (per_device_train_batch_size * n_GPU * gradient_accumulation_steps))
    # 1 epoch ~= almost 4533 steps (T2T training, 512 token_seq_len)
    
    #
    # optim: str = "adamw_torch" # default optimizer is adamw_torch
    #

    learning_rate: float = 5e-4   # 1e-4
    # linear: transformers.get_linear_schedule_with_warmup
    # cosine: transformers.get_cosine_schedule_with_warmup
    # cosine_with_restarts: transformers.get_cosine_with_hard_restarts_schedule_with_warmup
    # polynomial: transformers.get_polynomial_decay_schedule_with_warmup
    # constant: transformers.get_constant_schedule
    # constant_with_warmup: transformers.get_constant_schedule_with_warmup
    # inverse_sqrt: transformers.get_inverse_sqrt_schedule
    # https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_utils.py#L402
    lr_scheduler_type: str = "cosine_with_restarts" 
    # `num_steps_per_cycle` used for computing the `num_cycles`  
    num_steps_per_cycle: int = 10

    ############
    # warmup_ratio:float = 0.05           
    warmup_steps: int = 10      # full train 600

    dataloader_num_workers: int = 4   # full train 16

    do_train: bool = True
    do_eval: bool = True       

    max_grad_norm:float = 1.0
    #############
    gradient_accumulation_steps: int = 2  # full train 2
    
    per_device_train_batch_size: int = 16 # full train 32, (3009 vocab)
    per_device_eval_batch_size: int = 16 # full train 64

    # NOTE: config for evaluation, NOT take too long
    evaluation_strategy: str = "steps"  # "epoch" would wait for long time to have output next epoch
    #
    # if evaluation_strategy="steps". eval_steps will default to the same 
    # value as logging_steps if not set.
    # eval_steps must be an integer if bigger than 1
    eval_steps: int = 2               # full train 200
    #
    # Number of predictions steps to accumulate the output tensors for, 
    # before moving the results to the CPU. 
    # If left unset, the whole predictions are accumulated on 
    # GPU/NPU/TPU before being moved to the CPU (faster but requires more memory).
    eval_accumulation_steps: int = 1 # full train 200
    # 
    # NOTE: model load config
    # load_best_model_at_end requires the save and eval strategy to match.
    load_best_model_at_end: bool = True 
    # metric_for_best_model: str = None # for using loss if None
    greater_is_better: bool = False
    # 
    # # EarlyStoppingCallback requires load_best_model_at_end = True
    early_stop: bool = True
    # # config for early stop call back
    early_stopping_patience: int = 20
    early_stopping_threshold: float = 0.0001
 
    # NOTE: save config
    save_steps: int = 10       # full train 500
    save_strategy: str = "steps"
    # If a value is passed, will limit the total amount of checkpoints. 
    # Deletes the older checkpoints in output_dir.
    save_total_limit: int = 3 
    
    # NOTE: logging config 
    # https://stackoverflow.com/questions/73281901/is-there-a-way-to-plot-training-and-validation-losses-on-the-same-graph-with-hug
    # TensorBorad log dir
    logging_dir: str = os.path.join(
        PROJECT_ROOT_PATH, 
        f"biomlm/outputs/{SPECIES.value}_{TOKEN_MODEL.value}_{VOCAB_SIZE}_{MAX_LEN}/log")
    logging_steps: int = 2 # full train 100  for training loss displaying every two steps
    logging_strategy: str = "steps"
    # when installed 'tensorboard', then model_config_json = model.config.to_json_string()
    # AttributeError: dict object has no attribute 'to_json_string'
    # https://github.com/huggingface/peft/pull/1200
    # Thus, have to build a Config class that was inheritted PretrainedConfig class
    report_to: str = "tensorboard"
    # If there multiple events in the log_dir, plot will be mass in the tensorboard.
    # Thus, need to delete the previous results. 

    # If True, For the first run, when runing `training_config.main_process_first(desc="dataset map tokenization")` could cause
    # RuntimeError: [3] is setting up NCCL communicator 
    # and retrieving ncclUniqueId from [0] via c10d key-value store by key '0', but store->get('0') got error: Socket Timeout
    # see: https://github.com/stas00/ml-engineering/issues/1
    # https://github.com/huggingface/transformers/issues/15618
    # NOTE 
    sharded_ddp: bool = True   # speed up training under multi-GPU

    # https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.ddp_timeout
    ddp_timeout: int = 60 * 60 * 1 # 1-hour

    # find_unused_parameters in DistributedDataParallel
    # NOTE
    ddp_find_unused_parameters: bool = False


    # NOTE
    # https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Trainer.train.resume_from_checkpoint
    # If a str, local path to a saved checkpoint as saved by a previous instance of Trainer. 
    # If a bool and equals True, load the last checkpoint in args.output_dir as saved by a 
    # previous instance of Trainer. If present, training will resume from the model/optimizer/scheduler states loaded here.
    #
    resume_from_checkpoint: bool = RESUME_FROM_CHECKPOINT 

    seed: int = 42
    data_seed: int = 42

    # ----------
    # Only used for fine-tuning
    label_smoothing_factor: float=0.0 # or 0.01

    #
    # If input does not contained labels, then we need to use this
    # include_inputs_for_metrics: bool = True

    #
    disable_tqdm: bool = False # full train True

    # NOTE
    # BF16 can represent a wider range of integers, but with less precision in the mantissa; 
    # FP16 has a smaller integer range, but with higher mantissa precision.(Thus, fp16 is too prone to overflow.)
    # Thus, Inference is suited for fp16, training is suited for bf16.
    # TF32 uses the same 10-bit mantissa precision as half-precision (FP16) math, which is far beyond the precision requirements for AI workloads, 
    # providing ample margin. At the same time, TF32 uses the same 8-bit exponent as FP32, supporting the same numerical range.
    #
    # bf16: bool = True # v100 not support , residual_in_fp32 not support
    #
    tf32: bool = False # full train True

    # NOTE
    # fine tuning
    # https://huggingface.co/learn/nlp-course/chapter3/3?fw=pt#fine-tuning-a-model-with-the-trainer-api

    # NOTE
    # Hyperparameter Search
    # https://huggingface.co/docs/transformers/en/hpo_train

    #
    # if False, pad id will be replaced with -100 by `DataCollatorForLanguageModeling`,
    # and thus, pad id will be ignored by the CrossEntropy loss.
    keep_original_pad_token_ids: bool = False 

    # for debug
    max_train_samples: Optional[int] = field(
        default=500,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    
    max_eval_samples: Optional[int] = field(
        default=500,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )


