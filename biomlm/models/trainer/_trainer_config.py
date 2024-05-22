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
    # 'T2T': ['raw_dataset_chm13_t2t', 'raw_dataset_crgd_t2t'],
    'T2T': ['raw_dataset_chm13_t2t'],
    'Multi_species': ['raw_dataset_multi'],
    '1000G': [], # TODO
}

# -----------
# for tokenization
TOKEN_MODEL = TokenModel.BPE
SPECIES = SpeciesType.T2T
# SPECIES = SpeciesType.MULTI_SPECIES

VOCAB_SIZE = 10008
HIDDENSIZE=1536 # 2048: 2048 OR 1536 

# if both SELF_PRETAINED_MODEL=False and HF_PRETAINED_MODEL=False, then train from scratch.
HF_PRETAINED_MODEL=False
SELF_PRETAINED_MODEL=False
TOKENIZED_DATA_SHUFFLED = False

USE_MIXED_PRECISION = False
RESUME_FROM_CHECKPOINT = False

D_MODEL = {
    512: 512,
    768: 768, # hidden_size: d_model
    1024: 1024,  
    1536: 2048,
    2048: 2048,
}

MODEL_N_LAYER={
    512: 24,
    768: 24, # hidden_size: n_layer
    1024: 48,  
    1536: 48,
    2048: 48,
}

# max batch_size, otherwise GPU out_of_memory
BATCH_SIZE={
    5008:{ # vocab_size
        512: { # max_length
            # hidden_size: batch_size
            512: 112,   # GPU: 34997MiB   
            768: 80,    # GPU: 36933MiB
            1024: 24,   # GPU: 33015MiB
            1536: 12,   # GPU: 34253MiB
            2048: 4,    # GPU: 36797MiB
        },
    },
    10008:{ # vocab_size
        512: { # max_length
            # hidden_size: batch_size
            512: 104,   # GPU: 36693MiB
            768: 72,    # GPU: 35637MiB
            1024: 24,   # GPU: 34125MiB
            1536: 12,   # GPU: 34855MiB
            2048: 4,    # GPU: 37065MiB
        },
    },
    50008:{ # vocab_size
        512: { # max_length
            # hidden_size: batch_size
            512: 56,    # GPU: 35347MiB
            768: 48,    # GPU: 35991MiB
            1024: 24,   # GPU: 35813MiB
            1536: 4,    # GPU: 39641MiB
            2048: 4,    # GPU: 39641MiB
        },
        1024:{
            768:16,
            1024: 12,   # GPU: 35435MiB
        },
        2048:{
            768:8,
            1024: 4,
        },
    },
}

MAX_STEPS={ # hidden_size: max_steps
    512:  82000,
    768:  82000,
    1024: 82000,
    1536: 82000, 
    2048: 82000,
}

PROJECT_ROOT_PATH=r"/home/share/huadjyin/home/baiyong01/projects/biomlm/"
OUTPUT_SUBDIR = "USE_HF_PRETRAINED" if HF_PRETAINED_MODEL else "TRAINED_FROM_SCRATCH"

# for raw datasets generation
USE_STREAMING = False
# alway True, otherwise pay more attention 
# to dataset.map in _tokenizer_map.py
USE_FAST_TOKENIZER = True
# for dataset mapping operation.

NUM_PROCS = 5
MAX_LEN = 512 # 512 or 1024: max number of tokens in the sequence that feed into the model when training

HF_PRETRAINED_NAMES= { # hidden size: "model name" 
    768: "mamba-130m-hf",
    1024: "mamba-370m-hf",
    1536: "mamba-790m-hf",
    2408: "mamba-1.4b-hf",
}

@dataclass
class BioSeqMambaModelConfig:

    d_model: int = field(
        default=D_MODEL[HIDDENSIZE],
        metadata={
            "help": (
                "The maximum dimension for the input token embedding."
            )
        },
    )
    n_layer: int = field(
        default=MODEL_N_LAYER[HIDDENSIZE],
        metadata={
            "help": (
                "The number of mamba blocks."
            )
        },
    )

    hidden_size: int = HIDDENSIZE

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

    hidden_act: str = "silu"

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

    hf_pretrained_name_path: str = None
    self_pretrained_name_path: str = None

    self_pretrained: bool = SELF_PRETAINED_MODEL

    def __post_init__(self):
        if self.hf_pretrained_name_path is None and HF_PRETAINED_MODEL and not SELF_PRETAINED_MODEL:
            self.hf_pretrained_name_path = os.path.join(PROJECT_ROOT_PATH, f"biomlm/hf_pretrained/{HF_PRETRAINED_NAMES[HIDDENSIZE]}")
        
        if self.self_pretrained_name_path is None and SELF_PRETAINED_MODEL:
            self.self_pretrained_name_path = os.path.join(PROJECT_ROOT_PATH, f"biomlm/biomlm_checkpoint/{SPECIES.value}_{TOKEN_MODEL.value}_{VOCAB_SIZE}_{MAX_LEN}_{HIDDENSIZE}/{OUTPUT_SUBDIR}")


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
            self.raw_dataset_dirs = [os.path.join(PROJECT_ROOT_PATH, f'biomlm/datasets/{i_dir}') for i_dir in RAW_DATASET_DIRNAMES[self.dataset_name]]
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
    tokenized_data_shuffled: bool = TOKENIZED_DATA_SHUFFLED
    
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


@dataclass
class BioSeqMambaCausalLMTrainingConfig(TrainingArguments):

    # deepspeed = DEEPSPEED_CONFIG # r"/home/share/huadjyin/home/baiyong01/projects/biomlm/biomlm/train/dspeed_config.json"
    # output_dir: str = os.path.join(
    #     PROJECT_ROOT_PATH, 
    #     f"biomlm/outputs/{SPECIES.value}_{TOKEN_MODEL.value}_{VOCAB_SIZE}_{MAX_LEN}_{HIDDENSIZE}/")

    output_dir: str = os.path.join(
        PROJECT_ROOT_PATH, 
        f"biomlm/outputs/{SPECIES.value}_{TOKEN_MODEL.value}_{VOCAB_SIZE}_{MAX_LEN}_{HIDDENSIZE}/{OUTPUT_SUBDIR}")
    overwrite_output_dir: bool = True 
    #   
    # NOTE: to complete the whole training data during one epoch, 
    # there needs  steps = (n_whole_training_samples / (per_device_train_batch_size * n_GPU * gradient_accumulation_steps))
    # 1 epoch ~= almost 4533 steps (T2T training, 512 token_seq_len) 
    #
    learning_rate: float = 6e-4 if not SELF_PRETAINED_MODEL else 6e-5
    # linear: transformers.get_linear_schedule_with_warmup
    # cosine: transformers.get_cosine_schedule_with_warmup
    # cosine_with_restarts: transformers.get_cosine_with_hard_restarts_schedule_with_warmup
    # polynomial: transformers.get_polynomial_decay_schedule_with_warmup
    # constant: transformers.get_constant_schedule
    # constant_with_warmup: transformers.get_constant_schedule_with_warmup
    # inverse_sqrt: transformers.get_inverse_sqrt_schedule
    # https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_utils.py#L402 
    # lr_scheduler_type: str = "constant_with_warmup"
    # lr_scheduler_type: str = "linear"
    lr_scheduler_type: str = "cosine" # "cosine_with_restarts" if not SELF_PRETAINED_MODEL else "linear"
    # NOTE `num_steps_per_cycle` used for computing the `num_cycles`  
    num_steps_per_cycle: int = 2000  # should be smaller, like 1000, during training to improve accuracy quickly. But larger during fine-tuning.
    ############
    # warmup_ratio:float = 0.05           
    warmup_steps: int = 1000   
    # 
    # https://discuss.huggingface.co/t/streaming-dataset-into-trainer-does-not-implement-len-max-steps-has-to-be-specified/32893/7
    max_steps:int = MAX_STEPS[HIDDENSIZE]  #  50008-512-1024-24: ~100 epoch, ~ 2.8 days to finish training

    dataloader_num_workers: int = 48      
    #############
    gradient_accumulation_steps: int = 2  
    # 24: 50008-512-1024:  GPU: 35813MiB, (vocab_size, max_length, d_model)
    #  4: 50008-512-2048,  GPU: 39641MiB
    # 12: 50008-1024-1024, GPU: 35435MiB
    # 48: 50008-512-768,   GPU: 35991MiB
    # 56: 50008-512-512,   GPU: 35347MiB
    per_device_train_batch_size: int = BATCH_SIZE[VOCAB_SIZE][MAX_LEN][HIDDENSIZE] 
    per_device_eval_batch_size: int = BATCH_SIZE[VOCAB_SIZE][MAX_LEN][HIDDENSIZE]

    # NOTE: config for evaluation, NOT take too long
    evaluation_strategy: str = "steps"  # "epoch" would wait for long time to have output next epoch
    #
    # if evaluation_strategy="steps". eval_steps will default to the same 
    # value as logging_steps if not set.
    # eval_steps must be an integer if bigger than 1
    eval_steps: int = 200
    
    # NOTE: logging config 
    # https://stackoverflow.com/questions/73281901/is-there-a-way-to-plot-training-and-validation-losses-on-the-same-graph-with-hug
    # TensorBorad log dir
    # logging_dir: str = os.path.join(
    #     PROJECT_ROOT_PATH, 
    #     f"biomlm/outputs/{SPECIES.value}_{TOKEN_MODEL.value}_{VOCAB_SIZE}_{MAX_LEN}_{HIDDENSIZE}/log")
    logging_dir: str = os.path.join(
        PROJECT_ROOT_PATH, 
        f"biomlm/outputs/{SPECIES.value}_{TOKEN_MODEL.value}_{VOCAB_SIZE}_{MAX_LEN}_{HIDDENSIZE}/{OUTPUT_SUBDIR}/log")
    logging_steps: int = 200 #
    logging_strategy: str = "steps"
    # when installed 'tensorboard', then model_config_json = model.config.to_json_string()
    # AttributeError: dict object has no attribute 'to_json_string'
    # https://github.com/huggingface/peft/pull/1200
    # Thus, have to build a Config class that was inheritted PretrainedConfig class
    report_to: str = "tensorboard"
    # If there multiple events in the log_dir, plot will be mass in the tensorboard.
    # Thus, need to delete the previous results. 

    # NOTE: save config
    save_steps: int = 2000 
    save_strategy: str = "steps"
    # If a value is passed, will limit the total amount of checkpoints. 
    # Deletes the older checkpoints in output_dir.
    save_total_limit: int = 3

    #
    # optim: str = "adamw_torch" # default optimizer is adamw_torch
    #
    # The weight decay to apply (if not zero) to all layers 
    # except all bias and LayerNorm weights in AdamW optimizer.
    weight_decay: float = 0.01           # 0.1 or 0.01
    adam_beta1:float = 0.9              # default for AdamW
    adam_beta2:float = 0.999             # default: 0.999 or 0.95
    adam_epsilon:float = 1e-8

    do_train: bool = True
    do_eval: bool = True 
    max_grad_norm:float = 1.0  # lib defult value

    #
    # Number of predictions steps to accumulate the output tensors for, 
    # before moving the results to the CPU. 
    # If left unset, the whole predictions are accumulated on 
    # GPU/NPU/TPU before being moved to the CPU (faster but requires more memory).
    eval_accumulation_steps: int = 100 
    
    # # 
    # # NOTE: best model load config
    # # load_best_model_at_end requires the save and eval strategy to match.
    # load_best_model_at_end: bool = True 
    # # metric_for_best_model: str = None # for using eval loss if None
    # greater_is_better: bool = False
    # # 
    # # # EarlyStoppingCallback requires load_best_model_at_end = True
    # early_stop: bool = True
    # # # config for early stop call back
    # early_stopping_patience: int = 1000
    # early_stopping_threshold: float = 0.0001

    # If True, For the first run, when runing `training_config.main_process_first(desc="dataset map tokenization")` could cause
    # RuntimeError: [3] is setting up NCCL communicator 
    # and retrieving ncclUniqueId from [0] via c10d key-value store by key '0', but store->get('0') got error: Socket Timeout
    # see: https://github.com/stas00/ml-engineering/issues/1
    # https://github.com/huggingface/transformers/issues/15618
    # NOTE 
    sharded_ddp: bool = True   # speed up training under multi-GPU

    # https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.ddp_timeout
    ddp_timeout: int = 60 * 60 * 2 # 2-hour

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

    # -----------------
    # only used for fine tuning
    # 0: not using label smoother, see _trainer.loss.
    label_smoothing_factor: float = 0.0 # or 0.01, 

    #
    # If input does not contained labels, then we need to use this
    # include_inputs_for_metrics: bool = True

    #
    disable_tqdm: bool = True # full train True

    # NOTE
    # BF16 can represent a wider range of integers, but with less precision in the mantissa; 
    # FP16 has a smaller integer range, but with higher mantissa precision.(Thus, fp16 is too prone to overflow.)
    # Thus, Inference is suited for fp16, training is suited for bf16.
    # TF32 uses the same 10-bit mantissa precision as half-precision (FP16) math, which is far beyond the precision requirements for AI workloads, 
    # providing ample margin. At the same time, TF32 uses the same 8-bit exponent as FP32, supporting the same numerical range.
    #
    # bf16: bool = True # v100 not support , residual_in_fp32 not support
    #
    tf32: bool = not USE_MIXED_PRECISION 
    fp16: bool = USE_MIXED_PRECISION

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
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is only for debug
@dataclass
class BioSeqMambaCausalLMTrainingConfigDebug(TrainingArguments):

    # deepspeed = DEEPSPEED_CONFIG # r"/home/share/huadjyin/home/baiyong01/projects/biomlm/biomlm/train/dspeed_config.json"

    output_dir: str = os.path.join(
        PROJECT_ROOT_PATH, 
        f"biomlm/outputs/{SPECIES.value}_{TOKEN_MODEL.value}_{VOCAB_SIZE}_{MAX_LEN}_{HIDDENSIZE}_DEBUG/{OUTPUT_SUBDIR}")
    overwrite_output_dir: bool = True 
    #   
    # NOTE: to complete the whole training data during one epoch, 
    # there needs  steps = (n_whole_training_samples / (per_device_train_batch_size * n_GPU * gradient_accumulation_steps))
    # 1 epoch ~= almost 4533 steps (T2T training, 512 token_seq_len) 
    #
    learning_rate: float = 6e-4   # 6e-4
    # linear: transformers.get_linear_schedule_with_warmup
    # cosine: transformers.get_cosine_schedule_with_warmup
    # cosine_with_restarts: transformers.get_cosine_with_hard_restarts_schedule_with_warmup
    # polynomial: transformers.get_polynomial_decay_schedule_with_warmup
    # constant: transformers.get_constant_schedule
    # constant_with_warmup: transformers.get_constant_schedule_with_warmup
    # inverse_sqrt: transformers.get_inverse_sqrt_schedule
    # https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_utils.py#L402 
    # lr_scheduler_type: str = "constant_with_warmup"
    # lr_scheduler_type: str = "linear"
    lr_scheduler_type: str = "cosine_with_restarts"
    # NOTE `num_steps_per_cycle` used for computing the `num_cycles`  
    num_steps_per_cycle: int = 10  # should be smaller, like 1000, during training to improve accuracy quickly. But larger during fine-tuning.
    ############
    # warmup_ratio:float = 0.05           
    warmup_steps: int = 10   
    # 
    # https://discuss.huggingface.co/t/streaming-dataset-into-trainer-does-not-implement-len-max-steps-has-to-be-specified/32893/7
    max_steps:int = 100  #  50008-512-1024-24: ~100 epoch, ~ 2.8 days to finish training

    dataloader_num_workers: int = 16      
    #############
    gradient_accumulation_steps: int = 2  
    # 24: 50008-512-1024:  GPU: 35813MiB, (vocab_size, max_length, d_model)
    #  4: 50008-512-2048,  GPU: 39641MiB
    # 12: 50008-1024-1024, GPU: 35435MiB
    # 48: 50008-512-768,   GPU: 35991MiB
    # 48/56: 50008-512-512,   GPU: 30559MiB/35347MiB
    per_device_train_batch_size: int = BATCH_SIZE[VOCAB_SIZE][MAX_LEN][HIDDENSIZE] 
    per_device_eval_batch_size: int = BATCH_SIZE[VOCAB_SIZE][MAX_LEN][HIDDENSIZE]

    # NOTE: config for evaluation, NOT take too long
    evaluation_strategy: str = "steps"  # "epoch" would wait for long time to have output next epoch
    #
    # if evaluation_strategy="steps". eval_steps will default to the same 
    # value as logging_steps if not set.
    # eval_steps must be an integer if bigger than 1
    eval_steps: int = 2
    
    # NOTE: logging config 
    # https://stackoverflow.com/questions/73281901/is-there-a-way-to-plot-training-and-validation-losses-on-the-same-graph-with-hug
    # TensorBorad log dir
    logging_dir: str = os.path.join(
        PROJECT_ROOT_PATH, 
        f"biomlm/outputs/{SPECIES.value}_{TOKEN_MODEL.value}_{VOCAB_SIZE}_{MAX_LEN}_{HIDDENSIZE}_DEBUG/{OUTPUT_SUBDIR}/log")
    logging_steps: int = 2 #
    logging_strategy: str = "steps"
    # when installed 'tensorboard', then model_config_json = model.config.to_json_string()
    # AttributeError: dict object has no attribute 'to_json_string'
    # https://github.com/huggingface/peft/pull/1200
    # Thus, have to build a Config class that was inheritted PretrainedConfig class
    report_to: str = "tensorboard"
    # If there multiple events in the log_dir, plot will be mass in the tensorboard.
    # Thus, need to delete the previous results. 

    # NOTE: save config
    save_steps: int = 2 
    save_strategy: str = "steps"
    # If a value is passed, will limit the total amount of checkpoints. 
    # Deletes the older checkpoints in output_dir.
    save_total_limit: int = 3

    #
    # optim: str = "adamw_torch" # default optimizer is adamw_torch
    #
    # The weight decay to apply (if not zero) to all layers 
    # except all bias and LayerNorm weights in AdamW optimizer.
    weight_decay: float = 0.01          # 0.1 or 0.01
    adam_beta1:float = 0.9              # default for AdamW
    adam_beta2:float = 0.999            # default: 0.999 or 0.95
    adam_epsilon:float = 1e-8

    do_train: bool = True
    do_eval: bool = True 
    max_grad_norm:float = 1.0  # lib defult value

    #
    # Number of predictions steps to accumulate the output tensors for, 
    # before moving the results to the CPU. 
    # If left unset, the whole predictions are accumulated on 
    # GPU/NPU/TPU before being moved to the CPU (faster but requires more memory).
    eval_accumulation_steps: int = 2 
    
    # # 
    # # NOTE: best model load config
    # # load_best_model_at_end requires the save and eval strategy to match.
    # load_best_model_at_end: bool = True 
    # # metric_for_best_model: str = None # for using eval loss if None
    # greater_is_better: bool = False
    # # 
    # # # EarlyStoppingCallback requires load_best_model_at_end = True
    # early_stop: bool = True
    # # # config for early stop call back
    # early_stopping_patience: int = 1000
    # early_stopping_threshold: float = 0.0001

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

    # -----------------
    # only used for fine tuning
    # 0: not using label smoother, see _trainer.loss.
    label_smoothing_factor: float = 0.0 # or 0.01, 

    #
    # If input does not contained labels, then we need to use this
    # include_inputs_for_metrics: bool = True

    #
    disable_tqdm: bool = True # full train True

    # NOTE
    # BF16 can represent a wider range of integers, but with less precision in the mantissa; 
    # FP16 has a smaller integer range, but with higher mantissa precision.(Thus, fp16 is too prone to overflow.)
    # Thus, Inference is suited for fp16, training is suited for bf16.
    # TF32 uses the same 10-bit mantissa precision as half-precision (FP16) math, which is far beyond the precision requirements for AI workloads, 
    # providing ample margin. At the same time, TF32 uses the same 8-bit exponent as FP32, supporting the same numerical range.
    #
    # bf16: bool = True # v100 not support , residual_in_fp32 not support
    #
    tf32: bool = True 

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
        default=200,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    
    max_eval_samples: Optional[int] = field(
        default=200,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )


