#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		run_bioseqmamba_causal.py
@Time    :   	2024/03/13 22:22:00
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

@Desc    :   	

Adapted from:
https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py

"""

import logging
import warnings
import math
from pathlib import Path
import sys
import os
import gc

# import this FIRST, before anything from the pytorch or transformer libraries.
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#
# Why appear the following warning?
# ------------------------
# huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
# To disable this warning, you can either:
#   - Avoid using `tokenizers` before the fork if possible
#   - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
# ------------------------
# Caused by fast tokenizer in streaming mode
# See: https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import copy
import torch
import datasets
import evaluate
import transformers
from datasets import Dataset, DatasetDict, IterableDatasetDict, load_from_disk, concatenate_datasets
from transformers import (
    HfArgumentParser,
    set_seed
)

from transformers.utils import check_min_version
from transformers.trainer_utils import get_last_checkpoint

sys.path.append(
    str(Path(__file__).resolve().parents[1]) #
) 

from models.model import BioSeqForCausalLM
from models.trainer import (
    BioSeqMambaModelConfig, 
    BioSeqMambaCausalLMTrainingConfig,
    BioSeqDataSetConfig,
    BioSeqTokenizationConfig,
    BioSeqMambaCausalLMTrainer,
    BioSeqDataCollatorCausalLM,
    TokenModel,

    BioSeqMambaCausalLMTrainingConfigDebug,
)
from models.tokenizer import (
    BioSeqBPETokenizer,
    BioSeqUnigramTokenizer,
    BioSeqBPETokenizerFast,
    BioSeqUnigramTokenizerFast,
    BioSeqSPMTokenizer,
    BioSeqSPMTokenizerFast,
    BioSeqTokenizerMap
)
from models.utils import model_size

# current version(2024/03/13): 
# transformers=4.39.3 (2024/04/12)  # 4.38.2 will have issue: https://github.com/huggingface/transformers/issues/28119
# datasets=2.16.1
# torch=2.0.0
# mamba_ssm=1.1.4
# triton=2.1.0
# causal_conv1d=1.1.3.post1
# tokenizer:0.15.1

check_min_version("4.39.3")
logger = logging.getLogger(__name__)

def main():

    debug = False

    parser = HfArgumentParser((
        BioSeqMambaCausalLMTrainingConfigDebug, 
        BioSeqDataSetConfig, 
        BioSeqTokenizationConfig
    ))
    training_config, dataset_config, tokenization_config = parser.parse_args_into_dataclasses()

    if training_config.tf32:
        training_config.tf32 = False
    if training_config.disable_tqdm:
        training_config.disable_tqdm = False
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_config.should_log:
        # The default of training_args.log_level is passive, 
        # so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_config.get_process_log_level() # log_level = 20
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_config.local_rank}, device: {training_config.device}, n_gpu: {training_config.n_gpu}, "
        + f"distributed training: {training_config.parallel_mode.value == 'distributed'}, 16-bits training: {training_config.fp16}"
    )
    # logger.info(f"Training/evaluation parameters {training_config}")
    # logger.info(f"Model parameters {model_config}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_config.output_dir) and training_config.do_train and not training_config.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_config.output_dir)
        if last_checkpoint is None and len(os.listdir(training_config.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_config.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_config.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_config.seed)
    
    ####################################################################
    # Load dataset
    #
    # In distributed training, the load_dataset function guarantee that 
    # only one local process can concurrently work for dataset.
    # https://github.com/NVIDIA/nccl/issues/708#issuecomment-1626424234
    # https://huggingface.co/docs/datasets/en/stream
    ####################################################################
    logger.info(f"Start generate dataset for {dataset_config.dataset_name}")

    use_streaming = dataset_config.use_streaming

    logger.info(f"use_streaming: {use_streaming}")
    
    # load raw datasets
    raw_dataset_dirs:list = dataset_config.raw_dataset_dirs
    logger.info(f"raw datasets dir: {raw_dataset_dirs}")

    raw_dataset_dict_list = []
    for _raw_dataset_dir in raw_dataset_dirs:
        raw_dataset_dict_list.append(load_from_disk(_raw_dataset_dir))
    
    raw_dataset_dict = DatasetDict()
    for k in raw_dataset_dict_list[0].keys():
        raw_dataset_dict[k] = concatenate_datasets([i_dd[k] for i_dd in raw_dataset_dict_list])

    del raw_dataset_dict_list
    gc.collect()

    if use_streaming:
        # This will take a while (15 mintes)
        raw_datasetd = IterableDatasetDict()
        for ds_name, ds_val in raw_dataset_dict.items():
            raw_datasetd[ds_name] = ds_val.to_iterable_dataset(num_shards=len(raw_dataset_dict[ds_name]))
    else:
        raw_datasetd = copy.deepcopy(raw_dataset_dict)

    del raw_dataset_dict
    gc.collect()

    if debug:
        for raw_name_key in raw_datasetd:
            if not use_streaming:
                raw_datasetd[raw_name_key] = raw_datasetd[raw_name_key].select(range(2))
            else:
                raw_datasetd[raw_name_key] = raw_datasetd[raw_name_key].take(2)
    
    logger.info(raw_datasetd)
    # >>> (not streaming):
    # DatasetDict({
    #     train: Dataset({
    #         features: ['sequence', 'chromosome', 'start_pos', 'end_pos'],
    #         num_rows: 45
    #     })
    #     test: Dataset({
    #         features: ['sequence', 'chromosome', 'start_pos', 'end_pos'],
    #         num_rows: 2
    #     })
    # })
    #
    # >>> (streaming):
    # IterableDatasetDict({
    #     train: IterableDataset({
    #         features: ['sequence', 'chromosome', 'start_pos', 'end_pos'],
    #         n_shards: 45
    #     })
    #     test: IterableDataset({
    #         features: ['sequence', 'chromosome', 'start_pos', 'end_pos'],
    #         n_shards: 2
    #     })
    # })

    ####################################################################
    # Load pretrained tokenizer and tokenize dataset
    #
    # In distributed training, the .from_pretrained methods guarantee 
    # that only one local process can concurrently download model & vocab.
    # 
    ####################################################################
    logger.info(f"Tokenizer model name: {tokenization_config.token_model_type.value}")
    logger.info(f"Fast tokenizer: {tokenization_config.use_fast_tokenizer}")
    logger.info(f"Pretrained tokenizer loading from: {tokenization_config.tokenizer_pretrained_dir}")
    logger.info(f"tokenization_config.model_max_length: {tokenization_config.model_max_length}")

    with training_config.main_process_first(desc="dataset map tokenization"):
        tokenizer_class = None
        if not tokenization_config.use_fast_tokenizer:
            if tokenization_config.token_model_type == TokenModel.BPE:
                tokenizer_class = BioSeqBPETokenizer
            elif tokenization_config.token_model_type == TokenModel.UNIGRAM:
                tokenizer_class = BioSeqUnigramTokenizer
            elif tokenization_config.token_model_type in [TokenModel.SPM_BPE, TokenModel.SPM_UNIGRAM]:
                tokenizer_class = BioSeqSPMTokenizer
        else:
            if tokenization_config.token_model_type == TokenModel.BPE:
                tokenizer_class = BioSeqBPETokenizerFast
            elif tokenization_config.token_model_type == TokenModel.UNIGRAM:
                tokenizer_class = BioSeqUnigramTokenizerFast
            elif tokenization_config.token_model_type in [TokenModel.SPM_BPE, TokenModel.SPM_UNIGRAM]:
                tokenizer_class = BioSeqSPMTokenizerFast
        
        if tokenizer_class is None:
            raise ValueError(f"{tokenizer_class} cannot be None")
        
        tokenizer_init_kwargs = {
            "bos_token": tokenization_config.bos_token,
            "eos_token": tokenization_config.eos_token,
            "unk_token": tokenization_config.unk_token,
            "model_max_length": tokenization_config.model_max_length,
            "padding_side": tokenization_config.padding_side,
            # if `streaming` mode, we initially do not add <BOS> token or <EOS> token
            # until calling  `get_chunked_tokenized_dataset` function.
            "add_bos_token": tokenization_config.add_bos_token if not use_streaming else False,
            "add_eos_token": tokenization_config.add_eos_token if not use_streaming else False,
            "add_prefix_space": False, 
            "do_lower_case": False,
        }

        # We only support model_input_names = ["input_ids", "attention_mask"]
        tokenizer = tokenizer_class.from_pretrained(
            tokenization_config.tokenizer_pretrained_dir, 
            local_files_only=True, 
            **tokenizer_init_kwargs)
        # add pad token
        tokenizer.pad_token = tokenizer.eos_token
        
        logger.info(f"vocab size load by tokenizer: {tokenizer.vocab_size}")

        # Preprocessing the datasets.
        # First we tokenize all the sequence.
        column_names = list(raw_datasetd["train"].features)
        sequence_column_name = "sequence" if "sequence" in column_names else column_names[0]

        clm_tokenized_ds = BioSeqTokenizerMap(
            tokenizer,
            max_length = tokenization_config.model_max_length, # loss of function when padding and truncation are set to False
            stride = tokenization_config.stride,
            min_len_frac = tokenization_config.token_min_ctx_fraction,
            streaming = use_streaming,  # dataset is too large, so using streaming mode
        ).do_map(
            raw_datasetd,
            dataset_col_remove = column_names,
            dataset_col_tokenize = sequence_column_name,
            padding = tokenization_config.padding,
            truncation=tokenization_config.truncation,
            return_overflowing_tokens = tokenization_config.return_overflowing_tokens,
            load_from_cache_file = not tokenization_config.overwrite_cache,  # not used in streaming mode
            num_proc = tokenization_config.token_num_proc,  # not used in streaming mode
        ).get_chunked_tokenized_dataset(
            add_bos_token = tokenization_config.add_bos_token,
            add_eos_token = tokenization_config.add_eos_token
        )

        logger.info(f"saving the following dataset into {tokenization_config.tokenized_dataset_dir}.\n {clm_tokenized_ds}")
        # >>> (streaming)
        # IterableDatasetDict({
        #     train: IterableDataset({
        #         features: ['input_ids', 'attention_mask'],
        #         n_shards: 45
        #     })
        #     test: IterableDataset({
        #         features: ['input_ids', 'attention_mask'],
        #         n_shards: 2
        #     })
        # })

        if use_streaming:
            # this will take a while. (~6 min per chrosome)
            dd = DatasetDict()
            for ds_name, iterable_ds in clm_tokenized_ds.items():
                ds = Dataset.from_generator(lambda: (yield from iterable_ds), features=iterable_ds.features)
                dd[ds_name] = ds
            dd.save_to_disk(tokenization_config.tokenized_dataset_dir)
            dd.cleanup_cache_files()
        else:
            clm_tokenized_ds.save_to_disk(tokenization_config.tokenized_dataset_dir)
            clm_tokenized_ds.cleanup_cache_files()
    
    
if __name__ == "__main__":
    main()

