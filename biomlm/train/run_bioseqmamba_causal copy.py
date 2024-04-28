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
import sys
import os
import math
from pathlib import Path

# import this FIRST, before anything from the pytorch or transformer libraries.
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import datasets
import evaluate
import transformers
from datasets import load_dataset, IterableDatasetDict
from transformers import (
    HfArgumentParser,
    EvalPrediction,
    EarlyStoppingCallback,
    DataCollatorForLanguageModeling,
    set_seed
)

from transformers.utils import check_min_version
from transformers.trainer_utils import get_last_checkpoint

sys.path.append(
    str(Path(__file__).resolve().parents[1]) #
) 
from models.model import (
    BioSeqMambaModelConfig, 
    BioSeqMambaTrainingConfig,
    BioSeqDataSetConfig,
    BioSeqTokenizationConfig,
    BioSeqForCausalLM,
    BioSeqMambaCausalLMTrainer,
    BioSeqDataCollator
)
from models.tokenizer import (
    BioSeqBPETokenizer,
    BioSeqUnigramTokenizer
)
from models.utils import model_size

# current version(2024/03/13): 
# transformers=4.38.2
# datasets=2.16.1
# torch=2.0.0
# mamba_ssm=1.1.4
# triton=2.1.0
# causal_conv1d=1.1.3.post1

check_min_version("4.38.2")
logger = logging.getLogger(__name__)

def main():
    parser = HfArgumentParser((
        BioSeqMambaModelConfig, 
        BioSeqMambaTrainingConfig, 
        BioSeqDataSetConfig, 
        BioSeqTokenizationConfig
    ))
    model_config, training_config, dataset_config, tokenization_config = parser.parse_args_into_dataclasses()

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
    dataset = load_dataset(
        dataset_config.local_script,  
        chunk_len=dataset_config.dataset_chunk_len, 
        overlap=dataset_config.dataset_chunk_overlap, 
        data_files=dataset_config.raw_data_files,
        data_dir=dataset_config.raw_data_local_dir,
        cache_dir=dataset_config.origin_dataset_cache_dir,
        trust_remote_code=True,
    )

    logger.info(dataset)
    # DatasetDict({
    #     train: Dataset({
    #         features: ['sequence', 'chromosome', 'start_pos', 'end_pos'],
    #         num_rows: 21
    #     })
    #     validation: Dataset({
    #         features: ['sequence', 'chromosome', 'start_pos', 'end_pos'],
    #         num_rows: 1
    #     })
    #     test: Dataset({
    #         features: ['sequence', 'chromosome', 'start_pos', 'end_pos'],
    #         num_rows: 1
    #     })
    # })

    # https://huggingface.co/docs/datasets/en/stream
    # dataset = dataset.to_iterable_dataset().shuffle(seed=training_config.data_seed)
    # https://discuss.huggingface.co/t/num-worker-with-iterabledataset/58914/2
    # to avoid WARNING - datasets.iterable_dataset - Too many dataloader workers: 
    # 10 (max is dataset.n_shards=1). Stopping 9 dataloader workers.
    # datasets.iterable_dataset - To parallelize data loading, we give each process some 
    # shards (or data sources) to process. Therefore it's unnecessary to have a number of workers 
    # greater than dataset.n_shards=1. To enable more parallelism, please split the dataset in more files than 1.
    iterable_ds_dict = IterableDatasetDict()
    for ds_name, ds_val in dataset.items():
        iterable_ds_dict[ds_name] = ds_val.to_iterable_dataset(num_shards=len(dataset[ds_name]))

    # logger.info(next(iter(iterable_ds_dict["test"])))
    ####################################################################
    # Load pretrained tokenizer and tokenize dataset
    #
    # In distributed training, the .from_pretrained methods guarantee 
    # that only one local process can concurrently download model & vocab.
    ####################################################################

    logger.info(f"Loading initial tokenizer: {init_token_conifg.token_model_name}")
    logger.info(f"Initial tokenizer location: {init_token_conifg.tokenizer_pretrained_files['vocab_filename'].rsplit('/', 1)[0]}")
    logger.info(f"tokenization_config.model_max_length: {tokenization_config.model_max_length}")
    logger.info(f"training_config.sharded_ddp: {training_config.sharded_ddp}")

    if init_token_conifg.token_model_name == "BPE":
        init_tokenier = BioSeqBPETokenizer.from_file(
            vocab_filename=init_token_conifg.tokenizer_pretrained_files["vocab_filename"],
            merges_filename=init_token_conifg.tokenizer_pretrained_files["merges_filename"],
            unk_token=tokenization_config.unk_token
        )
    if init_token_conifg.token_model_name == "Unigram":
        init_tokenier = BioSeqUnigramTokenizer.from_file(
            vocab_filename=init_token_conifg.tokenizer_pretrained_files["vocab_filename"],
            unk_token=tokenization_config.unk_token
        )
    pretrained_tokenier = BioSeqTokenizer(
        tokenizer_object=init_tokenier,
        bos_token=tokenization_config.bos_token,
        eos_token=tokenization_config.eos_token,
        pad_token=tokenization_config.pad_token,
        unk_token=tokenization_config.unk_token,
        mask_token=tokenization_config.mask_token,
        model_max_length=tokenization_config.model_max_length,
        padding_side=tokenization_config.padding_side,
    )
    # pretrained_tokenier = BioSeqTokenizer(
    #     pretrained_model_name_or_path=tokenization_config.token_pretrained_name_or_path,
    #     local_files_only = tokenization_config.token_local_files_only,
    #     model_max_length = tokenization_config.token_model_max_length,
    #     padding_side = tokenization_config.token_padding_side,
    #     cache_dir = dataset_config.dataset_cache_dir,
    # )
    with training_config.main_process_first(desc="dataset map tokenization"):
        clm_tokenized_ds = pretrained_tokenier(
            # dataset, 
            iterable_ds_dict,
            max_length=tokenization_config.model_max_length, 
            truncation= tokenization_config.truncation,
            padding=tokenization_config.padding,
            return_overflowing_tokens=tokenization_config.return_overflowing_tokens, 
            stride=tokenization_config.stride, 
            min_ctx_fraction=tokenization_config.token_min_ctx_fraction,
            num_proc = tokenization_config.token_num_proc,
            load_from_cache_file = not tokenization_config.overwrite_cache,
            # remove_columns=dataset["train"].column_names
            remove_columns=iterable_ds_dict["train"].column_names
        )
    
    # logger.info(list(clm_tokenized_ds["test"].take(2)))

    # Vanilla Mamba do not need positional encoding
    # See: https://github.com/state-spaces/mamba/issues/51

    ####################################################################
    # Configure the model trainer
    #
    ####################################################################
    ######
    # https://discuss.huggingface.co/t/how-to-use-huggingface-trainer-streaming-datasets-without-wrapping-it-with-torchdatas-iterablewrapper/25230
    if training_config.do_train:
        if "train" not in clm_tokenized_ds:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = clm_tokenized_ds["train"].with_format("torch")

        if training_config.max_train_samples is not None:
            # max_train_samples = min(len(train_dataset), training_config.max_train_samples)
            train_dataset = train_dataset.take(max_train_samples)
    
    if training_config.do_eval:
        if "validation" not in clm_tokenized_ds:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = clm_tokenized_ds["validation"].with_format("torch")
        if training_config.max_eval_samples is not None:
            # max_eval_samples = min(len(eval_dataset), training_config.max_eval_samples)
            # eval_dataset = eval_dataset.select(range(max_eval_samples))
            eval_dataset = eval_dataset.take(max_eval_samples)
        
        def preprocess_logits_for_metrics(logits, labels):

            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            # print(f"metrics_labels:{labels.shape}")
            # print(f"metrics_logits:{logits.shape}")
            # NOTE: important:
            # https://discuss.huggingface.co/t/evalprediction-returning-one-less-prediction-than-label-id-for-each-batch/6958/6
            # https://github.com/huggingface/transformers/blob/198c335d219a5eb4d3f124fdd1ce1a9cd9f78a9b/src/transformers/trainer.py#L2647
            #
            # if isinstance(outputs, dict):
            #     logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
            # else:
            #     logits = outputs[1:]
            #
            # NOTE: This will discard the first element if your outputs is not a dictionary. 
            # My original outputs is a tensor and I wrap it to a dictionary to solve the question.
            #
                   
            return logits.argmax(dim=-1)
        
        # vanila evaluate module needs to connect to huggingface.co
        # thus have to download metric-modules from github:
        # https://github.com/huggingface/evaluate
        # solution: https://github.com/huggingface/evaluate/issues/456
        # 
        # metric = evaluate.load("accuracy", cache_dir=training_config.output_dir)
        metric = evaluate.load("accuracy")

        def compute_metrics(eval_preds: EvalPrediction):
            # for evaluation
            preds, labels = eval_preds

            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)
        
    # Initialize model
    model_args ={
        "d_model": model_config.d_model,
        "n_layer": model_config.n_layer,
        "vocab_size": model_config.vocab_size,
        "rms_norm": model_config.rms_norm,
        "residual_in_fp32": model_config.residual_in_fp32,
        "fused_add_norm": model_config.fused_add_norm,
        "pad_vocab_size_multiple": model_config.pad_vocab_size_multiple,
        "norm_epsilon": model_config.norm_epsilon,
        "tie_embeddings": model_config.tie_embeddings,

        "ssm_cfg": {
            "d_state": model_config.d_state,
            "expand": model_config.expand,             # 
            "dt_rank": model_config.dt_rank,            # 
            "d_conv": model_config.d_conv,                  # Local convolution width
            "dt_min": model_config.dt_min,
            "dt_max": model_config.dt_max,
            "dt_init": model_config.dt_init,           # str = 'random' or `constant`
            "dt_scale": model_config.dt_scale,
            "dt_init_floor": model_config.dt_init_floor, 
            "conv_bias": model_config.conv_bias, 
            "bias": model_config.bias,
            "use_fast_path": model_config.use_fast_path,        # Fused kernel options
        },

        "initializer_cfg": {
            "initializer_range": model_config.initializer_range,
            "rescale_prenorm_residual": model_config.rescale_prenorm_residual,
            "n_residuals_per_layer": model_config.n_residuals_per_layer,
        },
    } 
    
    # https://pytorch.org/docs/stable/notes/cuda.html
    logger.info(f"^^^^^^^^tf32 is set: {torch.backends.cuda.matmul.allow_tf32}")
    clm_model = BioSeqForCausalLM(
        device=training_config.device, 
        dtype=torch.bfloat16 if training_config.bf16 else torch.float32,
        **model_args,
    )
    clm_model.config.use_cache = False

    logger.info(f"model size: {model_size(clm_model)/1000**2:.1f}M parameters")
    
    # data_collator
    data_collator = None
    if training_config.keep_original_pad_token_ids:
        data_collator  = BioSeqDataCollator(
            pretrained_tokenier.tokenizer, 
            mlm=False, 
            replaced_padding_id=pretrained_tokenier.tokenizer.pad_token_id
        )
    else:
        # this will replace padding it with -100, so that pad will be ignored by CrossEntropy loss
        data_collator  = DataCollatorForLanguageModeling(
            pretrained_tokenier.tokenizer, 
            mlm=False, 
        )

    # Initialize Trainer
    # Some tensors share memory, this will lead to duplicate memory on disk and potential 
    # differences when loading them again: [{'lm_head.weight', 'backbone.embedding.weight'}].
    # A potential way to correctly save your model is to use `save_model`.
    # More information at https://huggingface.co/docs/safetensors/torch_shared_tensors

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=training_config.early_stopping_patience, 
        early_stopping_threshold=training_config.early_stopping_threshold)
    
    trainer = BioSeqMambaCausalLMTrainer(
        model=clm_model,
        args=training_config,
        train_dataset=train_dataset if training_config.do_train else None,
        eval_dataset=eval_dataset if training_config.do_eval else None,
        tokenizer=pretrained_tokenier,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_config.do_eval else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_config.do_eval else None,
        # Optimizers Will default to an instance of AdamW on your model and 
        # a scheduler given by get_linear_schedule_with_warmup() controlled by args.
        callbacks= [early_stopping, ] if training_config.early_stop else None,
    )

    ####################################################################
    # Training and evaluation
    #
    ####################################################################
    logger.info(">>>>>>>>>>>>>>>>Start training and evaluatoin......")

    # Training 
    if training_config.do_train:
        checkpoint = None
        if training_config.resume_from_checkpoint is not None:
            checkpoint = training_config.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        trainer.save_model(training_config.output_dir)  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            training_config.max_train_samples if training_config.max_train_samples is not None else len(train_dataset)
        )

        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_config.do_eval:
        metrics = trainer.evaluate()

        max_eval_samples = training_config.max_eval_samples if training_config.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


    # logger.info("<<<<<<<<<<<<<<<<Done")

    
if __name__ == "__main__":
    main()

