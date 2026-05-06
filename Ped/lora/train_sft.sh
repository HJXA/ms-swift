#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift sft \
    --model Qwen/Qwen3-8B \
    --tuner_type lora \
    --dataset /Users/hjxa/local_code/Pedia_clinical_agent/train/datasets/sft/train_dataset.jsonl \
    --load_from_cache_file true \
    --split_dataset_ratio 0 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 2 \
    --save_steps 100 \
    --save_total_limit 3 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir /Users/hjxa/local_code/Pedia_clinical_agent/train/ms-swift/Ped/output/sft \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --use_liger_kernel true \
    --save_only_model true
