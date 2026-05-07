#!/usr/bin/env bash
set -euo pipefail

export PATH="/ruilab/jxhe/miniconda3/envs/swift/bin:$PATH"
export NPROC_PER_NODE=1
export CUDA_VISIBLE_DEVICES=4
export NCCL_P2P_LEVEL=NVL
export HF_ENDPOINT=https://hf-mirror.com

OUTPUT_DIR=/ruilab/jxhe/Ped/output/lora

# model
# dataset
# num_train_epochs
# max_steps
# per_device_train_batch_size
# lora_rank
# lora_alpha # LoRA 缩放系数，常用 2-4 倍 rank。
# save_steps
# deepspeed

swift sft \
    --model /ruilab2/hjxa/checkpoints/qwen/Qwen3/4B/Qwen3-4B \
    --tuner_type lora \
    --dataset /ruilab/jxhe/Ped/Pedia_clinical_agent/train/datasets/sft_thinking/train_dataset.jsonl \
    --load_from_cache_file true \
    --split_dataset_ratio 0 \
    --torch_dtype bfloat16 \
    --num_train_epochs 5 \
    --max_steps 30 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --gradient_checkpointing true \
    --weight_decay 0.0 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --save_steps 100 \
    --logging_steps 1 \
    --max_length 4096 \
    --truncation_strategy right \
    --attn_impl flash_attention_2 \
    --output_dir $OUTPUT_DIR \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --dataset_shuffle true \
    --train_dataloader_shuffle true \
    --use_liger_kernel true \
    --deepspeed zero2 \
    --save_only_model false \
    --loss_scale ignore_empty_think \
    --report_to swanlab \
    --swanlab_project Ped \
    --swanlab_token WODn49OiskSyv0qBnFZcL


