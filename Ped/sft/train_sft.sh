#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift sft \
    --model Qwen/Qwen3-8B \
    --tuner_type full \
    --dataset /Users/hjxa/local_code/Pedia_clinical_agent/train/datasets/sft/train_dataset.jsonl \
    --load_from_cache_file true \
    --split_dataset_ratio 0 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 4 \
    --save_steps 100 \
    --save_total_limit 3 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir /Users/hjxa/local_code/Pedia_clinical_agent/train/ms-swift/Ped/sft \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --use_liger_kernel true \
    --deepspeed zero3 \
    --save_only_model true

# 参数说明：
# CUDA_VISIBLE_DEVICES: 指定可见 GPU，这里默认使用 8 张 A100。
# NPROC_PER_NODE: 单机训练进程数，通常等于使用的 GPU 数。
# swift sft: 启动 ms-swift 的监督微调入口。
# --model: 基座模型路径或模型 ID，这里使用 Qwen/Qwen3-8B。
# --tuner_type full: 使用全参数微调；不会创建 LoRA adapter。
# --dataset: SFT 数据集路径，格式为 ms-swift messages JSONL。
# --load_from_cache_file: 是否复用数据集预处理缓存。
# --split_dataset_ratio: 验证集切分比例；当前样本少，设为 0 不切分。
# --torch_dtype: 模型训练精度；bfloat16 适合 A100/H100。
# --num_train_epochs: 训练轮数。
# --per_device_train_batch_size: 每张 GPU 的训练 batch size。
# --per_device_eval_batch_size: 每张 GPU 的评估 batch size。
# --learning_rate: 全参 SFT 学习率，通常低于 LoRA SFT。
# --gradient_accumulation_steps: 梯度累计步数，用于扩大有效 batch。
# --save_steps: 每隔多少 step 保存 checkpoint。
# --save_total_limit: 最多保留 checkpoint 数量。
# --logging_steps: 每隔多少 step 打印训练日志。
# --max_length: 训练样本最大 token 长度。
# --output_dir: 全参 SFT checkpoint 输出目录；GRPO 阶段从这里的 checkpoint-xxx 继续。
# --warmup_ratio: 学习率 warmup 占总训练步数比例。
# --dataloader_num_workers: DataLoader worker 数量。
# --dataset_num_proc: 数据预处理并行进程数。
# --use_liger_kernel: 启用 Liger Kernel 节省显存并提升吞吐。
# --deepspeed zero3: 使用 DeepSpeed ZeRO-3 做全参多卡显存优化。
# --save_only_model: 只保存模型权重，不保存优化器等训练状态，节省磁盘空间。
