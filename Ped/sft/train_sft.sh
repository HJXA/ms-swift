#!/usr/bin/env bash
set -euo pipefail

export PATH="/ruilab/jxhe/miniconda3/envs/swift/bin:$PATH"
export NPROC_PER_NODE=1
export CUDA_VISIBLE_DEVICES=2
export NCCL_P2P_LEVEL=NVL
export HF_ENDPOINT=https://hf-mirror.com

swift sft \
    --model Qwen/Qwen3-8B \
    --tuner_type full \
    --dataset xxx \
    --load_from_cache_file true \
    --split_dataset_ratio 0 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --max_steps 20 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --gradient_checkpointing true \
    --weight_decay 0 \
    --save_steps 100 \
    --logging_steps 1 \
    --max_length 4096 \
    --truncation_strategy right \
    --attn_impl flash_attention_2 \
    --output_dir xxx \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --dataset_shuffle true \
    --train_dataloader_shuffle true \
    --use_liger_kernel true \
    --deepspeed zero2 \
    --save_only_model true \
    --loss_scale ignore_empty_think \
    --report_to swanlab \
    --swanlab_project Ped \
    --swanlab_token WODn49OiskSyv0qBnFZcL

# 参数说明：
# PATH: 指定优先使用的 swift Conda 环境。
# NPROC_PER_NODE: 单机训练进程数；当前脚本按单卡运行。
# CUDA_VISIBLE_DEVICES: 指定可见 GPU；当前使用物理 2 号卡。
# NCCL_P2P_LEVEL: NCCL P2P 通信级别；NVL 优先使用 NVLink。
# HF_ENDPOINT: HuggingFace 镜像地址。
# swift sft: 启动 ms-swift 的监督微调入口。
# --model: 基座模型路径或模型 ID，这里使用 Qwen/Qwen3-8B。
# --use_hf: 从 HuggingFace/HF 镜像侧解析模型 ID。
# --tuner_type full: 使用全参数微调；不会创建 LoRA adapter。
# --dataset: 使用 sft_thinking 数据；assistant 标准答案前有空 <think> 块。
# --load_from_cache_file: 是否复用数据集预处理缓存。
# --split_dataset_ratio: 验证集切分比例；当前样本少，设为 0 不切分。
# --torch_dtype: 模型训练精度；bfloat16 适合 A100/H100。
# --num_train_epochs: 训练轮数。
# --per_device_train_batch_size: 每张 GPU 的训练 batch size。
# --per_device_eval_batch_size: 每张 GPU 的评估 batch size。
# --learning_rate: 全参 SFT 学习率，通常低于 LoRA SFT。
# --lr_scheduler_type: 学习率调度器；cosine 表示余弦退火。
# --warmup_ratio: 学习率 warmup 占总训练步数比例。
# --gradient_accumulation_steps: 梯度累计步数，用于扩大有效 batch。
# --gradient_checkpointing: 启用梯度检查点，用计算换显存。
# --weight_decay: 权重衰减；0.0 表示不额外正则。
# --save_steps: 每隔多少 step 保存 checkpoint。
# --save_total_limit: 最多保留 checkpoint 数量。
# --logging_steps: 每隔多少 step 打印训练日志。
# --max_length: 训练样本最大 token 长度。
# --truncation_strategy: 超长样本截断策略；right 表示从右侧截断。
# --attn_impl: attention 实现；flash_attention_2 用于提升速度并节省显存。
# --output_dir: 全参 SFT checkpoint 输出目录；GRPO 阶段从这里的 checkpoint-xxx 继续。
# --dataloader_num_workers: DataLoader worker 数量。
# --dataset_num_proc: 数据预处理并行进程数。
# --dataset_shuffle: 训练前打乱数据集。
# --train_dataloader_shuffle: DataLoader 训练阶段打乱样本。
# --use_liger_kernel: 启用 Liger Kernel 节省显存并提升吞吐。
# --deepspeed zero2: 使用 DeepSpeed ZeRO-2 做显存优化。
# --save_only_model: 只保存模型权重，不保存优化器等训练状态，节省磁盘空间。
# --loss_scale ignore_empty_think: 忽略空 <think> 块损失，避免监督空 thinking 本身。
# --report_to swanlab: 训练指标上报 SwanLab；token 请通过登录或环境变量配置，不写入脚本。
# --swanlab_project: SwanLab 项目名。
