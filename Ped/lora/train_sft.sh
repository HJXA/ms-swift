#!/usr/bin/env bash
set -euo pipefail

export PATH="/ruilab/jxhe/miniconda3/envs/swift/bin:$PATH"
export NPROC_PER_NODE=1
export CUDA_VISIBLE_DEVICES=2
export NCCL_P2P_LEVEL=NVL
export HF_ENDPOINT=https://hf-mirror.com

swift sft \
    --model Qwen/Qwen3-8B \
    --tuner_type lora \
    --dataset /Users/hjxa/local_code/Pedia_clinical_agent/train/datasets/sft_thinking/train_dataset.jsonl \
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
    --weight_decay 0.0 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --save_steps 100 \
    --logging_steps 1 \
    --max_length 4096 \
    --truncation_strategy right \
    --attn_impl flash_attention_2 \
    --output_dir /Users/hjxa/local_code/Pedia_clinical_agent/train/ms-swift/Ped/lora/sft \
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
# --tuner_type lora: 使用 LoRA 参数高效微调；只训练 adapter。
# --dataset: 使用 sft_thinking 数据；assistant 标准答案前有空 <think> 块。
# --load_from_cache_file: 是否复用数据集预处理缓存。
# --split_dataset_ratio: 验证集切分比例；当前样本少，设为 0 不切分。
# --torch_dtype: 模型训练精度；bfloat16 适合 A100/H100。
# --num_train_epochs: 训练轮数。
# --max_steps: 最大训练 step 数；设置后可用于快速 smoke test。
# --per_device_train_batch_size: 每张 GPU 的训练 batch size。
# --per_device_eval_batch_size: 每张 GPU 的评估 batch size。
# --learning_rate: LoRA SFT 学习率；这里按 full SFT 参考脚本设为 1e-5。
# --lr_scheduler_type: 学习率调度器；cosine 表示余弦退火。
# --warmup_ratio: 学习率 warmup 占总训练步数比例。
# --gradient_checkpointing: 启用梯度检查点，用计算换显存。
# --weight_decay: 权重衰减。
# --lora_rank: LoRA 低秩矩阵 rank，越大可训练容量越高。
# --lora_alpha: LoRA 缩放系数，常用 2-4 倍 rank。
# --target_modules: LoRA 注入模块；all-linear 表示注入所有线性层。
# --save_steps: 每隔多少 step 保存 checkpoint。
# --logging_steps: 每隔多少 step 打印训练日志。
# --max_length: 训练样本最大 token 长度。
# --truncation_strategy: 超长样本截断策略；right 表示从右侧截断。
# --attn_impl: attention 实现；flash_attention_2 用于提升速度并节省显存。
# --output_dir: LoRA SFT adapter 输出目录；GRPO 阶段从这里的 checkpoint-xxx 继续。
# --dataloader_num_workers: DataLoader worker 数量。
# --dataset_num_proc: 数据预处理并行进程数。
# --dataset_shuffle: 训练前打乱数据集。
# --train_dataloader_shuffle: DataLoader 训练阶段打乱样本。
# --use_liger_kernel: 启用 Liger Kernel 节省显存并提升吞吐。
# --deepspeed zero2: 使用 DeepSpeed ZeRO-2 做显存优化。
# --save_only_model: 只保存模型权重，不保存优化器等训练状态，节省磁盘空间。
# --loss_scale ignore_empty_think: 忽略空 <think> 块损失，避免监督空 thinking 本身。
# --report_to swanlab: 训练指标上报 SwanLab。
# --swanlab_project: SwanLab 项目名。
# --swanlab_token: SwanLab 访问 token。
