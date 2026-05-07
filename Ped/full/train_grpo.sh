#!/usr/bin/env bash
set -euo pipefail

export NPROC_PER_NODE=4
export CUDA_VISIBLE_DEVICES=4,5,6,7
export NCCL_P2P_LEVEL=NVL
export HF_ENDPOINT=https://hf-mirror.com

MODEL=/ruilab/jxhe/Ped/output/full/sft/Qwen3_0.6B_sft_thinking/checkpoint-30
OUTPUT_DIR=/ruilab/jxhe/Ped/output/full/grpo/Qwen3_0.6B_sft_thinking_grpo_thinking

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
swift rlhf \
    --rlhf_type grpo \
    --model $MODEL \
    --use_hf true \
    --external_plugins /ruilab/jxhe/Ped/Pedia_clinical_agent/train/ms-swift/Ped/ped_reward.py \
    --reward_funcs ped_diagnosis_match \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.4 \
    --vllm_tensor_parallel_size 1 \
    --vllm_max_model_len 8192 \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --dataset /ruilab/jxhe/Ped/Pedia_clinical_agent/train/datasets/grpo_thinking/train_dataset.jsonl \
    --load_from_cache_file true \
    --split_dataset_ratio 0 \
    --max_length 4096 \
    --max_completion_length 1024 \
    --num_train_epochs 5 \
    --max_steps 30 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-6 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing true \
    --weight_decay 0.0 \
    --save_steps 50 \
    --logging_steps 1 \
    --output_dir $OUTPUT_DIR \
    --truncation_strategy delete \
    --attn_impl flash_attention_2 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --dataset_shuffle true \
    --train_dataloader_shuffle true \
    --generation_batch_size 8 \
    --num_generations 8 \
    --temperature 1.0 \
    --top_p 0.85 \
    --top_k 50 \
    --deepspeed zero2 \
    --use_liger_kernel true \
    --log_completions true \
    --num_iterations 1 \
    --beta 0.04 \
    --save_only_model false \
    --report_to swanlab \
    --swanlab_project Ped \
    --swanlab_token WODn49OiskSyv0qBnFZcL \
    --overlong_filter true

# 参数说明：
# NPROC_PER_NODE: 单机启动的训练进程数；当前脚本按 4 卡运行。
# CUDA_VISIBLE_DEVICES: 指定可见 GPU；当前使用物理 4-7 号卡。
# NCCL_P2P_LEVEL: NCCL P2P 通信级别；NVL 优先使用 NVLink。
# HF_ENDPOINT: HuggingFace 镜像地址。
# PYTORCH_CUDA_ALLOC_CONF: PyTorch CUDA 显存分配策略；expandable_segments 可缓解显存碎片问题。
# swift rlhf: 启动 ms-swift 的 RLHF/GRPO 训练入口。
# --rlhf_type grpo: 使用 GRPO 算法训练。
# MODEL: SFT 全参数 checkpoint 路径；启动前替换为实际 full SFT checkpoint。
# OUTPUT_DIR: 全参数 GRPO checkpoint 输出目录。
# --model: SFT 全参数 checkpoint 路径。
# --use_hf: 从 HuggingFace/HF 兼容路径解析模型。
# --external_plugins: 外部插件路径，用于注册自定义 reward function。
# --reward_funcs: 使用的 reward function 名称；ped_diagnosis_match 在 ped_reward.py 中注册。
# --use_vllm: 使用 vLLM 加速 GRPO rollout 采样。
# --vllm_mode: vLLM 部署模式；colocate 表示训练和采样服务共用 GPU。
# --vllm_gpu_memory_utilization: vLLM 可占用的单卡显存比例，降低可减少 OOM 风险。
# --vllm_tensor_parallel_size: vLLM 推理张量并行大小；单卡设置为 1。
# --vllm_max_model_len: vLLM 推理侧最大上下文长度。
# --sleep_level: rollout 和训练阶段切换时释放 vLLM 显存的级别，1 是常用显存保护设置。
# --offload_model: rollout 阶段将训练模型 offload，降低与 vLLM 共置时的显存占用。
# --offload_optimizer: rollout 阶段将优化器状态 offload，降低显存占用。
# --tuner_type full: 使用全参数 GRPO；当前策略会更新完整模型权重。
# --torch_dtype: 模型训练精度；bfloat16 适合 A100/H100。
# --dataset: GRPO 数据集路径，messages 中不含 assistant，solution 供 reward 使用。
# --load_from_cache_file: 是否复用数据集预处理缓存。
# --split_dataset_ratio: 验证集切分比例；当前样本很少，设为 0 不切分。
# --max_length: prompt 最大 token 长度。
# --max_completion_length: rollout 生成回复的最大 token 长度。
# --num_train_epochs: 训练轮数。
# --per_device_train_batch_size: 每张 GPU 上参与 loss 计算的 completion 数。
# --learning_rate: GRPO 阶段学习率，通常显著小于 SFT。
# --lr_scheduler_type: 学习率调度器；cosine 表示余弦退火。
# --warmup_ratio: 学习率 warmup 占总训练步数比例。
# --gradient_accumulation_steps: 梯度累计步数；单卡 batch 2 时设为 1。
# --gradient_checkpointing: 启用梯度检查点，用计算换显存。
# --weight_decay: 权重衰减；0.0 表示不额外正则。
# --save_steps: 每隔多少 step 保存 checkpoint。
# --logging_steps: 每隔多少 step 打印训练日志。
# --output_dir: 全参数 GRPO checkpoint 输出目录。
# --truncation_strategy: 超长 prompt 截断策略；delete 表示删除超长样本。
# --attn_impl: attention 实现；flash_attention_2 用于提升速度并节省显存。
# --dataloader_num_workers: DataLoader worker 数量。
# --dataset_num_proc: 数据预处理并行进程数。
# --dataset_shuffle: 训练前打乱数据集。
# --train_dataloader_shuffle: DataLoader 训练阶段打乱样本。
# --generation_batch_size: rollout 生成 batch size。
# --num_generations: 每个 prompt 采样的 completion 数；需与 batch 配置满足整除要求。
# --temperature: rollout 采样温度，越高生成越多样。
# --top_p: nucleus sampling 截断阈值。
# --top_k: top-k sampling 截断数量。
# --deepspeed zero2: 使用 DeepSpeed ZeRO-2 做显存优化。
# --use_liger_kernel: 启用 Liger Kernel 节省显存并提升吞吐。
# --log_completions: 保存/记录 rollout 生成结果和 reward，便于调试 reward。
# --num_iterations: 每批 rollout 数据重复更新次数；1 表示更接近 on-policy。
# --beta: KL 惩罚系数；全参 GRPO 未显式设置 ref_model 时，ms-swift 默认使用 --model 作为冻结参考策略。
# --save_only_model: false 表示保存完整训练状态，便于续训。
# --report_to swanlab: 训练指标上报 SwanLab；token 请通过登录或环境变量配置，不写入脚本。
# --swanlab_project: SwanLab 项目名。
# --overlong_filter: 过滤超长样本，避免截断破坏 prompt/answer 结构。
