#!/usr/bin/env bash
set -euo pipefail

export NPROC_PER_NODE=1
export CUDA_VISIBLE_DEVICES=2
export NCCL_P2P_LEVEL=NVL
export HF_ENDPOINT=https://hf-mirror.com

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3-8B \
    --use_hf true \
    --adapters /Users/hjxa/local_code/Pedia_clinical_agent/train/ms-swift/Ped/lora/sft/checkpoint-xxx \
    --ref_adapters /Users/hjxa/local_code/Pedia_clinical_agent/train/ms-swift/Ped/lora/sft/checkpoint-xxx \
    --external_plugins /Users/hjxa/local_code/Pedia_clinical_agent/train/ms-swift/Ped/ped_reward.py \
    --reward_funcs ped_diagnosis_match \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.4 \
    --vllm_tensor_parallel_size 1 \
    --vllm_max_model_len 8192 \
    --vllm_enable_lora true \
    --vllm_max_lora_rank 8 \
    --sleep_level 1 \
    --offload_model true \
    --offload_optimizer true \
    --tuner_type lora \
    --torch_dtype bfloat16 \
    --dataset /Users/hjxa/local_code/Pedia_clinical_agent/train/datasets/grpo/train_dataset.jsonl \
    --load_from_cache_file true \
    --split_dataset_ratio 0 \
    --max_length 4096 \
    --max_completion_length 1024 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --learning_rate 1e-6 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing true \
    --weight_decay 0.0 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --save_steps 50 \
    --save_total_limit 3 \
    --logging_steps 1 \
    --output_dir /Users/hjxa/local_code/Pedia_clinical_agent/train/ms-swift/Ped/lora/grpo \
    --truncation_strategy right \
    --attn_impl flash_attention_2 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --dataset_shuffle true \
    --train_dataloader_shuffle true \
    --num_generations 2 \
    --temperature 1.0 \
    --top_p 0.85 \
    --top_k 50 \
    --deepspeed zero2 \
    --use_liger_kernel true \
    --log_completions true \
    --num_iterations 1 \
    --beta 0.04 \
    --save_only_model true \
    --report_to swanlab \
    --swanlab_project Ped

# 参数说明：
# PATH: 指定优先使用的 swift Conda 环境。
# NPROC_PER_NODE: 单机启动的训练进程数；当前脚本按单卡运行。
# CUDA_VISIBLE_DEVICES: 指定可见 GPU；当前使用物理 2 号卡。
# NCCL_P2P_LEVEL: NCCL P2P 通信级别；NVL 优先使用 NVLink。
# HF_ENDPOINT: HuggingFace 镜像地址。
# PYTORCH_CUDA_ALLOC_CONF: PyTorch CUDA 显存分配策略；expandable_segments 可缓解显存碎片问题。
# swift rlhf: 启动 ms-swift 的 RLHF/GRPO 训练入口。
# --rlhf_type grpo: 使用 GRPO 算法训练。
# --model: 基座模型路径或模型 ID；LoRA GRPO 仍从 Qwen/Qwen3-8B 加载基座。
# --use_hf: 从 HuggingFace/HF 镜像侧解析模型 ID。
# --adapters: 待继续训练的 LoRA adapter 路径；先把 checkpoint-xxx 替换为 Ped/lora/sft 下实际 checkpoint。
# --ref_adapters: 参考策略使用的 LoRA adapter 路径；从 SFT 继续 GRPO 时通常与 --adapters 相同。
# --external_plugins: 外部插件路径，用于注册自定义 reward function。
# --reward_funcs: 使用的 reward function 名称；ped_diagnosis_match 在 ped_reward.py 中注册。
# --use_vllm: 使用 vLLM 加速 GRPO rollout 采样。
# --vllm_mode: vLLM 部署模式；colocate 表示训练和采样服务共用同一张 GPU。
# --vllm_gpu_memory_utilization: vLLM 可占用的单卡显存比例，降低可减少 OOM 风险。
# --vllm_tensor_parallel_size: vLLM 推理张量并行大小；单卡设置为 1。
# --vllm_max_model_len: vLLM 推理侧最大上下文长度。
# --vllm_enable_lora: vLLM 侧启用 LoRA adapter，同步 LoRA 权重而非全量权重。
# --vllm_max_lora_rank: vLLM 支持的最大 LoRA rank，应大于等于训练的 --lora_rank。
# --sleep_level: rollout 和训练阶段切换时释放 vLLM 显存的级别，1 是常用显存保护设置。
# --offload_model: rollout 阶段将训练模型 offload，降低与 vLLM 共置时的显存占用。
# --offload_optimizer: rollout 阶段将优化器状态 offload，降低显存占用。
# --tuner_type lora: 使用 LoRA GRPO；当前策略只更新 adapter。
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
# --lora_rank: LoRA 低秩矩阵 rank，需与 vLLM max_lora_rank 匹配。
# --lora_alpha: LoRA 缩放系数，常用 2-4 倍 rank。
# --target_modules: LoRA 注入模块；all-linear 表示注入所有线性层。
# --save_steps: 每隔多少 step 保存 checkpoint。
# --save_total_limit: 最多保留 checkpoint 数量。
# --logging_steps: 每隔多少 step 打印训练日志。
# --output_dir: LoRA GRPO adapter 输出目录。
# --truncation_strategy: 超长 prompt 截断策略；right 表示从右侧截断。
# --attn_impl: attention 实现；flash_attention_2 用于提升速度并节省显存。
# --dataloader_num_workers: DataLoader worker 数量。
# --dataset_num_proc: 数据预处理并行进程数。
# --dataset_shuffle: 训练前打乱数据集。
# --train_dataloader_shuffle: DataLoader 训练阶段打乱样本。
# --num_generations: 每个 prompt 采样的 completion 数；单卡 batch 2 时设为 2，满足整除要求。
# --temperature: rollout 采样温度，越高生成越多样。
# --top_p: nucleus sampling 截断阈值。
# --top_k: top-k sampling 截断数量。
# --deepspeed zero2: 使用 DeepSpeed ZeRO-2 做显存优化。
# --use_liger_kernel: 启用 Liger Kernel 节省显存并提升吞吐。
# --log_completions: 保存/记录 rollout 生成结果和 reward，便于调试 reward。
# --num_iterations: 每批 rollout 数据重复更新次数；1 表示更接近 on-policy。
# --beta: KL 惩罚系数；LoRA GRPO 中 reference 加载 --ref_adapters 指向的冻结 SFT adapter。
# --save_only_model: 只保存模型权重，不保存优化器等训练状态，节省磁盘空间。
# --report_to swanlab: 训练指标上报 SwanLab；token 请通过登录或环境变量配置，不写入脚本。
# --swanlab_project: SwanLab 项目名。
