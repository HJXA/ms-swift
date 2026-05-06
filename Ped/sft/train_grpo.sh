#!/usr/bin/env bash
set -euo pipefail

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift rlhf \
    --rlhf_type grpo \
    --model /Users/hjxa/local_code/Pedia_clinical_agent/train/ms-swift/Ped/sft/checkpoint-xxx \
    --external_plugins /Users/hjxa/local_code/Pedia_clinical_agent/train/ms-swift/Ped/ped_reward.py \
    --reward_funcs ped_diagnosis_match \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.4 \
    --vllm_tensor_parallel_size 1 \
    --vllm_max_model_len 8192 \
    --sleep_level 1 \
    --offload_model true \
    --offload_optimizer true \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --dataset /Users/hjxa/local_code/Pedia_clinical_agent/train/datasets/grpo/train_dataset.jsonl \
    --load_from_cache_file true \
    --split_dataset_ratio 0 \
    --max_length 4096 \
    --max_completion_length 1024 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 1 \
    --save_steps 50 \
    --save_total_limit 3 \
    --logging_steps 1 \
    --output_dir /Users/hjxa/local_code/Pedia_clinical_agent/train/ms-swift/Ped/grpo \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 16 \
    --temperature 1.0 \
    --top_p 0.85 \
    --top_k 50 \
    --deepspeed zero3 \
    --log_completions true \
    --num_iterations 1 \
    --beta 0.04 \
    --save_only_model true

# 参数说明：
# PYTORCH_CUDA_ALLOC_CONF: PyTorch CUDA 显存分配策略；expandable_segments 可缓解显存碎片问题。
# CUDA_VISIBLE_DEVICES: 指定可见 GPU，这里默认使用 8 张 A100。
# NPROC_PER_NODE: 单机启动的训练进程数，通常等于使用的 GPU 数。
# swift rlhf: 启动 ms-swift 的 RLHF/GRPO 训练入口。
# --rlhf_type grpo: 使用 GRPO 算法训练。
# --model: SFT 全参数 checkpoint 路径；先把 checkpoint-xxx 替换为 Ped/sft 下实际 checkpoint。
# --external_plugins: 外部插件路径，用于注册自定义 reward function。
# --reward_funcs: 使用的 reward function 名称；ped_diagnosis_match 在 ped_reward.py 中注册。
# --use_vllm: 使用 vLLM 加速 GRPO rollout 采样。
# --vllm_mode: vLLM 部署模式；colocate 表示训练和采样服务共用同一组 GPU。
# --vllm_gpu_memory_utilization: vLLM 可占用的单卡显存比例，降低可减少 OOM 风险。
# --vllm_tensor_parallel_size: vLLM 推理张量并行大小；Qwen3-8B 在 8 卡训练场景下先用 1。
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
# --gradient_accumulation_steps: 梯度累计步数，用于扩大有效 batch。
# --save_steps: 每隔多少 step 保存 checkpoint。
# --save_total_limit: 最多保留 checkpoint 数量。
# --logging_steps: 每隔多少 step 打印训练日志。
# --output_dir: 全参数 GRPO checkpoint 输出目录。
# --warmup_ratio: 学习率 warmup 占总训练步数比例。
# --dataloader_num_workers: DataLoader worker 数量。
# --dataset_num_proc: 数据预处理并行进程数。
# --num_generations: 每个 prompt 采样的 completion 数；8 卡 * batch 2 = 16，正好被 16 整除。
# --temperature: rollout 采样温度，越高生成越多样。
# --top_p: nucleus sampling 截断阈值。
# --top_k: top-k sampling 截断数量。
# --deepspeed zero3: 使用 DeepSpeed ZeRO-3 做全参数 GRPO 显存优化。
# --log_completions: 保存/记录 rollout 生成结果和 reward，便于调试 reward。
# --num_iterations: 每批 rollout 数据重复更新次数；1 表示更接近 on-policy。
# --beta: KL 惩罚系数；全参 GRPO 未显式设置 ref_model 时，ms-swift 默认使用 --model 作为冻结参考策略。
# --save_only_model: 只保存模型权重，不保存优化器等训练状态，节省磁盘空间。
