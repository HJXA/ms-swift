#!/usr/bin/env bash
set -euo pipefail

export NPROC_PER_NODE=4
export CUDA_VISIBLE_DEVICES=4,5,6,7
export NCCL_P2P_LEVEL=NVL
export HF_ENDPOINT=https://hf-mirror.com

OUTPUT_DIR=/ruilab/jxhe/Ped/output/lora/grpo/Qwen3_14B_sft_thinking_grpo_thinking

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
swift rlhf \
    --rlhf_type grpo \
    --model /ruilab2/hjxa/checkpoints/qwen/Qwen3/14B/Qwen3-14B \
    --adapters /ruilab/jxhe/Ped/output/lora/sft/Qwen3_14B_sft_thinking/v0-20260508-161811/checkpoint-30 \
    --ref_adapters /ruilab/jxhe/Ped/output/lora/sft/Qwen3_14B_sft_thinking/v0-20260508-161811/checkpoint-30 \
    --external_plugins /ruilab/jxhe/Ped/Pedia_clinical_agent/train/ms-swift/Ped/ped_reward.py \
    --reward_funcs ped_diagnosis_match \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.4 \
    --vllm_tensor_parallel_size 4 \
    --vllm_max_model_len 8192 \
    --vllm_enable_lora true \
    --vllm_max_lora_rank 8 \
    --sleep_level 0 \
    --offload_optimizer false \
    --offload_model false \
    --tuner_type lora \
    --torch_dtype bfloat16 \
    --dataset /ruilab/jxhe/Ped/Pedia_clinical_agent/train/datasets/grpo_thinking/train_dataset.jsonl \
    --load_from_cache_file true \
    --split_dataset_ratio 0 \
    --max_length 4096 \
    --max_completion_length 1024 \
    --num_train_epochs 500 \
    --max_steps 5000 \
    --learning_rate 1e-6 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --gradient_checkpointing true \
    --weight_decay 0.0 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --save_steps 100 \
    --logging_steps 1 \
    --output_dir $OUTPUT_DIR \
    --truncation_strategy delete \
    --attn_impl flash_attention_2 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --dataset_shuffle true \
    --train_dataloader_shuffle true \
    --temperature 1.0 \
    --top_p 0.85 \
    --top_k 50 \
    --deepspeed zero2 \
    --use_liger_kernel true \
    --log_completions true \
    --beta 0.04 \
    --save_only_model false \
    --report_to swanlab \
    --swanlab_project Ped \
    --swanlab_token WODn49OiskSyv0qBnFZcL \
    --overlong_filter true \
    --log_entropy false \
    --log_rollout_offpolicy_metrics true \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --generation_batch_size 128 \
    --num_generations 32 \
    --num_iterations 1 \


# 在 GRPO 中，batch_size 以 completion（模型生成结果） 为单位
# --per_device_train_batch_size 8 # 表示每张 GPU 在训练过程中会同时处理 8 个 completion 的 loss 计算。
# 训练阶段，在一次完整的梯度累计 batch 中，总的批量大小等于
# effective_batch_size = num_processes * per_device_train_batch_size * gradient_accumulation_steps
# --gradient_accumulation_steps 默认1
# 以下是生成阶段的控制参数
# --steps_per_generation: 每轮生成的优化步数，默认等于 gradient_accumulation_steps。与 generation_batch_size 只能同时设置一个。
# --generation_batch_size  总的采样 completion 批量大小。需要是 num_processes * per_device_train_batch_size 的倍数。默认等于 per_device_train_batch_size * steps_per_generation * num_processes。
# --num_generations 每个prompt采样的数量，论文中的G值，generation_batch_size 必须能被 num_generations 整除。
# --num_iterations 每条数据的更新次数，GRPO论文中的u值，默认为1。

# 想要以一次生成的数据在一个batch中被全部用上：num_processes * per_device_train_batch_size * gradient_accumulation_steps = generation_batch_size = effective_batch_size
# 示例
# num_processes = 8
# per_device_train_batch_size = 4
# gradient_accumulation_steps = 8
# generation_batch_size = 512
# num_generations = 64

# 采样需要的总数据(prompt)量等于 512 / 64 = 8
# 每次采样 512 条模型回复
# 每次更新模型权重批量大小为 8 *4 * 8 = 256