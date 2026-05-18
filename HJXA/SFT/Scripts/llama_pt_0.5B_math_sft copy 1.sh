#!/bin/bash
set -e

############################################
# 环境变量
############################################
export MASTER_PORT=20001
export PATH="/ruilab/jxhe/miniconda3/envs/swift/bin:$PATH"
export NPROC_PER_NODE=1
export CUDA_VISIBLE_DEVICES=1
export NCCL_P2P_LEVEL=NVL
export HF_ENDPOINT=https://hf-mirror.com

############################################
# 路径配置
############################################
MODEL_ROOT="/ruilab2/hjxa/checkpoints/pt/llama-0.5B-350B/little_sets"

OUTPUT_BASE_DIR="/ruilab2/hjxa/ms-swift/output/SFT/llama-0.5B-350B"
LOG_DIR="/ruilab2/hjxa/ms-swift/logs/SFT/llama-0.5B-350B"
swanlab_name=$(basename "${OUTPUT_BASE_DIR%/}")
DATASET="/ruilab2/hjxa/data/SFT/OpenMathInstruct-2-msswift_cached-LLaMA_0.5B/train"

mkdir -p "$OUTPUT_BASE_DIR"
mkdir -p "$LOG_DIR"

############################################
# 遍历 checkpoint-* 模型
############################################
for MODEL_PATH in ${MODEL_ROOT}/checkpoint-*; do

    MODEL_NAME=$(basename "$MODEL_PATH")

    echo "=========================================="
    echo "准备训练模型: $MODEL_NAME"
    echo "模型路径: $MODEL_PATH"
    echo "=========================================="

    OUTPUT_DIR="${OUTPUT_BASE_DIR}/${MODEL_NAME}"

    # 已训练跳过
    if [ -d "$OUTPUT_DIR" ]; then
        echo "跳过：已存在 $OUTPUT_DIR"
        continue
    fi

    mkdir -p "$OUTPUT_DIR"

############################################
# 启动训练
############################################

echo "启动训练: $MODEL_NAME"

swift sft \
  --model "$MODEL_PATH" \
  --template llama \
  --report_to swanlab \
  --truncation_strategy right \
  --swanlab_token WODn49OiskSyv0qBnFZcL \
  --swanlab_project $swanlab_name \
  --save_steps 2000 \
  --max_steps 10000 \
  --lr_scheduler_type cosine \
  --warmup_steps 2000 \
  --cached_dataset "$DATASET" \
  --use_hf true \
  --load_from_cache_file true \
  --split_dataset_ratio 0 \
  --tuner_type full \
  --torch_dtype bfloat16 \
  --per_device_train_batch_size 50 \
  --attn_impl flash_attention_2 \
  --learning_rate 1e-5 \
  --gradient_checkpointing true \
  --weight_decay 0.0 \
  --logging_steps 1 \
  --max_length 2048 \
  --output_dir "$OUTPUT_DIR" \
  --dataset_num_proc 4 \
  --dataloader_num_workers 4 \
  --deepspeed zero2 \
  --save_only_model false \
  --dataset_shuffle true \
  --train_dataloader_shuffle true \
  --use_liger_kernel true \
  2>&1 | tee "${LOG_DIR}/${MODEL_NAME}_train.log" \
  || echo "警告: ${MODEL_NAME} 训练失败，跳过..."

echo "完成: $MODEL_NAME"
echo

done