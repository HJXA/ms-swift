
export MASTER_PORT=29401
export PATH="/ruilab/jxhe/miniconda3/envs/msswift/bin:$PATH"
export NPROC_PER_NODE=1
export CUDA_VISIBLE_DEVICES=4
export NCCL_P2P_LEVEL=NVL
export HF_ENDPOINT=https://hf-mirror.com

export OUTPUT_DIR="/ruilab/jxhe/CoE_Monitor/ms-swift/output/SFT/test/PT_HJXA_Llama_104M"
# 确保目录存在
mkdir -p $OUTPUT_DIR
# datasets 1769,087
swift sft \
  --model /ruilab/jxhe/CoE_Monitor/checkpoints/coe_pt_init_models/Llama_minimind_104M \
  --template minimind \
  --report_to swanlab \
  --truncation_strategy right \
  --swanlab_token WODn49OiskSyv0qBnFZcL \
  --swanlab_project test \
  --save_steps 10000 \
  --max_steps 100000 \
  --lr_scheduler_type warmup_stable_decay \
  --lr_scheduler_kwargs '{"num_decay_steps":0}' \
  --warmup_steps 5000 \
  --cached_dataset /ruilab/jxhe/CoE_Monitor/data/LLM/SFT/Dolci-Instruct-SFT-Llama_template_cached_max_length_2048/train#10000 \
  --use_hf true \
  --load_from_cache_file true \
  --split_dataset_ratio 0 \
  --tuner_type full \
  --torch_dtype bfloat16 \
  --per_device_train_batch_size 128 \
  --attn_impl flash_attention_2 \
  --learning_rate 1e-5 \
  --gradient_checkpointing true \
  --weight_decay 0.0 \
  --logging_steps 1 \
  --max_length 2048 \
  --output_dir $OUTPUT_DIR \
  --dataset_num_proc 4 \
  --dataloader_num_workers 4 \
  --deepspeed zero2 \
  --save_only_model false \
  --dataset_shuffle true \
  --train_dataloader_shuffle true \
  --use_liger_kernel true \
  2>&1 | tee $OUTPUT_DIR/train.log