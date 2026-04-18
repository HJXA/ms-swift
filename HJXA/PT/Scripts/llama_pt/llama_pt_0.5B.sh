# 环境变量（不要用 \ 拆）
export MASTER_PORT=29506
export PATH="/ruilab/jxhe/miniconda3/envs/swift/bin:$PATH"
export NPROC_PER_NODE=4
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_P2P_LEVEL=NVL

# export SWANLAB_RESUME=True
# export SWANLAB_RUN_ID=<exp_id>

# 统一设置输出路径
export OUTPUT_DIR="/ruilab2/hjxa/ms-swift/output/PT_LLaMA_0.5B"
# 确保目录存在
mkdir -p $OUTPUT_DIR
# 启动训练
swift pt \
  --model /ruilab/jxhe/CoE_Monitor/checkpoints/PT_Init_Models/Llama_0.5B \
  --packing true \
  --packing_num_proc 16 \
  --padding_free true \
  --report_to swanlab \
  --truncation_strategy right \
  --swanlab_token WODn49OiskSyv0qBnFZcL \
  --swanlab_project CoE_PT_Main_Llama_Evolm \
  --save_steps 500 \
  --max_steps 664062 \
  --num_train_epochs 1 \
  --lr_scheduler_type cosine_with_min_lr \
  --lr_scheduler_kwargs '{"min_lr_rate": 0.1}' \
  --warmup_ratio 0.1 \
  --cached_dataset /ruilab2/hjxa/data/fineweb_edu_sample_350BT_cached_llama2/train \
  --load_from_cache_file true \
  --split_dataset_ratio 0 \
  --tuner_type full \
  --torch_dtype bfloat16 \
  --per_device_train_batch_size 128 \
  --attn_impl flash_attention_2 \
  --learning_rate 2.5e-4 \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --max_grad_norm 1 \
  --gradient_checkpointing true \
  --gradient_accumulation_steps 1 \
  --weight_decay 0.1 \
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

