# 环境变量（不要用 \ 拆）
export MASTER_PORT=29506
export PATH="/ruilab/jxhe/miniconda3/envs/msswift/bin:$PATH"
export NPROC_PER_NODE=4
export CUDA_VISIBLE_DEVICES=4,5,6,7
export NCCL_P2P_LEVEL=NVL

# export SWANLAB_RESUME=True
# export SWANLAB_RUN_ID=<exp_id>

# 统一设置输出路径
export OUTPUT_DIR="/ruilab/jxhe/CoE_Monitor/ms-swift/output/PT_HJXA_Llama_100M"
# 确保目录存在
mkdir -p $OUTPUT_DIR
# 启动训练
swift pt \
  --model /ruilab/jxhe/CoE_Monitor/checkpoints/coe_pt_init_models/Llama_100M \
  --packing true \
  --padding_free true \
  --report_to swanlab \
  --truncation_strategy right \
  --swanlab_token WODn49OiskSyv0qBnFZcL \
  --swanlab_project CoE_PT_Main_HJXA_Llama \
  --save_steps 500 \
  --max_steps 100000 \
  --lr_scheduler_type warmup_stable_decay \
  --lr_scheduler_kwargs '{"num_decay_steps":0}' \
  --warmup_steps 5000 \
  --cached_dataset ./data/fineweb_cached/CC-MAIN-2025-26/train#51200000 \
  --load_from_cache_file true \
  --split_dataset_ratio 0 \
  --tuner_type full \
  --torch_dtype bfloat16 \
  --per_device_train_batch_size 128 \
  --attn_impl flash_attention_2 \
  --learning_rate 1e-4 \
  --gradient_checkpointing true \
  --gradient_accumulation_steps 1 \
  --ddp_find_unused_parameters true \
  --weight_decay 0.0 \
  --logging_steps 1 \
  --max_length 2048 \
  --output_dir $OUTPUT_DIR \
  --dataset_num_proc 16 \
  --dataloader_num_workers 16 \
  --deepspeed zero2 \
  --save_only_model false \
  --dataset_shuffle false \
  --train_dataloader_shuffle false \
  --use_liger_kernel true \
  2>&1 | tee $OUTPUT_DIR/train.log

#   --dataset local_fineweb \
#   --columns '{"text":"content"}' \
#   --streaming true \ AssertionError: Cached dataset does not support streaming
#   --device_map /ruilab/jxhe/CoE_Monitor/ms-swift/HJXA/Custom_Model/fix_zero3/llama_25m.json \

# device_map: 模型使用的device_map配置，例如：'auto'、'cpu'、json字符串、json文件路径。该参数会透传入transformers的from_pretrained接口。默认为None，根据设备和分布式训练情况自动设置。

# PyTorch CUDA 内存分配策略：启用可扩展内存段，减少显存碎片
# NPROC_PER_NODE：每个节点使用的进程数（通常等于 GPU 数）
# CUDA_VISIBLE_DEVICES：指定可见的 GPU（这里使用 4,5,6,7 四张卡）
# 使用 swift 的 pt（PyTorch training）入口
#
# swanlab_exp_name：实验名，可以为空；为空时默认使用 --output_dir
# 
#
# --save_steps：每 1000 步保存一次模型
# --streaming true：启用流式数据集（仅顺序采样）
#   注意：流式数据集下即使 dataset_shuffle=false 也只会顺序采样
# --truncation_strategy split \
# --lr_scheduler_type warmup_stable_decay
# --lr_scheduler_kwargs：
#   num_stable_steps=5000，num_decay_steps=5000
#
# 训练数据集：jsonl 格式
# --load_from_cache_file false：不从 HF cache 加载
# --split_dataset_ratio 0：不划分验证集
#
# --train_type full：全参数训练（非 LoRA）
# --torch_dtype bfloat16：使用 bf16 精度
# --per_device_train_batch_size 1：每卡 batch size
#
# --attn_impl flash_attn：FlashAttention 加速并省显存
# --padding_free true：padding-free 训练
#
# --freeze_vit true：冻结视觉编码器
# --freeze_aligner false：不冻结多模态对齐模块
# --freeze_llm true：冻结语言模型
#
# --packing true：样本拼接，提高 token 利用率 # 使用padding-free实现所以要打包就还是自己实现, PT中假设大部分会被截断，而不是packing。
# --gradient_checkpointing true：启用梯度检查点
# --vit_gradient_checkpointing false：ViT 不使用梯度检查点
#
# --gradient_accumulation_steps 1：梯度累积
# --ddp_find_unused_parameters true：适配冻结模块的 DDP
#
# --logging_steps 1：每 1 步打印日志
# --max_length 2048：最大序列长度
# --warmup_ratio 0.1：学习率 warmup 比例
#
# --deepspeed：使用 ZeRO Stage 3 优化显存
# --dataset_num_proc 4：数据集预处理进程数
# --dataloader_num_workers 8：DataLoader worker 数

