export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/ruilab2/hjxa/.cache/huggingface # 指定cache位置
export PATH="/ruilab/jxhe/miniconda3/envs/swift/bin:$PATH"

# 禁用 tokenizer 多线程（最关键）
export TOKENIZERS_PARALLELISM=false

# 限制 datasets / numpy / torch 线程
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ================================PT数据集缓存脚本示例===============================
# swift export \
#     --model /ruilab2/hjxa/checkpoints/meta-llama/Llama-2-7b-chat-hf \
#     --dataset ruilab2_data:finemath \
#     --dataset_num_proc 24 \
#     --to_cached_dataset true \
#     --use_chat_template false \
#     --truncation_strategy right \
#     --loss_scale all \
#     --output_dir /ruilab2/hjxa/data/finemath_cached_llama2/


# ===============================SFT数据集缓存脚本示例===============================
swift export \
    --model /ruilab2/PT-LLaMA-0.5B-FineWeb-edu-350B/checkpoints/v8-20260511-151925/checkpoint-350000 \
    --dataset /ruilab2/hjxa/data/SFT/Dolci-Instruct-SFT/dolci_instruct_sft_msswift.jsonl \
    --use_hf true \
    --dataset_num_proc 48 \
    --to_cached_dataset true \
    --dataset_shuffle true \
    --output_dir /ruilab2/hjxa/data/SFT/Dolci-Instruct-SFT/dolci_instruct_sft_msswift_cached-LLaMA_0.5B/ \
    --template_mode train \