export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/ruilab2/jxhe/.cache/huggingface # 指定cache位置
# ================================PT数据集缓存脚本示例===============================
swift export \
    --model /ruilab/jxhe/CoE_Monitor/checkpoints/pythia_14m \
    --dataset /ruilab2/jxhe/data/pile_deduplicated \
    --dataset_num_proc 24 \
    --to_cached_dataset true \
    --use_chat_template false \
    --truncation_strategy right \
    --loss_scale all \
    --output_dir /ruilab2/jxhe/data/pile_deduplicated_cached/


# ===============================SFT数据集缓存脚本示例===============================
# swift export \
#     --model /ruilab/jxhe/CoE_Monitor/checkpoints/coe_pt_init_models/Llama_minimind_104M \
#     --dataset allenai/Dolci-Instruct-SFT \
#     --use_hf true \
#     --dataset_num_proc 24 \
#     --to_cached_dataset true \
#     --dataset_shuffle true \
#     --output_dir ./data/Dolci-Instruct-SFT-Llama_template_cached_max_length_32768
#.    --template_mode train \