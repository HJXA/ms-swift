export HF_ENDPOINT=https://hf-mirror.com
# ================================PT数据集缓存脚本示例===============================
# swift export \
#     --model /ruilab/jxhe/CoE_Monitor/checkpoints/coe_pt_init_models/Llama_25M \
#     --dataset local_fineweb:sample-350BT-part3 \
#     --dataset_num_proc 24 \
#     --to_cached_dataset true \
#     --use_chat_template false \
#     --truncation_strategy right \
#     --loss_scale all \
#     --output_dir ./data/fineweb_cached/sample-350BT/part3


# ===============================SFT数据集缓存脚本示例===============================
swift export \
    --model /ruilab/jxhe/CoE_Monitor/checkpoints/coe_pt_init_models/Llama_minimind_104M \
    --dataset allenai/Dolci-Instruct-SFT \
    --use_hf true \
    --dataset_num_proc 24 \
    --to_cached_dataset true \
    --dataset_shuffle true \
    --output_dir ./data/Dolci-Instruct-SFT-Llama_template_cached_max_length_32768