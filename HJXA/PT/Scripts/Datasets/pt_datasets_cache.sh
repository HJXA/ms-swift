# export HF_ENDPOINT=https://hf-mirror.com
# swift export \
#     --model /ruilab/jxhe/CoE_Monitor/checkpoints/coe_pt_init_models/Llama_25M \
#     --dataset local_fineweb:sample-350BT-part3 \
#     --dataset_num_proc 24 \
#     --to_cached_dataset true \
#     --use_chat_template false \
#     --truncation_strategy right \
#     --loss_scale all \
#     --output_dir ./data/fineweb_cached/sample-350BT/part3

swift export \
    --model /ruilab/jxhe/CoE_Monitor/checkpoints/coe_pt_init_models/Llama_25M \
    --cached_dataset ./data/fineweb_cached/CC-MAIN-2025-26/train#100000000 \
    --dataset_num_proc 24 \
    --to_cached_dataset true \
    --use_chat_template false \
    --truncation_strategy right \
    --loss_scale all \
    --dataset_shuffle true \
    --output_dir ./data/fineweb_cached/subsets-100M-samples