# export HF_ENDPOINT=https://hf-mirror.com
swift export \
    --model /ruilab/jxhe/CoE_Monitor/checkpoints/coe_pt_init_models/Llama_25M \
    --dataset local_fineweb:sample-350BT-part2 \
    --dataset_num_proc 24 \
    --to_cached_dataset true \
    --use_chat_template false \
    --truncation_strategy right \
    --loss_scale all \
    --output_dir ./data/fineweb_cached/sample-350BT/part2