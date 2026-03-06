swift export \
    --model /ruilab/jxhe/CoE_Monitor/checkpoints/coe_pt_init_models/Llama_25M \
    --dataset local_fineweb \
    --dataset_num_proc 128 \
    --to_cached_dataset true \
    --use_chat_template false \
    --truncation_strategy right \
    --loss_scale all \
    --output_dir ./data/fineweb_cached/sample-350BT