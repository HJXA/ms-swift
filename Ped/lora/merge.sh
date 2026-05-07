export CUDA_VISIBLE_DEVICES=4

swift export \
    --adapters /ruilab/jxhe/Ped/output/lora/v0-20260507-105444/checkpoint-30 \
    --merge_lora true