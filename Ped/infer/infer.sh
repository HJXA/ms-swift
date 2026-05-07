CUDA_VISIBLE_DEVICES=4 \
swift infer \
    --model /ruilab2/hjxa/checkpoints/qwen/Qwen3/4B/Qwen3-4B \
    --infer_backend transformers \
    --stream true \
    --max_new_tokens 2048 \