# coding=utf-8
import os
import gc
import torch
from transformers import AutoTokenizer
import sys

sys.path.append("ms-swift/HJXA/Custom_Model/hjxa_qwen2")
from configuration_hjxa_qwen2 import HJXA_Qwen2Config
from modular_hjxa_qwen2 import HJXA_Qwen2ForCausalLM


SAVE_ROOT = "./hjxa_models"
os.makedirs(SAVE_ROOT, exist_ok=True)

# tokenizer 只加载一次
tokenizer = AutoTokenizer.from_pretrained("checkpoints/only_tokenizer/Qwen2")


def count_parameters_detail(model):
    total = 0
    trainable = 0

    embed_params = 0
    lm_head_params = 0
    block_params = 0
    norm_params = 0
    other_params = 0

    for name, p in model.named_parameters():
        n = p.numel()
        total += n

        if p.requires_grad:
            trainable += n

        # 根据名字分类
        if "embed_tokens" in name:
            embed_params += n
        elif "lm_head" in name:
            lm_head_params += n
        elif "layers" in name:
            block_params += n
        elif "norm" in name:
            norm_params += n
        else:
            other_params += n

    return {
        "total": total,
        "trainable": trainable,
        "embed": embed_params,
        "lm_head": lm_head_params,
        "blocks": block_params,
        "norm": norm_params,
        "other": other_params,
    }


def build_save_one(name, cfg):
    print(f"\n===== Building {name} =====")

    config = HJXA_Qwen2Config(
        vocab_size=151936,
        hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["intermediate_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        tie_word_embeddings=False,
    )

    model = HJXA_Qwen2ForCausalLM(config)

    stats = count_parameters_detail(model)

    total = stats["total"]
    trainable = stats["trainable"]
    embed = stats["embed"]
    lm_head = stats["lm_head"]
    blocks = stats["blocks"]
    norm = stats["norm"]
    other = stats["other"]

    print(f"\n{name} total params:     {total:,} ({total/1e6:.2f}M)")
    print(f"{name} trainable params: {trainable:,}")

    print("-" * 60)
    print(f"Embedding:  {embed:,}    ({embed/total*100:.2f}%)")
    print(f"LM Head:    {lm_head:,}    ({lm_head/total*100:.2f}%)")
    print(f"Blocks:     {blocks:,}    ({blocks/total*100:.2f}%)")
    print(f"Norm:       {norm:,}    ({norm/total*100:.2f}%)")
    print(f"Other:      {other:,}    ({other/total*100:.2f}%)")
    print("-" * 60)

    print(f"Embed + LM ratio: {(embed + lm_head)/total*100:.2f}%")
    if blocks > 0:
        print(f"Embed / Blocks ratio: {embed/blocks:.2f}")

    print("-" * 60)

    save_path = os.path.join(SAVE_ROOT, f"HJXA_Qwen2_{name}")
    os.makedirs(save_path, exist_ok=True)

    print(f"Saved to: {save_path}")

    # 释放内存
    del model
    del config
    gc.collect()
    torch.cuda.empty_cache()


# 依次构建（一个一个来） 
MODEL_SIZES = [ ("10M", dict(hidden_size=512, num_hidden_layers=5, num_attention_heads=8, num_key_value_heads=2, intermediate_size=1365)), 
               # ("33M", dict(hidden_size=512, num_hidden_layers=8, num_attention_heads=8, num_key_value_heads=2, intermediate_size=2048)), 
               # ("100M", dict(hidden_size=768, num_hidden_layers=12, num_attention_heads=12, num_key_value_heads=4, intermediate_size=3072)), 
               # ("330M", dict(hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, num_key_value_heads=4, intermediate_size=4096)), 
               # ("1B", dict(hidden_size=1536, num_hidden_layers=32, num_attention_heads=24, num_key_value_heads=8, intermediate_size=6144)), 
            ]

for name, cfg in MODEL_SIZES:
    build_save_one(name, cfg)

print("\nAll models built one-by-one successfully.")