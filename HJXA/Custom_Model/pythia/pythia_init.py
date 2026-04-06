import os
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# ===== 路径配置 =====
SRC_MODEL_PATH = "/ruilab/jxhe/CoE_Monitor/checkpoints/Pythia_14M"
DST_MODEL_PATH = "/ruilab/jxhe/CoE_Monitor/checkpoints/coe_pt_init_models/Pythia_14M"

os.makedirs(DST_MODEL_PATH, exist_ok=True)

# ===== 1️⃣ 读取 config =====
config = AutoConfig.from_pretrained(SRC_MODEL_PATH)

# ===== 2️⃣ 用 config 从头初始化模型（随机权重）=====
model = AutoModelForCausalLM.from_config(config)

# （可选）确认是随机初始化
print("模型参数示例（随机初始化）:")
for name, param in model.named_parameters():
    print(name, param.mean().item())
    break

# ===== 3️⃣ 加载 tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(SRC_MODEL_PATH)

# ===== 4️⃣ 保存模型和 tokenizer =====
model.save_pretrained(DST_MODEL_PATH)
tokenizer.save_pretrained(DST_MODEL_PATH)

print(f"新模型（随机初始化）已保存到: {DST_MODEL_PATH}")