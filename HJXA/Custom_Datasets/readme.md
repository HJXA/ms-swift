# 注册
在 /ruilab/jxhe/CoE_Monitor/ms-swift/swift/dataset/dataset/llm.py 中注册

# OpenMathInstruct-2 转 ms-swift messages 格式

脚本: `convert_openmathinstruct2.py`

默认读取 Hugging Face 数据集 `nvidia/OpenMathInstruct-2` 的 `train_1M` split，并流式输出 JSONL：

```bash
python HJXA/Custom_Datasets/convert_openmathinstruct2.py \
  --output /path/to/openmathinstruct2_msswift.jsonl
```

输出格式示例：

```jsonl
{"messages": [{"role": "user", "content": "<problem>"}, {"role": "assistant", "content": "<generated_solution>"}]}
```

常用参数：

```bash
# 只转换前 1000 条，便于快速测试
python HJXA/Custom_Datasets/convert_openmathinstruct2.py \
  --max-samples 1000 \
  --output /path/to/openmathinstruct2_1k_msswift.jsonl

# 转完整 train split
python HJXA/Custom_Datasets/convert_openmathinstruct2.py \
  --split train \
  --output /path/to/openmathinstruct2_train_msswift.jsonl

# 为每条样本增加 system prompt
python HJXA/Custom_Datasets/convert_openmathinstruct2.py \
  --system "You are a helpful math assistant." \
  --output /path/to/openmathinstruct2_msswift.jsonl
```
