from datasets import load_from_disk

# ======================
# 1. 读取本地 dataset
# ======================
dataset_path = "/data/dataset_math"

ds = load_from_disk(dataset_path,streaming=True)

# 如果是 DatasetDict
if "train" in ds:
    ds = ds["train"]

for i,x in enumerate(ds):
    print(x)
    if i >= 5:
        break
