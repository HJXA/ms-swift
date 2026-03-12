from datasets import load_from_disk
from tqdm import tqdm

ds = load_from_disk("/ruilab/jxhe/CoE_Monitor/data/fineweb_cached/sample-350BT/part2/train")

bins = {
    "0-256": 0,
    "256-512": 0,
    "512-1024": 0,
    "1024-1536": 0,
    "1536-2048": 0,
    ">=2048": 0
}

for l in tqdm(ds["lengths"], total=len(ds)):
    x = l[0]

    if x < 256:
        bins["0-256"] += 1
    elif x < 512:
        bins["256-512"] += 1
    elif x < 1024:
        bins["512-1024"] += 1
    elif x < 1536:
        bins["1024-1536"] += 1
    elif x < 2048:
        bins["1536-2048"] += 1
    else:
        bins[">=2048"] += 1

print("\nLength distribution:")

for k, v in bins.items():
    print(f"{k:12s} {v:10d}  ({v/len(ds):.3%})")