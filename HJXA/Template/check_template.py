import json
from pathlib import Path
from transformers import AutoTokenizer

MODEL_DIR = Path("/ruilab2/hjxa/ms-swift/output/SFT/llama-0.5B-350B-math/checkpoint-70000/v0-20260519-060202/checkpoint-1000")


print("model:", MODEL_DIR)



print("\n--- AutoTokenizer test ---")

try:
    tok = AutoTokenizer.from_pretrained(
        str(MODEL_DIR),
        trust_remote_code=True,
        use_fast=False,
    )
    print("AutoTokenizer load: OK")

    rendered = tok.apply_chat_template(
        [{"role": "user", "content": "你好"},
         {"role": "assistant", "content": "你好！"}],
        tokenize=False,
        add_generation_prompt=True,
    )

    print("apply_chat_template: OK")
    print("\nrendered:")
    print(rendered)

except Exception as e:
    print("FAILED:")
    print(type(e).__name__, e)