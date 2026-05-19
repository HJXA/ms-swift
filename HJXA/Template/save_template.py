from pathlib import Path
from transformers import AutoTokenizer

ROOTS = [
    Path("/ruilab2/hjxa/ms-swift/output/SFT/llama-0.5B-350B-math-full"),
]

chat_template = r"""{% if messages[0]['role'] == 'system' %}
{% set loop_messages = messages[1:] %}
{% set system_message = messages[0]['content'] %}
{% else %}
{% set loop_messages = messages %}
{% set system_message = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information." %}
{% endif %}
{% for message in loop_messages %}
{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
{% endif %}
{% if message['role'] == 'user' %}
{% if loop.index0 == 0 and system_message %}
{{ bos_token + '[INST] <<SYS>>\n' + system_message + '\n<</SYS>>\n\n' + message['content'].strip() + ' [/INST]' }}
{% else %}
{{ bos_token + '[INST] ' + message['content'].strip() + ' [/INST]' }}
{% endif %}
{% elif message['role'] == 'assistant' %}
{{ ' ' + message['content'].strip() + eos_token }}
{% endif %}
{% endfor %}"""


def find_tokenizer_dirs(root: Path):
    """
    找到 root 下所有可以 AutoTokenizer.from_pretrained() 的 checkpoint 目录。
    只处理目录名为 checkpoint-* 且里面有 tokenizer 配置/词表文件的目录。
    """
    candidates = []

    for ckpt in sorted(root.rglob("checkpoint-*")):
        if not ckpt.is_dir():
            continue

        has_tokenizer_file = any(
            (ckpt / name).exists()
            for name in [
                "tokenizer_config.json",
                "tokenizer.json",
                "tokenizer.model",
                "vocab.json",
                "vocab.txt",
                "spiece.model",
            ]
        )

        if has_tokenizer_file:
            candidates.append(ckpt)

    return candidates


def patch_one_model(model_dir: Path):
    print(f"[LOAD] {model_dir}")

    tok = AutoTokenizer.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        use_fast=False,
    )

    tok.chat_template = chat_template
    tok.save_pretrained(str(model_dir))

    print(f"[OK]   saved chat_template -> {model_dir}")


def main():
    all_dirs = []

    for root in ROOTS:
        if not root.exists():
            print(f"[WARN] root not found: {root}")
            continue
        all_dirs.extend(find_tokenizer_dirs(root))

    # 去重
    all_dirs = sorted(set(all_dirs))

    print(f"Found {len(all_dirs)} tokenizer checkpoint dirs.")

    ok = 0
    failed = 0

    for model_dir in all_dirs:
        try:
            patch_one_model(model_dir)
            ok += 1
        except Exception as e:
            failed += 1
            print(f"[FAIL] {model_dir}")
            print(f"       {type(e).__name__}: {e}")

    print()
    print(f"Done. ok={ok}, failed={failed}")


if __name__ == "__main__":
    main()