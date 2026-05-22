import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional


# 使用方法：
#
# 1. 默认读取本地 parquet：
#    /ruilab2/hjxa/data/SFT/Dolci-Instruct-SFT/dolci_instruct_sft.parquet
#    并转换为 ms-swift 标准 messages JSONL 格式：
#    python HJXA/Custom_Datasets/convert_dolci_instruct_sft.py
#
# 2. 指定输出路径：
#    python HJXA/Custom_Datasets/convert_dolci_instruct_sft.py \
#      --output /ruilab2/hjxa/data/SFT/Dolci-Instruct-SFT/dolci_instruct_sft_msswift.jsonl
#
# 3. 只转换前 1000 条，便于快速检查数据格式：
#    python HJXA/Custom_Datasets/convert_dolci_instruct_sft.py \
#      --max-samples 1000 \
#      --output /ruilab2/hjxa/data/SFT/Dolci-Instruct-SFT/dolci_instruct_sft_1k_msswift.jsonl
#
# 4. 为每条样本增加 system prompt：
#    python HJXA/Custom_Datasets/convert_dolci_instruct_sft.py \
#      --system "You are a helpful assistant." \
#      --output /ruilab2/hjxa/data/SFT/Dolci-Instruct-SFT/dolci_instruct_sft_msswift.jsonl
#
# 输出示例：
# {"messages": [{"role": "user", "content": "<instruction>"}, {"role": "assistant", "content": "<output>"}]}
#
# 常用参数：
# --input: 输入 parquet 文件路径，默认使用上面的 Dolci-Instruct-SFT 路径。
# --output: 输出 JSONL 文件路径。
# --max-samples: 最多写入多少条有效样本。
# --skip-samples: 跳过前多少条原始样本。
# --system: 可选，为 messages 增加 system 角色。
# --strict: 遇到缺少 instruction/output 的样本时报错，而不是跳过。


DEFAULT_INPUT = '/ruilab2/hjxa/data/SFT/Dolci-Instruct-SFT/dolci_instruct_sft.parquet'
DEFAULT_OUTPUT = '/ruilab2/hjxa/data/SFT/Dolci-Instruct-SFT/dolci_instruct_sft_msswift.jsonl'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Convert Dolci-Instruct-SFT parquet to ms-swift messages JSONL.')
    parser.add_argument(
        '--input',
        default=DEFAULT_INPUT,
        help=f'Input parquet path. Default: {DEFAULT_INPUT}')
    parser.add_argument(
        '--output',
        default=DEFAULT_OUTPUT,
        help=f'Output JSONL path. Default: {DEFAULT_OUTPUT}')
    parser.add_argument('--system', default=None, help='Optional system prompt inserted before each user message.')
    parser.add_argument('--max-samples', type=int, default=None, help='Maximum number of valid samples to write.')
    parser.add_argument(
        '--skip-samples',
        type=int,
        default=0,
        help='Number of source samples to skip before converting.')
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Raise an error when a sample lacks non-empty instruction/output instead of skipping it.')
    return parser.parse_args()


def as_non_empty_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def load_parquet_dataset(input_path: str) -> Iterable[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError('Please install the Hugging Face datasets package before loading parquet files.') from e

    return load_dataset('parquet', data_files=input_path, split='train', streaming=True)


def convert_sample(sample: Dict[str, Any], system: Optional[str] = None) -> Optional[Dict[str, Any]]:
    instruction = as_non_empty_string(sample.get('instruction'))
    output = as_non_empty_string(sample.get('output'))
    if instruction is None or output is None:
        return None

    messages = []
    if system:
        messages.append({'role': 'system', 'content': system})
    messages.extend([
        {'role': 'user', 'content': instruction},
        {'role': 'assistant', 'content': output},
    ])
    return {'messages': messages}


def iter_converted_samples(args: argparse.Namespace) -> Iterator[Dict[str, Any]]:
    if args.max_samples is not None and args.max_samples <= 0:
        return

    dataset = load_parquet_dataset(args.input)
    system = as_non_empty_string(args.system)
    skipped_invalid = 0
    written = 0

    try:
        from tqdm import tqdm

        iterator = tqdm(dataset, desc='Converting', unit='sample')
    except ImportError:
        iterator = dataset

    for source_idx, sample in enumerate(iterator):
        if source_idx < args.skip_samples:
            continue

        row = convert_sample(sample, system=system)
        if row is None:
            skipped_invalid += 1
            if args.strict:
                raise ValueError(f'Sample #{source_idx} lacks non-empty instruction/output fields: {sample}')
            continue

        yield row
        written += 1
        if args.max_samples is not None and written >= args.max_samples:
            break

    if skipped_invalid:
        print(f'Skipped {skipped_invalid} invalid samples.')


def write_jsonl(rows: Iterable[Dict[str, Any]], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
            count += 1
    return count


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).expanduser()
    count = write_jsonl(iter_converted_samples(args), output_path)
    print(f'Wrote {count} samples to {output_path}')


if __name__ == '__main__':
    main()
