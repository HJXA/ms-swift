import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional


# 使用方法：
#
# 1. 默认转换 Hugging Face 数据集 nvidia/OpenMathInstruct-2 的 train_1M split，
#    输出为 ms-swift 标准 messages JSONL 格式：
#    python HJXA/Custom_Datasets/convert_openmathinstruct2.py \
#      --output /path/to/openmathinstruct2_msswift.jsonl
#
# 2. 转换完整 train split：
#    python HJXA/Custom_Datasets/convert_openmathinstruct2.py \
#      --split train \
#      --output /path/to/openmathinstruct2_train_msswift.jsonl
#
# 3. 只转换前 1000 条，便于快速检查数据格式：
#    python HJXA/Custom_Datasets/convert_openmathinstruct2.py \
#      --max-samples 1000 \
#      --output /path/to/openmathinstruct2_1k_msswift.jsonl
#
# 4. 为每条样本增加 system prompt：
#    python HJXA/Custom_Datasets/convert_openmathinstruct2.py \
#      --system "You are a helpful math assistant." \
#      --output /path/to/openmathinstruct2_msswift.jsonl
#
# 输出示例：
# {"messages": [{"role": "user", "content": "<problem>"}, {"role": "assistant", "content": "<generated_solution>"}]}
#
# 常用参数：
# --split: 可选 train、train_1M、train_2M、train_5M，默认 train_1M。
# --max-samples: 最多写入多少条有效样本。
# --skip-samples: 跳过前多少条原始样本。
# --system: 可选，为 messages 增加 system 角色。
# --keep-extra: 额外保留 expected_answer 和 problem_source 字段。
# --no-streaming: 关闭流式读取，完整 train split 可能占用较多磁盘和内存。


DEFAULT_DATASET = 'nvidia/OpenMathInstruct-2'
DEFAULT_OUTPUT = 'openmathinstruct2_msswift.jsonl'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Convert nvidia/OpenMathInstruct-2 samples to the ms-swift messages JSONL format.')
    parser.add_argument(
        '--dataset',
        default=DEFAULT_DATASET,
        help=f'Hugging Face dataset name. Default: {DEFAULT_DATASET}')
    parser.add_argument('--config-name', default=None, help='Optional Hugging Face dataset config name.')
    parser.add_argument(
        '--split',
        default='train',
        help='Dataset split to convert. Available OpenMathInstruct-2 splits include '
        'train, train_1M, train_2M, train_5M.')
    parser.add_argument('--revision', default=None, help='Optional dataset revision.')
    parser.add_argument('--cache-dir', default=None, help='Optional Hugging Face datasets cache directory.')
    parser.add_argument('--hf-token', default=None, help='Optional Hugging Face token.')
    parser.add_argument(
        '--no-streaming',
        action='store_true',
        help='Disable streaming. This may require large disk/RAM for the full train split.')
    parser.add_argument(
        '--num-proc',
        type=int,
        default=None,
        help='num_proc passed to datasets.load_dataset when streaming is disabled.')
    parser.add_argument('--output', default=DEFAULT_OUTPUT, help=f'Output JSONL path. Default: {DEFAULT_OUTPUT}')
    parser.add_argument('--system', default=None, help='Optional system prompt inserted before each user message.')
    parser.add_argument('--max-samples', type=int, default=None, help='Maximum number of valid samples to write.')
    parser.add_argument(
        '--skip-samples',
        type=int,
        default=0,
        help='Number of source samples to skip before converting.')
    parser.add_argument(
        '--keep-extra',
        action='store_true',
        help='Keep expected_answer and problem_source as extra top-level fields for traceability.')
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Raise an error when a sample lacks problem/generated_solution instead of skipping it.')
    return parser.parse_args()


def load_openmath_dataset(args: argparse.Namespace) -> Iterable[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError('Please install the Hugging Face datasets package before loading OpenMathInstruct-2.') from e

    load_kwargs = {
        'split': args.split,
        'streaming': not args.no_streaming,
    }
    if args.revision:
        load_kwargs['revision'] = args.revision
    if args.cache_dir:
        load_kwargs['cache_dir'] = args.cache_dir
    if args.hf_token:
        load_kwargs['token'] = args.hf_token
    if args.no_streaming and args.num_proc:
        load_kwargs['num_proc'] = args.num_proc

    if args.config_name:
        return load_dataset(args.dataset, args.config_name, **load_kwargs)
    return load_dataset(args.dataset, **load_kwargs)


def as_non_empty_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def convert_sample(
        sample: Dict[str, Any],
        system: Optional[str] = None,
        keep_extra: bool = False) -> Optional[Dict[str, Any]]:
    problem = as_non_empty_string(sample.get('problem'))
    solution = as_non_empty_string(sample.get('generated_solution'))
    if problem is None or solution is None:
        return None

    messages = []
    if system:
        messages.append({'role': 'system', 'content': system})
    messages.extend([
        {'role': 'user', 'content': problem},
        {'role': 'assistant', 'content': solution},
    ])

    row = {'messages': messages}
    if keep_extra:
        expected_answer = sample.get('expected_answer')
        problem_source = sample.get('problem_source')
        if expected_answer is not None:
            row['expected_answer'] = expected_answer
        if problem_source is not None:
            row['problem_source'] = problem_source
    return row


def iter_converted_samples(args: argparse.Namespace) -> Iterator[Dict[str, Any]]:
    dataset = load_openmath_dataset(args)
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

        row = convert_sample(sample, system=system, keep_extra=args.keep_extra)
        if row is None:
            skipped_invalid += 1
            if args.strict:
                raise ValueError(f'Sample #{source_idx} lacks non-empty problem/generated_solution fields: {sample}')
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
