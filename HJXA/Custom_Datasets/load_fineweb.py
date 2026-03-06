from swift.dataset import (
    DatasetMeta,
    register_dataset,
    load_dataset,
    SubsetDataset,
    DatasetSyntax,
)
from swift.dataset.loader import DatasetLoader
from swift.dataset.preprocessor import AutoPreprocessor, RowPreprocessor
from datasets import load_dataset as hf_load_dataset
from datasets import Dataset as HfDataset
from typing import Dict, Any, List, Literal, Optional, Tuple, Union

import os

class FineWeb_Loader(DatasetLoader):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def load(
        self,
        dataset_syntax: Optional[DatasetSyntax] = None,
        dataset_meta: Optional[DatasetMeta] = None,
        *,
        num_proc: int = 1,
        load_from_cache_file: bool = True,
        streaming: bool = False,
        use_hf: Optional[bool] = None,
        hub_token: Optional[str] = None,
        strict: bool = False,
        download_mode: Literal['force_redownload', 'reuse_dataset_if_exists'] = 'reuse_dataset_if_exists',
        columns: Optional[Dict[str, str]] = None,
        remove_unused_columns: bool = True,
        **kwargs
    ) -> HfDataset:
        # 1. 定义子集名称到相对路径的映射字典
        subset_map = {
            "CC-MAIN-2025-26": "data/CC-MAIN-2025-26",
            "sample-350BT": "sample/350BT"
        }
        datasets = []

        # 🔥 遍历所有子集
        for subset_name in dataset_syntax.subsets:

            relative_path = subset_map.get(subset_name)
            if not relative_path:
                raise ValueError(
                    f"Unknown FineWeb subset: {subset_name}. "
                    f"Available: {list(subset_map.keys())}"
                )

            full_pattern = os.path.join(
                dataset_meta.dataset_path,
                relative_path,
                "*.parquet"
            )

            dataset = hf_load_dataset(
                "parquet",
                data_files=full_pattern,
                split=kwargs.get("split", "train"),
                streaming=self.streaming,
                num_proc=self.num_proc,
            )

            datasets.append(dataset)

        # 🔥 多子集合并
        if len(datasets) == 1:
            dataset = datasets[0]
        else:
            dataset = concatenate_datasets(datasets)

        # 🔥 后处理
        if self.columns:
            dataset = RowPreprocessor.safe_rename_columns(dataset, self.columns)

        dataset = dataset_meta.preprocess_func(
            dataset,
            num_proc=self.num_proc,
            load_from_cache_file=self.load_from_cache_file,
            strict=self.strict,
        )

        if self.remove_unused_columns:
            dataset = RowPreprocessor.remove_useless_columns(dataset)

        return dataset

def register_local_fineweb():
    register_dataset(
        DatasetMeta(
            dataset_path="/ruilab/jxhe/CoE_Monitor/data/fineweb",
            dataset_name="local_fineweb",
            subsets=[
                SubsetDataset("CC-MAIN-2025-26", split=["train"]),
                SubsetDataset("sample-350BT", split=["train"]),
            ],
            loader=FineWeb_Loader,
            huge_dataset=True,
        )
    )



def load_local_fineweb(split): # To be changed
    # register_local_fineweb() # 已在源码内集成
    dataset = load_dataset(f"local_fineweb:{split}",num_proc=4,columns={"text":"content"},streaming=True)[0]
    return dataset

    


# =========================
# 4️⃣ 测试
# =========================
if __name__ == "__main__":

    dataset = load_local_fineweb("sample-350BT") # 只和Loader.load有关,只有原本的load才会自动调用preprocess_func

    print(dataset)

    for i, sample in enumerate(dataset):
        print(sample)
        if i == 2:
            break