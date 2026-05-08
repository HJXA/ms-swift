# 开启 postponed evaluation of annotations，避免运行期立即解析所有类型注解。
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional
# ORM 是 ms-swift 的 Outcome Reward Model 基类；orms 是 reward 注册表。
from swift.rewards import ORM, orms

from reward_utils import *


# 预测列表的位置权重：越靠前的预测诊断权重越高。
PRED_WEIGHTS = [1.0, 0.75, 0.5, 0.25, 0.05]
# 标准答案列表的位置权重：越靠前的 gold 诊断权重越高。
GOLD_WEIGHTS = [1.0, 0.75, 0.5, 0.25, 0.05]
# 最多只计算模型输出前 5 个诊断，避免输出很多诊断刷 coverage。
MAX_PREDICTIONS = 5
# 裁判模型最多重试次数。
JUDGE_MAX_RETRIES = 5
# agent judge 默认最大并发数。
MAX_WORKERS = 128
# 持久化 cache 路径：多进程共享同一个 SQLite 文件；需要改路径时只改这个变量。
CACHE_PATH = Path("/ruilab/jxhe/Ped/output/reward_cache/disease_match_cache.sqlite3")



# 病名匹配裁判：用 LLM 判断 pred 和 gold 是否医学上等价。
class DiseaseNameJudge:
    # 裁判调用和 JSON 解析整体最多重试 5 次。
    max_retries = JUDGE_MAX_RETRIES

    def __init__(self, cache: dict[tuple[str, str], bool], cache_lock: threading.Lock):
        # cache/cache_lock 由 PedDiagnosisMatchReward 传入，保证同一进程内 reward 实例共享内存 cache。
        self.cache = cache
        self.cache_lock = cache_lock
        self.cache_path = CACHE_PATH
        self._initialize_cache()

    def _connect_cache(self) -> sqlite3.Connection:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.cache_path), timeout=30)
        # 多进程同时访问时，遇到短暂写锁最多等待 30s，避免立即报 database is locked。
        conn.execute("PRAGMA busy_timeout=30000")
        # NORMAL 比 FULL 少一些同步开销；WAL 模式下通常足够保证训练 cache 的可靠性。
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _initialize_cache(self) -> None:
        with closing(self._connect_cache()) as conn:
            # WAL 允许读写并发：一个进程写入时，其他进程仍可读已提交的数据。
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            # 训练程序只负责读写 SQLite；旧 txt 迁移请单独运行 Ped/migrate_reward_cache.py。
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS disease_match_cache (
                    pred TEXT NOT NULL,
                    gold TEXT NOT NULL,
                    match INTEGER NOT NULL,
                    PRIMARY KEY (pred, gold)
                )
                """
            )
            conn.commit()

    def _load_cache(self, pairs: list[tuple[str, str]]) -> None:
        # 每个 batch 只同步当前 batch 需要的 keys，避免 cache 越大时反复全量扫描。
        keys = list(dict.fromkeys((pred.strip(), gold.strip()) for pred, gold in pairs if pred and gold))
        if not keys:
            return
        with closing(self._connect_cache()) as conn:
            for pred, gold in keys:
                row = conn.execute(
                    "SELECT match FROM disease_match_cache WHERE pred = ? AND gold = ?",
                    (pred, gold),
                ).fetchone()
                if row is None:
                    continue
                with self.cache_lock:
                    self.cache[(pred, gold)] = bool(row[0])

    def _write_cache(self, rows: list[tuple[str, str, bool]]) -> None:
        # rows 是当前 batch 新 judge 出来的结果；本函数统一写入并在退出事务时提交。
        values: list[tuple[str, str, int]] = []
        for pred, gold, match in rows:
            pred = pred.strip()
            gold = gold.strip()
            if not pred or not gold:
                continue
            match_value = int(match)
            # 直接持久化双向结果；读取后按普通 (pred, gold) key 查询即可，不需要反向查找逻辑。
            values.append((pred, gold, match_value))
            values.append((gold, pred, match_value))

        if not values:
            return

        with closing(self._connect_cache()) as conn:
            # with conn 会开启一个事务；executemany 全部成功后在代码块结束时自动 commit。
            with conn:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO disease_match_cache(pred, gold, match)
                    VALUES (?, ?, ?)
                    """,
                    values,
                )

    @staticmethod
    def _parse_match_value(value: Any) -> Optional[bool]:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            value = value.strip().lower()
            if value in {"true", "1", "yes", "y", "是", "匹配", "一致"}: # 其实只会有true
                return True
            if value in {"false", "0", "no", "n", "否", "不匹配", "不一致"}: # 其实只会有false
                return False
        return None

    # 调用 Agno Agent，并返回已解析且包含有效 match 字段的 JSON object。
    def _chat(self, input_payload: dict[str, str]) -> Optional[dict[str, Any]]:
        for attempt in range(1, self.max_retries + 1):
            try:
                response = disease_match_judge_agent.run(
                    input=json.dumps(input_payload, ensure_ascii=False)
                )
                content = getattr(response, "content", response)
                if not isinstance(content, str):
                    content = str(content)

                success, obj = get_json(content)
                if success and isinstance(obj, dict):
                    match = self._parse_match_value(obj.get("match"))
                    if match is not None:
                        obj["match"] = match
                        return obj
            except Exception:
                pass

            if attempt < self.max_retries:
                time.sleep(min(0.1 * attempt, 0.5))

        return None

    # 判断单个预测病名 pred 是否匹配单个标准病名 gold。
    def is_match(self, pred: str, gold: str) -> Optional[bool]:
        # 构造裁判输入：本任务只比较两个病名，不再传入 Top-1 评测场景字段。
        input_payload = {
            "reference_disease": gold,
            "predicted_disease": pred,
        }
        # 调用裁判模型并获取已解析 JSON；失败时该 pair 不参与奖励计算，也不写入 cache。
        obj = self._chat(input_payload)
        if obj is None:
            return None
        # 标准输出中 match 应为 JSON boolean；为兼容异常输出，也接受少量字符串形式。
        return obj.get("match") 


# ms-swift 调用的 ORM reward 类，负责批量 completion 的奖励计算。
class PedDiagnosisMatchReward(ORM):
    # 类级缓存：不同 reward 实例共享 pred/gold 匹配结果，减少重复 LLM judge 调用。
    _match_cache: dict[tuple[str, str], bool] = {}
    # 多线程并发 judge 时保护类级 cache。
    _cache_lock = threading.Lock()

    # 初始化 reward 实例；args 由 ms-swift 传入，当前逻辑不依赖 args。
    def __init__(self, args=None, **kwargs):
        # 调用 ORM 父类初始化，保持与 ms-swift reward 接口兼容。
        super().__init__(args, **kwargs)
        # judge 初始化时会准备持久化 cache；多 reward 实例共享类级内存 cache。
        self.judge = DiseaseNameJudge(self._match_cache, self._cache_lock)
        self.max_workers = MAX_WORKERS

    # 判断 pred 和 gold 是否匹配；先做归一化精确匹配，再必要时调用 LLM judge。
    def _match(self, pred: str, gold: str) -> Optional[bool]:
        # 归一化预测诊断，生成缓存 key 的一部分。
        # 任一名称为空时无法匹配。
        if not pred or not gold:
            return False
        # 归一化后完全相同，直接认为匹配，避免不必要的 LLM judge 调用。
        if pred == gold:
            return True
        # 用归一化后的 pair 做缓存 key，减少重复医学等价判定。
        key = (pred, gold)
        # judge 调用只发生在 _prefetch_matches；这里 cache miss 表示该 pair 不参与奖励计算。
        with self._cache_lock:
            return self._match_cache.get(key)


    # 对 batch 中所有需要 LLM judge 且未缓存的 pair 一次性并发预取。
    def _prefetch_matches(self, pairs: list[tuple[str, str]]) -> tuple[int, int]:
        # 每个 batch 只从 SQLite 同步当前 batch 需要的 keys，避免随着 cache 增大反复全量 IO。
        self.judge._load_cache(pairs)
        cache_hit_count = 0
        pending: dict[tuple[str, str], tuple[str, str]] = {}
        for pred, gold in pairs:
            if not pred or not gold or pred == gold:
                continue
            key = (pred, gold)
            with self._cache_lock:       
                if self._match_cache.get(key) is None:
                # pending 只收集本进程当前内存仍未命中的 pair，后续才会调用耗时的 LLM judge。
                    pending.setdefault(key, (pred, gold))
                else:
                    cache_hit_count += 1

        llm_count = len(pending)

        if not pending:
            return cache_hit_count, llm_count

        new_rows: list[tuple[str, str, bool]] = []
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(pending))) as executor:
            future_to_key = {
                executor.submit(self.judge.is_match, pred, gold): key
                for key, (pred, gold) in pending.items()
            }
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    match = future.result()
                except Exception:
                    continue
                if match is None:
                    continue
                pred, gold = key
                pred = pred.strip()
                gold = gold.strip()

                with self._cache_lock:
                    self._match_cache[(pred, gold)] = match
                    self._match_cache[(gold, pred)] = match
                new_rows.append((pred, gold, match))

        # 一个 batch 的新增 judge 结果统一提交一次
        self.judge._write_cache(new_rows)
        return cache_hit_count, llm_count

    # 计算一组已解析 pred/gold 诊断列表的奖励。
    def _score_names(self, pred: list[str], gold: list[str]) -> float:
        # pred 或 gold 为空时无法计算匹配，奖励为 0。
        if not pred or not gold:
            return 0.0

        # matched_gold_indices 记录哪些 gold 被任意 pred 命中，用于 coverage_score。
        matched_gold_indices: set[int] = set()
        # best_match_score 记录所有 pred/gold 组合中的最高位置加权命中得分。
        best_match_score = 0.0

        # 遍历预测诊断，最多考虑前 MAX_PREDICTIONS 个。
        for pred_idx, pred_name in enumerate(pred[:MAX_PREDICTIONS]):
            # 根据预测位置读取 pred 权重；越靠前权重越高。
            pred_weight = PRED_WEIGHTS[pred_idx] if pred_idx < len(PRED_WEIGHTS) else PRED_WEIGHTS[-1]
            # 当前 pred 与所有 gold 逐一比较。
            for gold_idx, gold_name in enumerate(gold):
                # 不匹配则跳过当前 pred/gold pair。
                if not self._match(pred_name, gold_name):
                    continue
                # 匹配时记录该 gold 已被覆盖。
                matched_gold_indices.add(gold_idx)
                # 根据 gold 位置读取 gold 权重；越靠前权重越高。
                gold_weight = GOLD_WEIGHTS[gold_idx] if gold_idx < len(GOLD_WEIGHTS) else GOLD_WEIGHTS[-1]
                # 更新最佳单点匹配得分。
                # 从前往后的匹配中,哪个是最佳的
                best_match_score = max(best_match_score, pred_weight * gold_weight)

        # coverage_score 表示 gold 中有多少诊断被模型命中。
        coverage_score = len(matched_gold_indices) / len(gold)
        # 按 README 定义混合最佳命中得分和覆盖率得分。
        # reward = 0.8 * best_match_score + 0.2 * coverage_score
        reward = best_match_score + coverage_score

        # 将 reward 裁剪到 [0, 1]，防止异常权重或未来修改造成越界。
        # reward = max(0.0, min(1.0, reward))
        # 返回单条样本奖励。
        return reward

    # ms-swift 会调用 __call__ 批量计算 reward；completions 和 solution 按位置一一对应。
    def __call__(self, completions, solution, **kwargs) -> list[float]:
        parsed: list[tuple[list[str], list[str]]] = []
        pairs: list[tuple[str, str]] = []
        for completion, sol in zip(completions, solution):
            pred = extract_hypotheses(completion, max_predictions=MAX_PREDICTIONS)
            gold = extract_gold(sol)
            parsed.append((pred, gold))
            for pred_name in pred[:MAX_PREDICTIONS]:
                for gold_name in gold:
                    pairs.append((pred_name, gold_name))
        prefetch_start = time.perf_counter()
        cache_hit_count, llm_count = self._prefetch_matches(pairs)
        print(
            f"[ped_reward] prefetch_elapsed={time.perf_counter() - prefetch_start:.3f}s "
            f"pairs={len(pairs)} cache_hit={cache_hit_count} llm={llm_count}"
        )
        # 对 batch 中每条 completion/solution 计算奖励并返回 float 列表。
        rewards = [self._score_names(pred, gold) for pred, gold in parsed]
        return rewards


# 将自定义 ORM 注册到 ms-swift reward 注册表，训练脚本中通过 --reward_funcs ped_diagnosis_match 使用。
orms["ped_diagnosis_match"] = PedDiagnosisMatchReward
