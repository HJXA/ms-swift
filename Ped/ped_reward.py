"""Self-contained GRPO reward for pediatric diagnosis hypothesis training.

The reward follows ``train/README.md``:

    reward = 0.8 * best_match_score + 0.2 * coverage_score

Disease matching is judged by a Top-1 LLM judge implemented locally in this
file. This plugin intentionally does not import project evaluation utilities.
"""

from __future__ import annotations

import ast
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

from swift.rewards import ORM, orms


PRED_WEIGHTS = [1.0, 0.7, 0.45, 0.25, 0.15]
GOLD_WEIGHTS = [1.0, 0.75, 0.5, 0.3, 0.2]
MAX_PREDICTIONS = 5


JUDGE_SYSTEM_PROMPT = """
你是一位资深儿科主任医师兼医学教育专家，拥有30年临床经验和丰富的病案讨论与教学经验。

## 任务
你需要根据参考诊断（reference_diagnosis）评估AI诊断（ai_diagnosis）的匹配程度，并给出结构化评价结果。

输入包含：
- reference_diagnosis：参考诊断（可能包含多个候选，但表示“可能正确答案之一”，而不是要求AI诊断全部覆盖参考答案）
- ai_diagnosis：AI诊断（注意当AI诊断有多个时只看第一个诊断，即Top1诊断进行匹配评估）
- status：可选字段，表示 AI 诊断状态，可能为 solved 或 researching。

## 核心原则（必须遵守）
1. 你要像一个真实的上级医师审核下级医师的诊断报告一样进行评判。
2. 疾病名称的匹配应基于临床实质，而非文字表面。
3. 以下情况应视为“匹配”：
- 同一疾病的不同名称（如“川崎病”=“黏膜皮肤淋巴结综合征”）。
- 同一疾病的中英文名称（如“Kawasaki Disease”=“川崎病”）。
- 临床上等价的诊断表述（如“化脓性扁桃体炎”≈“急性细菌性扁桃体炎”）。
- 正确识别了疾病大类且亚型方向正确（如标准答案是“急性淋巴细胞白血病”，AI回答“急性白血病”，应视为部分匹配）。
4. 以下情况不应视为匹配：
- 仅仅是同一器官/系统的疾病但病因和处置完全不同。
- 过于宽泛的诊断（如标准答案是“川崎病”，AI回答“发热待查”）。
- 虽然症状相似但临床处置路径完全不同的疾病。
5. 诊断优先于症状：若参考诊断包含“疾病”和“症状”，优先以“疾病级别”进行匹配判断；症状仅在没有疾病级别时进行匹配。
6. researching 状态评测规则：当 status="researching" 时，ai_diagnosis 可能是“症状诊断:倾向诊断”的临床诊断格式。此时只评估冒号后的倾向诊断 Top1，不要用冒号前的症状诊断命中参考诊断。

## 判定规则（严格执行）
1. 若AI诊断命中参考诊断中任一的“疾病” -> 完全一致。
2. 若诊断均未命中 -> 不一致。

## 输出要求（必须严格遵守JSON格式）
{
  "ai_diagnosis_assessment": "对匹配关系的医学分析（需明确指出是否命中、为何未命中）",
  "diagnosis_level": "完全一致 / 不一致"
}

## 注意事项
- 必须明确指出：要说明AI诊断命中的是哪个疾病，同时说明原因。如果未命中，要说明AI诊断的疾病与参考诊断中的疾病为何不匹配。
- 忽略AI诊断中的可能性字样，只要出现疾病名称即视为正常诊断。
""".strip()


def _load_local_env() -> None:
    """Load .env from this file's parents without depending on project helpers."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass

    for parent in Path(__file__).resolve().parents:
        env_path = parent / ".env"
        if not env_path.is_file():
            continue
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
        except OSError:
            continue


def _matching_brace_end(text: str, start: int) -> Optional[int]:
    depth = 0
    in_string = False
    escaped = False
    quote = ""
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                in_string = False
            continue
        if char in {'"', "'"}:
            in_string = True
            quote = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return idx + 1
    return None


def _parse_json_object(content: Any) -> Optional[dict[str, Any]]:
    if isinstance(content, dict):
        return content
    if content is None:
        return None
    text = str(content).strip()
    if not text:
        return None

    search_regions = []
    think_end = text.rfind("</think>")
    if think_end >= 0:
        search_regions.append(text[think_end + len("</think>") :])
    search_regions.append(text)

    for region in search_regions:
        start = region.find("{")
        while start >= 0:
            end = _matching_brace_end(region, start)
            if end is None:
                break
            candidate = region[start:end]
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(candidate)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    continue
            start = region.find("{", start + 1)
    return None


def _extract_hypotheses(content: Any) -> list[str]:
    obj = _parse_json_object(content)
    if not isinstance(obj, dict):
        return []
    hypotheses = obj.get("合理假设", [])
    if not isinstance(hypotheses, list):
        return []

    names: list[str] = []
    for item in hypotheses[:MAX_PREDICTIONS]:
        if isinstance(item, str):
            name = item.strip()
        elif isinstance(item, dict):
            name = str(item.get("诊断") or item.get("疾病名称") or item.get("name") or "").strip()
        else:
            name = str(item).strip()
        if name:
            names.append(name)
    return names


def _extract_gold(solution: Any) -> list[str]:
    if isinstance(solution, dict):
        hypotheses = solution.get("合理假设", [])
    elif isinstance(solution, list):
        hypotheses = solution
    else:
        obj = _parse_json_object(solution)
        hypotheses = obj.get("合理假设", []) if isinstance(obj, dict) else []

    if not isinstance(hypotheses, list):
        return []

    names: list[str] = []
    for item in hypotheses:
        if isinstance(item, str):
            name = item.strip()
        elif isinstance(item, dict):
            name = str(item.get("诊断") or item.get("疾病名称") or item.get("name") or "").strip()
        else:
            name = str(item).strip()
        if name:
            names.append(name)
    return names


def _normalize_disease(name: str) -> str:
    text = str(name).lower().strip()
    text = re.sub(r"^(主要诊断|ai诊断|诊断)[:：]", "", text)
    text = re.sub(r"[\s;；,，。.!！?？:：\-—_、（）()【】\[\]<>《》\"'“”‘’]+", "", text)
    return text


def _is_local_url(url: str) -> bool:
    return "127.0.0.1" in url or "localhost" in url or "0.0.0.0" in url


class Top1DiagnosisJudge:
    def __init__(self):
        _load_local_env()
        self.model = os.getenv("PED_JUDGE_MODEL", "z-ai/glm-5-turbo")
        self.base_url = (
            os.getenv("PED_JUDGE_BASE_URL")
            or os.getenv("OPENROUTER_BASE_URL")
            or "https://openrouter.ai/api/v1"
        ).rstrip("/")
        self.api_key = os.getenv("PED_JUDGE_API_KEY") or os.getenv("OPENROUTER_API_KEY") or ""
        if not self.api_key and _is_local_url(self.base_url):
            self.api_key = "EMPTY"
        if not self.api_key:
            raise RuntimeError(
                "PED/OpenRouter judge is not configured. Set OPENROUTER_API_KEY or PED_JUDGE_API_KEY."
            )
        self.timeout = float(os.getenv("PED_JUDGE_TIMEOUT", "120"))
        self.max_retries = int(os.getenv("PED_JUDGE_MAX_RETRIES", "3"))
        self.temperature = float(os.getenv("PED_JUDGE_TEMPERATURE", "0"))
        self.max_tokens = int(os.getenv("PED_JUDGE_MAX_TOKENS", "768"))
        self.reasoning_enabled = os.getenv("PED_JUDGE_REASONING", "true").lower() not in {"0", "false", "no"}

    def _chat(self, input_payload: dict[str, str]) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(input_payload, ensure_ascii=False)},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.reasoning_enabled:
            payload["reasoning"] = {"enabled": True, "effort": "high"}

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers=headers,
            method="POST",
        )

        last_error: Optional[BaseException] = None
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = response.read().decode("utf-8")
                obj = json.loads(raw)
                return str(obj["choices"][0]["message"].get("content") or "")
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"HTTP {exc.code}: {body[:1000]}")
            except Exception as exc:
                last_error = exc
            if attempt < self.max_retries - 1:
                time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"Diagnosis judge request failed: {last_error}")

    def is_match(self, pred: str, gold: str) -> bool:
        input_payload = {
            "reference_diagnosis": gold,
            "ai_diagnosis": f"主要诊断：{pred}",
        }
        content = self._chat(input_payload)
        obj = _parse_json_object(content)
        if not isinstance(obj, dict):
            if os.getenv("PED_REWARD_DEBUG", "0") == "1":
                print(f"[ped_reward] judge JSON parse failed: {content[:500]}")
            return False
        return str(obj.get("diagnosis_level", "")).strip() == "完全一致"


class PedDiagnosisMatchReward(ORM):
    _match_cache: dict[tuple[str, str], bool] = {}

    def __init__(self, args=None, **kwargs):
        super().__init__(args, **kwargs)
        self._judge: Optional[Top1DiagnosisJudge] = None
        self.debug = os.getenv("PED_REWARD_DEBUG", "0") == "1"

    @property
    def judge(self) -> Top1DiagnosisJudge:
        if self._judge is None:
            self._judge = Top1DiagnosisJudge()
        return self._judge

    def _match(self, pred: str, gold: str) -> bool:
        pred_norm = _normalize_disease(pred)
        gold_norm = _normalize_disease(gold)
        if not pred_norm or not gold_norm:
            return False
        if pred_norm == gold_norm:
            return True
        key = (pred_norm, gold_norm)
        if key not in self._match_cache:
            self._match_cache[key] = self.judge.is_match(pred, gold)
        return self._match_cache[key]

    def _score_one(self, completion: Any, solution: Any) -> float:
        pred = _extract_hypotheses(completion)
        gold = _extract_gold(solution)
        if not pred or not gold:
            return 0.0

        matched_gold_indices: set[int] = set()
        best_match_score = 0.0

        for pred_idx, pred_name in enumerate(pred[:MAX_PREDICTIONS]):
            pred_weight = PRED_WEIGHTS[pred_idx] if pred_idx < len(PRED_WEIGHTS) else PRED_WEIGHTS[-1]
            for gold_idx, gold_name in enumerate(gold):
                if not self._match(pred_name, gold_name):
                    continue
                matched_gold_indices.add(gold_idx)
                gold_weight = GOLD_WEIGHTS[gold_idx] if gold_idx < len(GOLD_WEIGHTS) else GOLD_WEIGHTS[-1]
                best_match_score = max(best_match_score, pred_weight * gold_weight)

        coverage_score = len(matched_gold_indices) / len(gold)
        reward = 0.8 * best_match_score + 0.2 * coverage_score
        reward = max(0.0, min(1.0, reward))
        if self.debug:
            print(
                "[ped_reward]",
                json.dumps(
                    {
                        "pred": pred,
                        "gold": gold,
                        "best_match_score": best_match_score,
                        "coverage_score": coverage_score,
                        "reward": reward,
                    },
                    ensure_ascii=False,
                ),
            )
        return reward

    def __call__(self, completions, solution, **kwargs) -> list[float]:
        return [self._score_one(completion, sol) for completion, sol in zip(completions, solution)]


orms["ped_diagnosis_match"] = PedDiagnosisMatchReward
