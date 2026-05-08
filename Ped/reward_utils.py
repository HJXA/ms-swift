# 开启 postponed evaluation of annotations，避免运行期立即解析所有类型注解。
from __future__ import annotations
import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Optional

JUDGE_SYSTEM_PROMPT = """
你是一位资深儿科医生。你的任务是判断两个诊断名称在临床语义上是否表示同一种疾病或高度等价的诊断。

## 任务
请比较标准诊断名称（reference_disease）和模型预测诊断名称（predicted_disease），判断二者是否匹配。

输入包含：
- reference_disease：标准诊断名称。
- predicted_disease：模型预测诊断名称。

## 核心原则（必须遵守）
1. 疾病名称的匹配应基于临床实质，而不是文字表面。
2. 以下情况应判为匹配：
- 同一疾病的不同名称（如“川崎病”=“黏膜皮肤淋巴结综合征”）。
- 同一疾病的中英文名称（如“Kawasaki Disease”=“川崎病”）。
- 临床上等价的诊断表述（如“化脓性扁桃体炎”≈“急性细菌性扁桃体炎”）。
- 正确识别了疾病大类且亚型方向正确（如标准答案是“急性淋巴细胞白血病”，预测为“急性白血病”）。
3. 以下情况应判为不匹配：
- 仅仅是同一器官/系统的疾病但病因和处置完全不同。
- 过于宽泛的诊断（如标准答案是“川崎病”，AI回答“发热待查”）。
- 虽然症状相似但临床处置路径完全不同的疾病。
4. 诊断优先于症状：如果一个名称是明确疾病，另一个只是症状，通常不应判为匹配；只有标准诊断本身也是症状诊断时才可匹配症状。

## 输出要求（必须严格遵守JSON格式）
只输出 JSON，不要输出 Markdown，不要输出额外解释。

字段含义：
- match：布尔值，true 表示两个病名匹配，false 表示不匹配。
- reason：一句话说明判断依据。

输出格式：
{
  "match": true,
  "reason": "说明为什么匹配或不匹配"
}
"""

# 通用 JSON 解析函数：从文本中截取最外层 JSON 片段并解析。
def get_json(content: str, type: str = "dict") -> tuple[bool, dict | list]:
    """
    通用 JSON 解析函数。
    首先尝试使用 json.loads 解析，失败时回退到 eval 解析。
    """
    # 所有解析异常统一在外层捕获，返回 success=False。
    try:
        # dict 模式从第一个 { 截到最后一个 }。
        if type == "dict":
            start_idx = content.index("{")
            end_idx = content.rindex("}") + 1
            json_str = content[start_idx:end_idx]
        # list 模式从第一个 [ 截到最后一个 ]。
        else:
            start_idx = content.index("[")
            end_idx = content.rindex("]") + 1
            json_str = content[start_idx:end_idx]

        # 优先按标准 JSON 解析。
        try:
            result = json.loads(json_str)
            return True, result
        # 如果模型输出不是严格 JSON，例如单引号 dict，则回退 eval。
        except json.JSONDecodeError:
            result = eval(json_str)
            return True, result

    # 解析失败时返回错误信息，调用方可自行决定是否给 0 reward。
    except Exception as e:
        return False, {"error": "解析JSON失败", "\ncontent:\n": content, "exception": str(e)}
    

# 从模型 completion 中抽取预测的诊断假设列表。
def extract_hypotheses(content: Any, max_predictions: int = 6) -> list[str]:
    # 先从 completion 中解析 JSON object。
    success, obj = get_json(content)
    # 如果解析不到 dict，说明模型没有输出可用 JSON，直接给空列表。
    if not success or not isinstance(obj, dict):
        return []
    # 读取任务约定字段“合理假设”。
    hypotheses = obj.get("合理假设", [])
    # “合理假设”必须是列表，否则视为无效输出。
    if not isinstance(hypotheses, list):
        return []

    # names 保存最终抽取出的诊断名称字符串。
    names: list[str] = []
    # 只遍历前 max_predictions 个，防止模型输出过长列表刷覆盖率。
    for item in hypotheses[:max_predictions]:
        name = ""
        # 最理想情况：列表元素本身就是疾病名称字符串。
        if isinstance(item, str):
            name = item.strip()
        # 只保留非空诊断名称。
        if name:
            names.append(name)
    # 返回按模型原始顺序排列的诊断名称。
    return names

# 从数据集 solution 字段中抽取 gold 诊断列表。
def extract_gold(solution: Any) -> list[str]:
    hypotheses = []
    # 标准 GRPO 数据中 solution 是 dict，形如 {"合理假设": [...]}。
    if isinstance(solution, dict):
        hypotheses = solution.get("合理假设", [])

    # gold 必须能形成列表，否则本样本无法计算奖励。
    if not isinstance(hypotheses, list):
        return []

    # names 保存标准答案中的诊断名称。
    names: list[str] = []
    # 遍历 gold 中所有诊断；gold 不截断，因为 coverage 需要覆盖所有标准诊断。
    for item in hypotheses:
        name = ""
        # 标准情况：gold 元素是字符串。
        if isinstance(item, str):
            name = item.strip()
        # 只保留非空名称。
        if name:
            names.append(name)
    # 返回 gold 诊断列表。
    return names

import json
import logging
import os
import re
import time
import datetime
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import copy
import requests
import redis
from agno.agent import Agent
from agno.db.mongo import MongoDb
from agno.exceptions import ModelProviderError
from agno.models.openai import OpenAILike
from agno.models.deepseek import DeepSeek
from dotenv import load_dotenv
from openai.types.chat import ChatCompletion
from concurrent_log_handler import ConcurrentRotatingFileHandler



# ============ HTTP 连接池与模型调用 ============

_pedia_sessions: dict[str, requests.Session] = {}

def _get_pedia_session() -> requests.Session:
    url = os.getenv('OPENROUTER_BASE_URL', "").rstrip("/")
    key = os.getenv('OPENROUTER_API_KEY', "")

    if url not in _pedia_sessions:
        session = requests.Session()
        session.headers.update({
            "Authorization": f"Bearer {key}",
            "Connection": "keep-alive",
            "Content-Type": "application/json",
            "Accept": "application/json",
        })
        adapter = HTTPAdapter(
            pool_connections=1,
            pool_maxsize=200,
            max_retries=Retry(
                total=1,
                backoff_factor=0.05,
                status_forcelist=[429, 502, 503, 504],
                allowed_methods=["POST"],
                raise_on_status=False,
            ),
            pool_block=False,
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        _pedia_sessions[url] = session
    return _pedia_sessions[url]

@dataclass
class PediaOpenAILike(OpenAILike):
    local: bool = True

    def invoke(self, messages, assistant_message, response_format=None,
               tools=None, tool_choice=None, run_response=None, compress_tool_results=False):
        session = _get_pedia_session()


        url = (self.base_url or os.getenv('OPENROUTER_BASE_URL', "")).rstrip("/") + "/chat/completions"

        request_params = self.get_request_params(
            response_format=response_format, tools=tools,
            tool_choice=tool_choice, run_response=run_response,
        )
        formatted_messages = self._format_all_messages(messages, compress_tool_results)
        payload = {"model": self.id, "messages": formatted_messages, **request_params}

        resp = session.post(url, json=payload, timeout=120)

        if resp.status_code != 200:
            raise ModelProviderError(
                message=f"{resp.status_code} {resp.text[:2000]}",
                model_name=self.name,
                error=resp.text,
            )

        chat_completion = ChatCompletion.model_validate(resp.json())
        return self._parse_provider_response(chat_completion, response_format=response_format)


@dataclass
class PediaOpenRouter(PediaOpenAILike):
    id: str = "gpt-4o"
    name: str = "PediaOpenRouter"
    provider: str = "OpenRouter"
    api_key: Optional[str] = None
    base_url: str = "https://openrouter.ai/api/v1"
    local: bool = False

    def _parse_provider_response(self, response, response_format=None):
        model_response = super()._parse_provider_response(response, response_format)

        if response.choices and len(response.choices) > 0:
            resp_msg = response.choices[0].message
            if hasattr(resp_msg, "reasoning_details") and resp_msg.reasoning_details:
                if model_response.provider_data is None:
                    model_response.provider_data = {}
                model_response.provider_data["reasoning_details"] = resp_msg.reasoning_details
            elif hasattr(resp_msg, "model_extra"):
                extra = getattr(resp_msg, "model_extra", None)
                if extra and isinstance(extra, dict) and extra.get("reasoning_details"):
                    if model_response.provider_data is None:
                        model_response.provider_data = {}
                    model_response.provider_data["reasoning_details"] = extra["reasoning_details"]

        return model_response
    

def get_model(model_id: str = 'glm-5.1', extra_body: Dict[str, Any] = {}):
    load_dotenv()
    return PediaOpenRouter(id=model_id, extra_body=extra_body, max_retries=2)


disease_match_judge_agent = Agent(
    name="病名匹配裁判Agent",
    model=get_model(
        model_id=os.getenv("JUDGE_MODEL", "deepseek/deepseek-v4-flash"),
        extra_body={"reasoning": {"enabled": False, "effort": "high"}},
    ),
    instructions=JUDGE_SYSTEM_PROMPT,
    telemetry=False,
)


