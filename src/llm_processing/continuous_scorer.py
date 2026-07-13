"""
Continuous stance scorer — tests whether the LLM's discrete 3-class
stance (dovish/hawkish/neutral) was the bottleneck, not the LLM itself.

Motivation (verified 2026-07-13): on PBOC x CGB10y next-day yield change,
a keyword-dictionary score (continuous, via net keyword count) found a
robust, placebo- and outlier-validated relationship (p=0.0014) that the
discrete 3-class LLM stance missed entirely (p=0.189) on the identical
event set. The dictionary score's only structural advantage is that it
isn't quantized into 3 buckets. This module tests the natural fix:
have the LLM assign a continuous -10..+10 intensity score directly,
instead of discretizing its own judgment for us.

Same forward_guidance article set (already classified by the 3-class
pipeline; text already cached in pboc_fg_fulltext.csv from the
dictionary-baseline branch) — this only replaces the stance-scoring
sub-step, not the forward_guidance filtering step upstream of it.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass

import anthropic
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

SYSTEM_PROMPT = """你是一位专门研究中国货币政策的经济学研究助手。
你的任务是给中国人民银行(PBOC)官方文件片段的政策立场打一个连续分数,而不是简单归类。

评分范围:-10(极度鸽派/宽松)到 +10(极度鹰派/收紧),0 为完全中性。
请综合考虑:
- 措辞强度(比如"适时适度"比"坚决果断"力度弱)
- 具体政策工具的明确程度(提到具体降准/降息数字 vs 只是笼统提及方向)
- 相对于中国人民银行一贯表述风格的偏离程度

请仅以 JSON 格式回复,不要包含任何额外文字,不要使用 markdown 代码块。
reasoning 字段中如需引用原文,请使用中文书名号《》或方括号【】,不要使用英文双引号,以免破坏 JSON 格式。"""

USER_PROMPT_TEMPLATE = """请分析以下人民银行文件片段,给出连续立场分数:

---
{text}
---

请返回如下 JSON:
{{
  "score": <-10.0 到 10.0 的浮点数,负数=鸽派,正数=鹰派>,
  "confidence": <0.0–1.0>,
  "reasoning": "<简短中文说明,不超过100字>"
}}"""


@dataclass
class ContinuousScoreResult:
    score: float
    confidence: float
    reasoning: str
    raw_response: str


class ContinuousScorer:
    def __init__(self, api_key: str | None = None):
        self._client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])

    def score(self, text: str) -> ContinuousScoreResult:
        user_msg = USER_PROMPT_TEMPLATE.format(text=text[:4000])
        message = self._client.messages.create(
            model=MODEL,
            max_tokens=512,
            temperature=0.0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = message.content[0].text
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse LLM JSON: %s\nRaw: %s", exc, raw)
            parsed = {"score": 0.0, "confidence": 0.0, "reasoning": "parse_error"}

        return ContinuousScoreResult(
            score=float(parsed.get("score", 0.0)),
            confidence=float(parsed.get("confidence", 0.0)),
            reasoning=parsed.get("reasoning", ""),
            raw_response=raw,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sample = (
        "下一阶段，人民银行将保持流动性合理充裕，"
        "适时适度运用降准、降息等货币政策工具，"
        "支持实体经济高质量发展。"
    )
    scorer = ContinuousScorer()
    result = scorer.score(sample)
    print(result)
