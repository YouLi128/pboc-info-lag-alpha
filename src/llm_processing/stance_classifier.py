"""
LLM-based stance classifier for PBOC communications.

Classifies each document into:
  - segment type: "forward_guidance" | "descriptive" | "historical"
  - stance:       "dovish" | "hawkish" | "neutral"
  - confidence:   float [0, 1]

Uses structured output via the Anthropic API.  The prompt is in Chinese to
leverage the model's native comprehension of PBOC register and terminology.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

import anthropic
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

# ---------------------------------------------------------------------------
# Classification schema
# ---------------------------------------------------------------------------

SEGMENT_TYPES = ("forward_guidance", "descriptive", "historical")
STANCES = ("dovish", "hawkish", "neutral")


@dataclass
class ClassificationResult:
    segment_type: str   # one of SEGMENT_TYPES
    stance: str         # one of STANCES
    confidence: float   # model self-reported confidence [0, 1]
    reasoning: str      # brief Chinese-language rationale
    raw_response: str   # full model output for audit


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """你是一位专门研究中国货币政策的经济学研究助手。
你的任务是分析中国人民银行（PBOC）的官方文件片段，并按照以下维度进行分类：

1. 内容类型（segment_type）：
   - forward_guidance：包含对未来货币政策方向、利率走势或流动性管理的前瞻性指引
   - descriptive：描述当前经济形势、货币政策执行情况等现状
   - historical：回顾历史数据或过去政策行动

2. 政策立场（stance）—— 仅当 segment_type 为 forward_guidance 时填写：
   - dovish：偏鸽派，暗示宽松或降息预期
   - hawkish：偏鹰派，暗示收紧或升息预期
   - neutral：中性，未明确倾向
   若非 forward_guidance，请填写 neutral。

3. 置信度（confidence）：0.0–1.0，反映你对分类结果的把握程度。

请仅以 JSON 格式回复，不要包含任何额外文字，不要使用 markdown 代码块。
reasoning 字段中如需引用原文，请使用中文书名号《》或方括号【】，不要使用英文双引号，以免破坏 JSON 格式。"""

USER_PROMPT_TEMPLATE = """请分析以下人民银行文件片段：

---
{text}
---

请返回如下 JSON：
{{
  "segment_type": "<forward_guidance|descriptive|historical>",
  "stance": "<dovish|hawkish|neutral>",
  "confidence": <0.0–1.0>,
  "reasoning": "<简短中文说明，不超过100字>"
}}"""


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class StanceClassifier:
    def __init__(self, api_key: str | None = None):
        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ["ANTHROPIC_API_KEY"]
        )

    def classify(self, text: str) -> ClassificationResult:
        """
        Classify a single PBOC text segment.

        Args:
            text: Raw Chinese-language excerpt from a PBOC document.

        Returns:
            ClassificationResult with parsed fields.
        """
        user_msg = USER_PROMPT_TEMPLATE.format(text=text[:4000])  # guard token budget

        message = self._client.messages.create(
            model=MODEL,
            max_tokens=512,
            temperature=0.0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )

        raw = message.content[0].text
        # Strip markdown code fences if the model wraps its JSON output
        import re as _re
        cleaned = _re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse LLM JSON: %s\nRaw: %s", exc, raw)
            # Return a low-confidence neutral result rather than crashing.
            parsed = {
                "segment_type": "descriptive",
                "stance": "neutral",
                "confidence": 0.0,
                "reasoning": "parse_error",
            }

        return ClassificationResult(
            segment_type=parsed.get("segment_type", "descriptive"),
            stance=parsed.get("stance", "neutral"),
            confidence=float(parsed.get("confidence", 0.0)),
            reasoning=parsed.get("reasoning", ""),
            raw_response=raw,
        )

    def classify_batch(self, texts: list[str]) -> list[ClassificationResult]:
        """Classify a list of segments sequentially (rate-limit friendly)."""
        return [self.classify(t) for t in texts]


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sample = (
        "下一阶段，人民银行将保持流动性合理充裕，"
        "适时适度运用降准、降息等货币政策工具，"
        "支持实体经济高质量发展。"
    )
    clf = StanceClassifier()
    result = clf.classify(sample)
    print(result)
