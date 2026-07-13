"""
Keyword-dictionary stance baseline — the comparison H1 always specified
("超越关键词词典基准") but was never built until now.

Standard net-keyword-count approach (same family as Loughran-McDonald
for financial sentiment): count dovish vs hawkish terms in each article,
score = (dovish_count - hawkish_count) / (dovish_count + hawkish_count + 1).
Denominator +1 avoids divide-by-zero and dampens the score for articles
with very few keyword hits (low-confidence by construction, mirroring
how the LLM's confidence score does the same job).

Keywords are standard PBOC communication vocabulary, not tuned to this
project's data — tuning the dictionary to match known outcomes would
defeat the point of a baseline.
"""

from __future__ import annotations

import re

import pandas as pd

DOVISH_TERMS = [
    "降准", "降息", "宽松", "合理充裕", "逆周期调节", "跨周期调节",
    "稳增长", "支持实体经济", "降低融资成本", "降低社会融资成本",
    "加大力度", "定向降准", "再贷款", "再贴现", "稳健偏松",
    "加大逆周期", "扩大内需", "减费让利", "让利实体经济",
]

HAWKISH_TERMS = [
    "收紧", "从紧", "去杠杆", "防风险", "防范风险", "防范化解重大金融风险",
    "上调", "严格监管", "打击投机", "该出手就出手", "超调风险",
    "坚决防范", "遏制投机", "去泡沫", "稳健中性", "管控风险",
    "从严监管", "压降",
]


def keyword_score(text: str) -> dict:
    dovish_count = sum(text.count(term) for term in DOVISH_TERMS)
    hawkish_count = sum(text.count(term) for term in HAWKISH_TERMS)
    score = (dovish_count - hawkish_count) / (dovish_count + hawkish_count + 1)
    return {"dovish_count": dovish_count, "hawkish_count": hawkish_count, "dict_score": score}


def score_corpus(fulltext_csv: str) -> pd.DataFrame:
    df = pd.read_csv(fulltext_csv)
    df["text"] = df["text"].fillna("")
    scores = df["text"].apply(keyword_score).apply(pd.Series)
    return pd.concat([df, scores], axis=1)


if __name__ == "__main__":
    df = score_corpus("data/processed/pboc_fg_fulltext.csv")
    print(df[["published", "dovish_count", "hawkish_count", "dict_score"]].describe())
    print(f"\n{len(df)} articles scored")
    print(f"Zero-keyword articles (score defaults toward 0): {((df['dovish_count']==0) & (df['hawkish_count']==0)).sum()}")
    df.to_csv("data/processed/pboc_fg_dict_scored.csv", index=False)
