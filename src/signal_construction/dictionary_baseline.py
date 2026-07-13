"""
Keyword-dictionary stance baseline — the comparison H1 always specified
("超越关键词词典基准") but was never built until now.

Standard net-keyword-count approach (same family as Loughran-McDonald
for financial sentiment): count dovish vs hawkish terms in each article,
score = (dovish_count - hawkish_count) / (dovish_count + hawkish_count + 1).
Denominator +1 avoids divide-by-zero and dampens the score for articles
with very few keyword hits (low-confidence by construction, mirroring
how the LLM's confidence score does the same job).

Keywords are standard vocabulary for each source's policy domain, not
tuned to this project's data — tuning the dictionary to match known
outcomes would defeat the point of a baseline. PBOC's vocabulary is
monetary-policy specific (rate cuts, RRR, liquidity); NDRC and MOF use
different vocabularies (industrial policy, fiscal policy respectively)
because their mandates don't map onto dovish/hawkish monetary language
— same rationale as the source-specific LLM prompts in
stance_classifier.py. Applying the PBOC dictionary as-is to NDRC/MOF
text would silently return near-zero scores for almost every article
(verified: MOF's median keyword count was 0 under the PBOC dictionary)
purely from vocabulary mismatch, not an absence of signal.
"""

from __future__ import annotations

import re

import pandas as pd

PBOC_DOVISH_TERMS = [
    "降准", "降息", "宽松", "合理充裕", "逆周期调节", "跨周期调节",
    "稳增长", "支持实体经济", "降低融资成本", "降低社会融资成本",
    "加大力度", "定向降准", "再贷款", "再贴现", "稳健偏松",
    "加大逆周期", "扩大内需", "减费让利", "让利实体经济",
]

PBOC_HAWKISH_TERMS = [
    "收紧", "从紧", "去杠杆", "防风险", "防范风险", "防范化解重大金融风险",
    "上调", "严格监管", "打击投机", "该出手就出手", "超调风险",
    "坚决防范", "遏制投机", "去泡沫", "稳健中性", "管控风险",
    "从严监管", "压降",
]

NDRC_DOVISH_TERMS = [
    "扩大投资", "放松管制", "降价", "减税降费", "支持", "促进发展",
    "加大投资力度", "稳增长", "优化营商环境", "扩大内需", "民间投资",
    "简政放权", "培育壮大", "加快推进", "予以支持", "鼓励",
]

NDRC_HAWKISH_TERMS = [
    "加强管控", "提高价格", "收紧审批", "去产能", "去杠杆", "严格监管",
    "遏制", "清理整顿", "从严审批", "限制", "整治", "规范秩序", "淘汰落后",
]

MOF_DOVISH_TERMS = [
    "减税", "降费", "减税降费", "财政支出", "财政贴息", "贴息", "补贴",
    "优惠政策", "税收优惠", "支持", "扩大内需", "专项债", "转移支付",
    "减免", "加大投入", "予以支持",
]

MOF_HAWKISH_TERMS = [
    "加税", "削减", "收紧", "从严征管", "强化征管", "从紧", "压降",
    "清理规范", "从严监管", "取消优惠", "严格管理", "规范管理",
]

TERM_SETS = {
    "pboc": (PBOC_DOVISH_TERMS, PBOC_HAWKISH_TERMS),
    "ndrc": (NDRC_DOVISH_TERMS, NDRC_HAWKISH_TERMS),
    "mof": (MOF_DOVISH_TERMS, MOF_HAWKISH_TERMS),
}


def keyword_score(text: str, source: str = "pboc") -> dict:
    dovish_terms, hawkish_terms = TERM_SETS[source]
    dovish_count = sum(text.count(term) for term in dovish_terms)
    hawkish_count = sum(text.count(term) for term in hawkish_terms)
    score = (dovish_count - hawkish_count) / (dovish_count + hawkish_count + 1)
    return {"dovish_count": dovish_count, "hawkish_count": hawkish_count, "dict_score": score}


def score_corpus(fulltext_csv: str, source: str = "pboc") -> pd.DataFrame:
    df = pd.read_csv(fulltext_csv)
    df["text"] = df["text"].fillna("")
    scores = df["text"].apply(lambda t: keyword_score(t, source=source)).apply(pd.Series)
    return pd.concat([df, scores], axis=1)


if __name__ == "__main__":
    df = score_corpus("data/processed/pboc_fg_fulltext.csv")
    print(df[["published", "dovish_count", "hawkish_count", "dict_score"]].describe())
    print(f"\n{len(df)} articles scored")
    print(f"Zero-keyword articles (score defaults toward 0): {((df['dovish_count']==0) & (df['hawkish_count']==0)).sum()}")
    df.to_csv("data/processed/pboc_fg_dict_scored.csv", index=False)
