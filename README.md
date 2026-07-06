# Lost in Translation
## LLM-Based Cross-Border Information Lag in PBOC Communications

NUS Master of Computing 毕业论文项目。

---

## 研究问题

利用大语言模型从中国人民银行（PBOC）中文官方文件中提取"前瞻性指引"立场，
研究这一信息是否能在离岸人民币（CNH）市场中预测短期价格/波动率变动——
超越英文电报已捕捉到的同日反应，即验证中英文信息处理存在时滞。

---

## 研究假设

| 编号 | 假设 |
|------|------|
| **H1** | LLM 提取的前瞻性指引立场对 CNH 短期收益率具有统计显著的预测力（格兰杰因果 / 事件研究法），超越关键词词典基准 |
| **H2** | 非头条渠道（发布会问答、地区分行表态、报告脚注措辞）的预测滞后大于头条政策行动（调息/降准），因为后者已被英文电报近实时覆盖 |
| **H3** *(stretch)* | 该滞后随时间收窄，与市场消化中文货币政策信息的效率提升一致 |

---

## 四阶段流程

```
第一阶段 — 数据采集
  ├── PBOC 中文语料          src/scraping/pboc_scraper.py
  │     新闻发布、问答实录、货币政策报告
  └── 市场价格              src/scraping/market_data_loader.py
        CNH/USD — Bloomberg/Wind 手动导出 CSV
        BTC/USDT — Binance 公开 REST API（无需密钥）

第二阶段 — LLM 处理
  └── 立场分类              src/llm_processing/stance_classifier.py
        输入：中文文本片段
        输出：segment_type  {forward_guidance | descriptive | historical}
              stance        {dovish | hawkish | neutral}
              confidence    [0, 1]
              reasoning     简短中文说明

第三阶段 — 信号构建
  └── 惊喜度 + 时间对齐      src/signal_construction/surprise_score.py
        Surprise_t = 加权立场分_t − 滚动均值基线_{t-k}
        对齐至发布时间戳后各时间窗口收益率 [5, 15, 30, 60 分钟]

第四阶段 — 假设检验
  └── 统计分析              src/analysis/granger_test.py
        格兰杰因果检验（H1）
        渠道分层事件研究（H2）
        滚动窗口 / Chow 检验（H3）
```

---

## 目录结构

```
pboc-info-lag-alpha/
├── config.yaml                        # 统一配置（URL、模型、时间窗口等）
├── .env.example                       # 复制为 .env 并填入 API 密钥
├── requirements.txt
├── data/
│   ├── raw/                           # 已 gitignore；存放原始语料和价格导出
│   └── processed/                     # 已 gitignore；存放清洗后的 DataFrame
├── src/
│   ├── scraping/
│   │   ├── pboc_scraper.py            # 人民银行列表页 + 全文抓取
│   │   └── market_data_loader.py      # CNH CSV 读取 + Binance BTC 拉取
│   ├── llm_processing/
│   │   └── stance_classifier.py       # Claude API 中文立场分类
│   ├── signal_construction/
│   │   └── surprise_score.py          # 惊喜度计算 + 市场收益率对齐
│   └── analysis/
│       └── granger_test.py            # 格兰杰检验 + 事件研究
├── notebooks/
│   └── 00_pilot_exploration.ipynb     # 探索性分析（占位）
└── tests/
    ├── test_pboc_scraper.py
    └── test_surprise_score.py
```

---

## 环境配置

```bash
# 1. 克隆仓库
git clone https://github.com/YouLi128/pboc-info-lag-alpha.git
cd pboc-info-lag-alpha

# 2. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置密钥
cp .env.example .env
# 编辑 .env，填入 ANTHROPIC_API_KEY=sk-ant-...
```

### 市场数据说明

- **CNH/USD**：从 Bloomberg（`USDCNH Curncy`）或 Wind 手动导出分钟级 CSV，
  放入 `data/raw/`，并按需更新 `market_data_loader.py` 中的 `CNH_COLUMN_MAP`
- **BTC/USDT**：自动从 Binance 拉取，无需手动操作

---

## 冒烟测试

```bash
# 测试人民银行爬虫（无需 API 密钥）
python -m src.scraping.pboc_scraper

# 测试 Binance 行情拉取（无需 API 密钥）
python -m src.scraping.market_data_loader

# 测试 LLM 立场分类（需要 ANTHROPIC_API_KEY）
python -m src.llm_processing.stance_classifier

# 运行单元测试
pytest
```

### 已验证输出示例

**爬虫**（2026-07 实测，人民银行官网）：
```
{'title': '中国人民银行行长潘功胜出席国际清算银行行长例会及年度股东大会',
 'url': 'https://www.pbc.gov.cn/goutongjiaoliu/...', 'published': '2026-06-28'}
```

**LLM 分类**（测试文本：关于"适时适度运用降准、降息"的前瞻性表述）：
```
segment_type = forward_guidance
stance       = dovish
confidence   = 0.95
reasoning    = 文段以【下一阶段】开篇，明确指向未来政策方向…直接暗示降准降息预期，立场明显偏鸽派
```

---

## 进度 / 路线图

> **早期脚手架阶段 — 待导师签字确认及 pilot 测试。**

- [x] 项目结构与模块骨架
- [x] PBOC 爬虫（列表页采集、翻页、CSS 选择器已对线上页面验证）
- [x] LLM 立场分类（中文 prompt + Claude API，JSON 解析已处理）
- [x] 惊喜度构建与市场收益率对齐逻辑
- [x] 格兰杰检验与事件研究检验桩
- [x] 单元测试（6/6 通过）
- [ ] 全文抓取（单篇文章正文提取，CSS 选择器待验证）
- [ ] 采集 pilot 语料：2022–2024 年约 50–100 篇 PBOC 文件
- [ ] 从 Bloomberg/Wind 获取 CNH tick/分钟级数据
- [ ] 运行 pilot 分类，与人工标注对比一致性
- [ ] 完成 pilot 阶段格兰杰检验
- [ ] 扩展至更多渠道（发布会实录、地区分行表态）
- [ ] 英文电报时间戳采集（用于量化时滞）
- [ ] 全样本分析与论文写作

---

## 参考文献

- Gürkaynak, Sack & Swanson (2005) — 货币政策惊喜度分解
- Hansen, McMahon & Prat (2018) — 央行沟通与市场反应
- Miranda-Agrippino & Ricco (2021) — 货币政策的信息效应

---

*本项目仅供学术研究使用。*
*指导教师：[待定]，新加坡国立大学计算学院。*
