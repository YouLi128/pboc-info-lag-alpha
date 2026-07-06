# Lost in Translation: LLM-Based Cross-Border Information Lag in PBOC Communications

NUS Master of Computing capstone project.

---

## Research Question

Does LLM-extracted *forward guidance* stance from PBOC communications predict
short-horizon price and volatility moves in offshore RMB (CNH) beyond what
same-day headline-driven reactions already capture?

---

## Hypotheses

| ID | Hypothesis |
|----|------------|
| **H1** | LLM-extracted forward-guidance stance has statistically significant predictive power (Granger causality / event-study) for short-horizon CNH returns, beyond a keyword-dictionary baseline. |
| **H2** | The predictive lag is larger for *non-headline* channels (press conference Q&A, regional branch remarks, footnote-level report language) than for headline policy actions (rate/RRR changes), which are already reflected in near-instant English wire coverage. |
| **H3** *(stretch)* | The lag has been shrinking over time, consistent with improving market efficiency in digesting Chinese-language monetary policy communication. |

---

## Pipeline

```
Stage 1 — Data Collection
  ├── PBOC Chinese-language texts   (src/scraping/pboc_scraper.py)
  │     press releases, Q&A transcripts, monetary policy reports
  └── Market prices                 (src/scraping/market_data_loader.py)
        CNH/USD  — Bloomberg/Wind manual export (CSV)
        BTC/USDT — Binance REST API (public, no key required)

Stage 2 — LLM Processing
  └── Stance classification         (src/llm_processing/stance_classifier.py)
        Input : raw Chinese text segment
        Output: segment_type {forward_guidance | descriptive | historical}
                stance         {dovish | hawkish | neutral}
                confidence     [0, 1]

Stage 3 — Signal Construction
  └── Surprise score + alignment    (src/signal_construction/surprise_score.py)
        Surprise_t = weighted_stance_t − rolling_baseline_{t-k}
        Align to market data at release timestamp + forward windows [5, 15, 30, 60 min]

Stage 4 — Hypothesis Testing
  └── Statistical analysis          (src/analysis/granger_test.py)
        Granger causality: surprise → CNH return (H1)
        Channel stratification      (H2)
        Rolling-window / Chow test  (H3)
```

---

## Repository Layout

```
pboc-info-lag-alpha/
├── config.yaml                     # central config (URLs, model, windows)
├── .env.example                    # copy to .env, fill API keys
├── requirements.txt
├── data/
│   ├── raw/                        # gitignored; scraped texts, price exports
│   └── processed/                  # gitignored; cleaned DataFrames
├── src/
│   ├── scraping/
│   │   ├── pboc_scraper.py         # PBOC listing + full-text fetcher
│   │   └── market_data_loader.py   # CNH CSV loader + Binance BTC fetcher
│   ├── llm_processing/
│   │   └── stance_classifier.py    # Anthropic API classification
│   ├── signal_construction/
│   │   └── surprise_score.py       # surprise score + market alignment
│   └── analysis/
│       └── granger_test.py         # Granger causality + event study
├── notebooks/
│   └── 00_pilot_exploration.ipynb  # (placeholder)
└── tests/
    ├── test_pboc_scraper.py
    └── test_surprise_score.py
```

---

## Setup

```bash
# 1. Clone
git clone https://github.com/YouLi128/pboc-info-lag-alpha.git
cd pboc-info-lag-alpha

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure secrets
cp .env.example .env
# Edit .env — add ANTHROPIC_API_KEY at minimum

# 5. Smoke-test the scraper (no API key needed)
python -m src.scraping.pboc_scraper

# 6. Smoke-test the Binance loader (no API key needed)
python -m src.scraping.market_data_loader

# 7. Run tests
pytest
```

### Market Data

- **CNH/USD**: Export 1-minute or 5-minute bars from Bloomberg (`USDCNH Curncy`)
  or Wind as CSV and place in `data/raw/cnh_<start>_<end>.csv`.
  Update `CNH_COLUMN_MAP` in `market_data_loader.py` to match your export columns.
- **BTC/USDT**: Fetched automatically from Binance — no manual step required.

---

## Status / Roadmap

> **Early-stage capstone scaffold — pending advisor sign-off and pilot test.**

- [x] Repo structure and module skeletons
- [x] PBOC scraper skeleton (connectivity; pagination and full-text are TODOs)
- [x] LLM stance classifier (prompt + Anthropic API call)
- [x] Surprise score construction and market alignment logic
- [x] Granger causality and event-study test stubs
- [ ] Validate PBOC scraper against live site (CSS selector verification)
- [ ] Collect pilot dataset: ~50–100 PBOC documents (2022–2024)
- [ ] Obtain CNH tick/bar data from Bloomberg/Wind
- [ ] Run pilot classification and check inter-rater reliability vs. human labels
- [ ] First pass Granger test on pilot data
- [ ] Expand to additional PBOC channels (press conference transcripts, branch statements)
- [ ] English wire timestamp collection for lag measurement
- [ ] Full sample analysis and write-up

---

## Citation / Acknowledgements

Project advised by [Advisor TBD], NUS School of Computing.

Key methodological references:
- Gürkaynak, Sack & Swanson (2005) — monetary policy surprise decomposition
- Hansen, McMahon & Prat (2018) — central bank communication and market reactions
- Miranda-Agrippino & Ricco (2021) — information effects of monetary policy

---

*This project is for academic research purposes only.*
