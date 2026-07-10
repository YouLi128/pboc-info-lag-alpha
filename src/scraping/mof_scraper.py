"""
MOF (财政部, Ministry of Finance) press release scraper.

Third policy-document source for the broad-market extension branch:
fiscal policy (tax cuts, fee reductions, budget/transfer announcements)
is a macro tool distinct from PBOC monetary policy and NDRC industrial
policy — completes the "各种政府银行文件" coverage the advisor asked for.

Mirrors pboc_scraper.py / ndrc_scraper.py conventions. Note: MOF articles
are hosted across many department subdomains (szs.mof.gov.cn,
kjs.mof.gov.cn, gss.mof.gov.cn, ...), which are individually less
reliable than the main site — fetch_article_text() tolerates failures
the same way classify_corpus.py already does downstream.

Output schema matches pboc_scraper.py / ndrc_scraper.py, with
"source": "mof".
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Generator

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

BASE_URL = "https://www.mof.gov.cn"

# 政策发布 channel. Verified 2026-07: pagination is index.htm (page 0),
# index_1.htm, index_2.htm, ... (20 pages at time of writing).
LISTING_PATH = "/zhengwuxinxi/zhengcefabu/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9",
}

REQUEST_DELAY = 2


def fetch_listing_page(page: int = 0) -> BeautifulSoup:
    if page == 0:
        url = f"{BASE_URL}{LISTING_PATH}index.htm"
    else:
        url = f"{BASE_URL}{LISTING_PATH}index_{page}.htm"

    logger.info("GET %s", url)
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or "utf-8"
    return BeautifulSoup(resp.text, "lxml")


def parse_listing(soup: BeautifulSoup, scraped_at: str) -> list[dict]:
    """
    Live MOF structure (verified 2026-07):
        <div class="xwfb_listerji">
          <ul class="xwfb_listbox">
            <li><a href="http://.../t20260703_....htm" title="...">...</a>
                <span>2026-07-03</span></li>
          </ul> ...
        </div>
    Article URLs are absolute (scattered across *.mof.gov.cn subdomains).
    """
    articles = []
    container = soup.select_one(".xwfb_listerji")
    items = container.select("li") if container else []

    if not items:
        logger.warning("No items matched '.xwfb_listerji li' — listing HTML may have changed.")

    for item in items:
        a_tag = item.find("a")
        if not a_tag:
            continue

        try:
            title = a_tag.get("title") or a_tag.get_text(strip=True)
            url = a_tag.get("href", "")
            span = item.find("span")
            published = span.get_text(strip=True) if span else ""

            articles.append(
                {
                    "title": title,
                    "url": url,
                    "published": published,
                    "scraped_at": scraped_at,
                    "source": "mof",
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping malformed item: %s", exc)

    return articles


def fetch_article_text(url: str) -> str:
    """Content container verified 2026-07: div.TRS_Editor (same TRS WCM marker as NDRC)."""
    time.sleep(REQUEST_DELAY)
    logger.info("Fetching article: %s", url)
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or "utf-8"
    soup = BeautifulSoup(resp.text, "lxml")

    content_div = soup.find("div", class_="TRS_Editor")
    if not content_div:
        logger.warning("Content div not found at %s", url)
        return ""

    return content_div.get_text(separator="\n", strip=True)


def scrape_listings(max_pages: int = 5, start_page: int = 0) -> Generator[dict, None, None]:
    scraped_at = datetime.now(timezone.utc).isoformat()

    for page in range(start_page, start_page + max_pages):
        logger.info("Scraping listing page %d / %d", page + 1, max_pages)
        try:
            soup = fetch_listing_page(page)
            articles = parse_listing(soup, scraped_at)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                logger.info("Page %d returned 404 — reached end of listing.", page)
                break
            raise

        if not articles:
            logger.info("No articles on page %d — stopping.", page)
            break

        yield from articles
        time.sleep(REQUEST_DELAY)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print("Fetching first listing page …")
    for article in scrape_listings(max_pages=1):
        print(article)
