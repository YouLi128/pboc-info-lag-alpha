"""
NDRC (国家发展和改革委员会) press release scraper.

Second corpus source for the broad-market extension branch: alongside
PBOC monetary policy communications, NDRC news releases cover industrial
policy, price controls, and macro guidance that the A-share market reacts
to directly (no cross-border translation lag story here — this is a
separate "policy text -> broad index" prediction branch, not part of the
core CNH information-lag hypothesis).

Mirrors the structure of pboc_scraper.py so downstream classification /
alignment code can treat any source corpus interchangeably.

Output schema (one dict per article):
    {
        "title":      str,   # Chinese headline
        "url":        str,   # absolute URL to full article
        "published":  str,   # raw date string from listing page (YYYY/MM/DD)
        "scraped_at": str,   # ISO-8601 UTC timestamp of this scrape run
        "source":     str,   # "ndrc" — lets downstream code tag corpus origin
    }
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Generator

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

BASE_URL = "https://www.ndrc.gov.cn"

# 新闻发布 channel — press releases, press conferences, leader activities.
# Verified 2026-07: pagination is index.html (page 0), index_1.html, index_2.html, ...
# going back to 2017-11 (40 pages at time of writing).
LISTING_PATH = "/xwdt/xwfb/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (research-bot; NUS capstone; contact: research@example.com)"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9",
}

REQUEST_DELAY = 2  # seconds between requests (polite crawl)


def fetch_listing_page(page: int = 0) -> BeautifulSoup:
    """Fetch one page of the NDRC press-release listing and return parsed HTML."""
    if page == 0:
        url = f"{BASE_URL}{LISTING_PATH}index.html"
    else:
        url = f"{BASE_URL}{LISTING_PATH}index_{page}.html"

    logger.info("GET %s", url)
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or "utf-8"
    return BeautifulSoup(resp.text, "lxml")


def parse_listing(soup: BeautifulSoup, scraped_at: str) -> list[dict]:
    """
    Extract article stubs from a listing page.

    Live NDRC structure (verified 2026-07):
        <ul class="u-list">
          <li><a href="./202607/t20260703_1406245.html" title="...">...</a>
              <span>2026/07/03</span></li>
          <li class="empty"></li>   # padding rows with no article — skip
          ...
        </ul>
    """
    articles = []
    items = soup.select("ul.u-list li")

    if not items:
        logger.warning(
            "No items matched selector 'ul.u-list li' — listing HTML may have changed."
        )

    for item in items:
        a_tag = item.find("a")
        if not a_tag:
            continue  # padding <li class="empty">

        try:
            title = a_tag.get("title") or a_tag.get_text(strip=True)
            href = a_tag.get("href", "")
            if href.startswith("http"):
                url = href
            else:
                # Relative to the listing directory (href starts with "./" or "../").
                base_dir = f"{BASE_URL}{LISTING_PATH}"
                url = requests.compat.urljoin(base_dir, href)

            span = item.find("span")
            published = span.get_text(strip=True) if span else ""

            articles.append(
                {
                    "title": title,
                    "url": url,
                    "published": published,
                    "scraped_at": scraped_at,
                    "source": "ndrc",
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping malformed item: %s", exc)

    return articles


def fetch_article_text(url: str) -> str:
    """
    Fetch and return the full Chinese-language body text of a single article.

    Content container verified 2026-07: div.TRS_Editor (standard TRS WCM
    CMS marker used across most Chinese central-government sites).
    """
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
    """Yield article stubs across multiple listing pages."""
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
