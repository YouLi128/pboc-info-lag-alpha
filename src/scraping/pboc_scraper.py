"""
PBOC press release scraper.

Fetches the Chinese-language press release listing from pbc.gov.cn and
extracts per-article metadata (title, URL, publish timestamp).  Full-text
extraction and pagination are marked as TODOs — this skeleton is enough to
validate connectivity and HTML structure before building out the full crawler.

Output schema (one dict per article):
    {
        "title":      str,   # Chinese headline
        "url":        str,   # absolute URL to full article
        "published":  str,   # raw date string from listing page (YYYY-MM-DD)
        "scraped_at": str,   # ISO-8601 UTC timestamp of this scrape run
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

# ---------------------------------------------------------------------------
# Constants — override via config.yaml / environment if needed
# ---------------------------------------------------------------------------

BASE_URL = "https://www.pbc.gov.cn"

# Press release listing page (one of several channels; add more in TODO below)
LISTING_PATH = "/goutongjiaoliu/113456/113469/index.html"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (research-bot; NUS capstone; contact: research@example.com)"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9",
}

REQUEST_DELAY = 2  # seconds between requests (polite crawl)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_listing_page(page: int = 0) -> BeautifulSoup:
    """
    Fetch one page of the press-release listing and return parsed HTML.

    Args:
        page: Zero-indexed page offset.  The PBOC site uses `index_{n}.html`
              naming for pages after the first — see TODO_PAGINATION below.
    """
    if page == 0:
        url = f"{BASE_URL}{LISTING_PATH}"
    else:
        # Verified 2026-07: PBOC paginates as /goutongjiaoliu/113456/113469/11040-{n}.html
        # where 11040 is the channel ID for this listing and n starts at 2.
        url = f"{BASE_URL}/goutongjiaoliu/113456/113469/11040-{page + 1}.html"

    logger.info("GET %s", url)
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or "utf-8"
    return BeautifulSoup(resp.text, "lxml")


def parse_listing(soup: BeautifulSoup, scraped_at: str) -> list[dict]:
    """
    Extract article stubs from a listing page.

    Live PBOC structure (verified 2026-07):
        <font class="newslist_style">
          <a href="/goutongjiaoliu/.../index.html" ...>title</a>
        </font>
        <span class="hui12">YYYY-MM-DD</span>
    """
    articles = []

    # Each article link sits inside a <font class="newslist_style"> element.
    items = soup.select("font.newslist_style")

    if not items:
        logger.warning(
            "No items matched selector 'font.newslist_style' — listing HTML may have changed."
        )

    for item in items:
        try:
            a_tag = item.find("a")
            if not a_tag:
                continue

            title = a_tag.get("title") or a_tag.get_text(strip=True)
            href = a_tag.get("href", "")
            url = href if href.startswith("http") else f"{BASE_URL}{href}"

            # Date is in the next sibling <span class="hui12">
            date_span = item.find_next_sibling("span", class_="hui12")
            published = date_span.get_text(strip=True) if date_span else ""

            articles.append(
                {
                    "title": title,
                    "url": url,
                    "published": published,
                    "scraped_at": scraped_at,
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping malformed item: %s", exc)

    return articles


def fetch_article_text(url: str) -> str:
    """
    Fetch and return the full Chinese-language body text of a single article.

    TODO_FULLTEXT: Identify the correct content container selector for article
    pages (likely "div#zoom" or "div.article" based on historical PBOC markup).
    Some documents are PDFs — add PDF extraction (e.g. pdfminer) here.
    """
    time.sleep(REQUEST_DELAY)
    logger.info("Fetching article: %s", url)
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or "utf-8"
    soup = BeautifulSoup(resp.text, "lxml")

    # TODO_FULLTEXT: verify / replace this selector.
    content_div = soup.find("div", id="zoom") or soup.find("div", class_="article")
    if not content_div:
        logger.warning("Content div not found at %s", url)
        return ""

    return content_div.get_text(separator="\n", strip=True)


def scrape_listings(max_pages: int = 5, start_page: int = 0) -> Generator[dict, None, None]:
    """
    Yield article stubs across multiple listing pages.

    Args:
        max_pages:  Stop after this many pages.
        start_page: Zero-indexed page to start from (0 = most recent).
    """
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


# ---------------------------------------------------------------------------
# CLI entry point for quick manual testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print("Fetching first listing page …")
    for article in scrape_listings(max_pages=1):
        print(article)
