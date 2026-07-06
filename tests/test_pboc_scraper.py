"""Unit tests for PBOC scraper helpers (no network calls)."""

from bs4 import BeautifulSoup

from src.scraping.pboc_scraper import parse_listing


def _make_soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")


def test_parse_listing_extracts_articles():
    html = """
    <html><body>
      <ul class="news_ul">
        <li><a href="/news/2024/01/article1.html">人民银行开展公开市场操作</a><span>2024-01-15</span></li>
        <li><a href="/news/2024/01/article2.html">货币政策执行报告</a><span>2024-01-20</span></li>
      </ul>
    </body></html>
    """
    soup = _make_soup(html)
    articles = parse_listing(soup, scraped_at="2024-01-21T00:00:00+00:00")
    assert len(articles) == 2
    assert articles[0]["title"] == "人民银行开展公开市场操作"
    assert articles[0]["url"].startswith("https://")
    assert articles[0]["published"] == "2024-01-15"


def test_parse_listing_empty_page():
    html = "<html><body><ul class='news_ul'></ul></body></html>"
    soup = _make_soup(html)
    articles = parse_listing(soup, scraped_at="2024-01-21T00:00:00+00:00")
    assert articles == []


def test_parse_listing_no_matching_selector():
    html = "<html><body><div>no list here</div></body></html>"
    soup = _make_soup(html)
    # Should warn but not raise
    articles = parse_listing(soup, scraped_at="2024-01-21T00:00:00+00:00")
    assert articles == []
