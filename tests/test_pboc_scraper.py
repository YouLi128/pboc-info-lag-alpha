"""Unit tests for PBOC scraper helpers (no network calls)."""

from bs4 import BeautifulSoup

from src.scraping.pboc_scraper import parse_listing


def _make_soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")


def test_parse_listing_extracts_articles():
    # Mirrors actual PBOC HTML structure (verified 2026-07):
    # <font class="newslist_style"><a title="..." href="...">title</a></font>
    # <span class="hui12">YYYY-MM-DD</span>
    html = """
    <html><body>
      <table><tr><td>
        <font class="newslist_style">
          <a href="/goutongjiaoliu/113456/113469/2024011500001/index.html"
             title="人民银行开展公开市场操作">人民银行开展公开市场操作</a>
        </font>
        <span class="hui12">2024-01-15</span>
      </td></tr><tr><td>
        <font class="newslist_style">
          <a href="/goutongjiaoliu/113456/113469/2024012000002/index.html"
             title="货币政策执行报告">货币政策执行报告</a>
        </font>
        <span class="hui12">2024-01-20</span>
      </td></tr></table>
    </body></html>
    """
    soup = _make_soup(html)
    articles = parse_listing(soup, scraped_at="2024-01-21T00:00:00+00:00")
    assert len(articles) == 2
    assert articles[0]["title"] == "人民银行开展公开市场操作"
    assert articles[0]["url"].startswith("https://")
    assert articles[0]["published"] == "2024-01-15"


def test_parse_listing_empty_page():
    html = "<html><body><table></table></body></html>"
    soup = _make_soup(html)
    articles = parse_listing(soup, scraped_at="2024-01-21T00:00:00+00:00")
    assert articles == []


def test_parse_listing_no_matching_selector():
    html = "<html><body><div>no list here</div></body></html>"
    soup = _make_soup(html)
    # Should warn but not raise
    articles = parse_listing(soup, scraped_at="2024-01-21T00:00:00+00:00")
    assert articles == []
